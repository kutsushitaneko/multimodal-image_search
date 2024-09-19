import os
import io
from typing import Union, List
import ftfy, html, re
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, BatchFeature
import oracledb
from dotenv import load_dotenv, find_dotenv
import json
import oci
from oci.config import from_file
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import EmbedTextDetails, OnDemandServingMode

_ = load_dotenv(find_dotenv())

IMAGES_PER_PAGE = 16

# データベース接続情報
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
dsn = os.getenv("DB_DSN")

# Japanese Stable CLIPモデルのロード
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "stabilityai/japanese-stable-clip-vit-l-16"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

# OCI設定
CONFIG_PROFILE = "DEFAULT"
config = from_file('~/.oci/config', CONFIG_PROFILE)
compartment_id = os.getenv("OCI_COMPARTMENT_ID")
model_id = "cohere.embed-multilingual-v3.0"
generative_ai_inference_client = GenerativeAiInferenceClient(config=config, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))



def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def tokenize(
    texts: Union[str, List[str]],
    max_seq_len: int = 77,
):
    if isinstance(texts, str):
        texts = [texts]
    texts = [whitespace_clean(basic_clean(text)) for text in texts]

    inputs = tokenizer(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[tokenizer.bos_token_id] + ids for ids in inputs["input_ids"]]
    attention_mask = [[1] + am for am in inputs["attention_mask"]]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    return BatchFeature(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }
    )

def compute_text_embeddings(text):
    """
    テキストの埋め込みベクトルを計算する関数。

    Args:
        text (str or List[str]): 埋め込みを計算するテキスト。単一の文字列または文字列のリスト。

    Returns:
        torch.Tensor: 正規化されたテキストの埋め込みベクトル。

    処理の流れ:
    1. 入力が単一の文字列の場合、リストに変換。
    2. テキストをトークン化。
    3. モデルを使用してテキスト特徴量を抽出。
    4. 特徴量ベクトルを正規化。
    5. 不要なメモリを解放。
    6. CPUに移動し、勾配計算を無効化して結果を返す。
    """
    if isinstance(text, str):
        text = [text]
    text = tokenize(texts=text)
    text_features = model.get_text_features(**text.to(device))
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    del text
    return text_features.cpu().detach()

def compute_image_embeddings(image):
    """
    画像の埋め込みベクトルを計算する関数。

    Args:
        image (PIL.Image.Image): 埋め込みを計算する画像。

    Returns:
        torch.Tensor: 正規化された画像の埋め込みベクトル。

    処理の流れ:
    1. 画像をモデルの入力形式に変換し、デバイスに移動。
    2. 勾配計算を無効化して、モデルを使用して画像特徴量を抽出。
    3. 特徴量ベクトルを正規化。
    4. 不要なメモリを解放。
    5. CPUに移動し、勾配計算を無効化して結果を返す。
    """
    image = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    del image
    return image_features.cpu().detach()

def get_image_data(image_id):
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    cursor.execute("SELECT image_data FROM IMAGES WHERE image_id = :image_id", {'image_id': image_id})
    image_data = cursor.fetchone()[0].read()

    cursor.close()
    connection.close()

    return image_data

def load_initial_images(page=1):
    offset = (page - 1) * IMAGES_PER_PAGE
    results = get_latest_images(limit=IMAGES_PER_PAGE, offset=offset)
    images = []
    image_info = []
    for image_id, file_name, generation_prompt in results:
        image_data = get_image_data(image_id)
        images.append(Image.open(io.BytesIO(image_data)))
        image_info.append({
            'file_name': file_name,
            'generation_prompt': generation_prompt,
            'vector_distance': 'N/A'
        })
    return images, image_info

def get_latest_images(limit=16, offset=0):
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT i.image_id, i.file_name, i.generation_prompt
        FROM IMAGES i
        ORDER BY i.upload_date DESC
        OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
    """, {'limit': limit, 'offset': offset})

    results = cursor.fetchall()
    
    # LOBオブジェクトを文字列に変換
    processed_results = []
    for row in results:
        image_id, file_name, generation_prompt = row
        processed_results.append((
            image_id,
            file_name,
            generation_prompt.read() if generation_prompt else None
        ))

    cursor.close()
    connection.close()

    return processed_results

def search_images(query, search_method, search_target, limit=16):
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    if search_method == "自然言語ベクトル検索":
        if search_target == "画像":
            embedding_json = json.dumps(compute_text_embeddings(query).tolist()[0])
            cursor.execute("""
                SELECT i.image_id, i.file_name, i.generation_prompt,
                       cie.embedding <#> :query_embedding as vector_distance
                FROM CURRENT_IMAGE_EMBEDDINGS cie
                JOIN IMAGES i ON cie.image_id = i.image_id
                ORDER BY vector_distance
                FETCH FIRST :limit ROWS ONLY
            """, {'query_embedding': embedding_json, 'limit': limit})
        elif search_target == "キャプション":
            embed_text_detail = EmbedTextDetails()
            embed_text_detail.serving_mode = OnDemandServingMode(model_id=model_id)
            embed_text_detail.inputs = [query]
            embed_text_detail.truncate = "NONE"
            embed_text_detail.compartment_id = compartment_id
            embed_text_detail.is_echo = False
            embed_text_detail.input_type = "SEARCH_QUERY"

            embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)
            embedding_json = json.dumps(embed_text_response.data.embeddings[0])
            
            cursor.execute("""
                SELECT i.image_id, i.file_name, i.generation_prompt,
                       id.embedding <#> :query_embedding as vector_distance
                FROM IMAGE_DESCRIPTIONS id
                JOIN IMAGES i ON id.image_id = i.image_id
                ORDER BY vector_distance
                FETCH FIRST :limit ROWS ONLY
            """, {'query_embedding': embedding_json, 'limit': limit})
        elif search_target == "プロンプト":
            embed_text_detail = EmbedTextDetails()
            embed_text_detail.serving_mode = OnDemandServingMode(model_id=model_id)
            embed_text_detail.inputs = [query]
            embed_text_detail.truncate = "NONE"
            embed_text_detail.compartment_id = compartment_id
            embed_text_detail.is_echo = False
            embed_text_detail.input_type = "SEARCH_QUERY"

            embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)
            embedding_json = json.dumps(embed_text_response.data.embeddings[0])
            
            cursor.execute("""
                SELECT i.image_id, i.file_name, i.generation_prompt,
                       i.prompt_embedding <#> :query_embedding as vector_distance
                FROM IMAGES i
                ORDER BY vector_distance
                FETCH FIRST :limit ROWS ONLY
            """, {'query_embedding': embedding_json, 'limit': limit})
    elif search_method == "自然言語全文検索":
        if search_target == "キャプション":
            cursor.execute("""
                SELECT i.image_id, i.file_name, i.generation_prompt, score(1) as relevance
                FROM IMAGES i
                JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
                WHERE CONTAINS(id.description, :query, 1) > 0
                ORDER BY relevance DESC
                FETCH FIRST :limit ROWS ONLY
            """, {'query': query, 'limit': limit})
        elif search_target == "プロンプト":
            cursor.execute("""
                SELECT image_id, file_name, generation_prompt, score(1) as relevance
                FROM IMAGES
                WHERE CONTAINS(generation_prompt, :query, 1) > 0
                ORDER BY relevance DESC
                FETCH FIRST :limit ROWS ONLY
            """, {'query': query, 'limit': limit})
    elif search_method == "画像ベクトル検索":
        embedding_json = json.dumps(compute_image_embeddings(query).tolist()[0])
        cursor.execute("""
            SELECT i.image_id, i.file_name, i.generation_prompt,
                   cie.embedding <#> :query_embedding as vector_distance
            FROM CURRENT_IMAGE_EMBEDDINGS cie
            JOIN IMAGES i ON cie.image_id = i.image_id
            ORDER BY vector_distance
            FETCH FIRST :limit ROWS ONLY
        """, {'query_embedding': embedding_json, 'limit': limit})
    else:
        raise ValueError("Invalid search method")

    results = cursor.fetchall()
    
    # LOBオブジェクトを文字列に変換
    processed_results = []
    for row in results:
        if search_method == "自然言語全文検索":
            image_id, file_name, generation_prompt, relevance = row
            vector_distance = relevance  # OracleTextのスコアをベクトル距離として使用
        else:
            image_id, file_name, generation_prompt, vector_distance = row
        processed_results.append((
            image_id,
            file_name,
            generation_prompt.read() if generation_prompt else None,
            vector_distance
        ))

    cursor.close()
    connection.close()

    return processed_results



def search(query, search_method, search_target, page=1):
    results = search_images(query, search_method, search_target, limit=16)
    images = []
    image_info = []
    for image_id, file_name, generation_prompt, vector_distance in results:
        image_data = get_image_data(image_id)
        images.append(Image.open(io.BytesIO(image_data)))
        image_info.append({
            'file_name': file_name,
            'generation_prompt': generation_prompt,
            'vector_distance': vector_distance if vector_distance is not None else 'N/A'
        })
    return images, image_info



with gr.Blocks(title="画像検索") as demo:
    image_info_state = gr.State([])
    current_page = gr.State(1)

    gr.Markdown("# マルチモーダル画像検索")
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=3):
                    search_method = gr.Radio(
                        ["自然言語ベクトル検索", "自然言語全文検索", "画像ベクトル検索"],
                        label="検索方法",
                        value="自然言語ベクトル検索"
                    )
                with gr.Column(scale=2):
                    search_target = gr.Radio(
                        ["画像", "キャプション", "プロンプト"],
                        label="検索対象",
                        value="画像"
                    )
            with gr.Row():
                text_input = gr.Textbox(label="検索テキスト", lines=2)
            with gr.Row():  
                with gr.Column(scale=2):
                    search_button = gr.Button("検索")
                with gr.Column(scale=1):
                    clear_button = gr.Button("クリア")
        with gr.Column(scale=1):
            image_input = gr.Image(label="画像による検索", type="pil", height=280, width=500, interactive=False)
    with gr.Row():    
        with gr.Column(scale=7):
            gallery = gr.Gallery(label="検索結果", show_label=False, elem_id="gallery", columns=[8], rows=[2], height=380, interactive=False, show_download_button=True)
    with gr.Row():
        prev_button = gr.Button("前")
        page_info = gr.Textbox(interactive=False, show_label=False, container=False)
        next_button = gr.Button("次")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_name = gr.Textbox(label="ファイル名")
            distance = gr.Textbox(label="ベクトル距離（-1 x 内積) or 全文検索スコア")
        with gr.Column(scale=2):        
            generation_prompt = gr.Textbox(label="画像生成プロンプト", lines=4)
        with gr.Column(scale=2):
            caption = gr.Textbox(label="キャプション", lines=4)

    def get_total_image_count():
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM IMAGES")
        count = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return count
    
    def get_total_pages():
        total_images = get_total_image_count()
        return (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    
    def change_page(direction, current_page):
        total_pages = get_total_pages()

        if direction == "next" and current_page < total_pages:
            current_page += 1
        elif direction == "prev" and current_page > 1:
            current_page -= 1
        
        images, image_info = load_initial_images(current_page)
        gallery_images = [(img, None) for img in images]
        page_info_text = f"{current_page} / {total_pages}"
        return gallery_images, image_info, gr.update(interactive=current_page > 1), gr.update(interactive=current_page < total_pages), current_page, page_info_text

    def next_page(current_page):
        return change_page("next", current_page)

    def prev_page(current_page):
        return change_page("prev", current_page)


    def search_wrapper(text_query, image_query, search_method, search_target):
        if text_query:
            images, image_info = search(text_query, search_method, search_target)
            image_input_update = gr.update(value=None, interactive=False)
            text_input_update = gr.update(interactive=True)
            prev_button_update = gr.update(interactive=False)
            next_button_update = gr.update(interactive=False)
        elif image_query is not None:
            images, image_info = search(image_query, search_method, search_target)
            image_input_update = gr.update(interactive=True)
            text_input_update = gr.update(interactive=False)
            prev_button_update = gr.update(interactive=False)
            next_button_update = gr.update(interactive=False)
        else: # 検索条件がない場合は、初期画像（最近アップロードされた画像）を表示
            images, image_info = load_initial_images()
            image_input_update = gr.update(interactive=search_method == "画像ベクトル検索")
            text_input_update = gr.update(interactive=search_method != "画像ベクトル検索")
            prev_button_update = gr.update(interactive=True)
            next_button_update = gr.update(interactive=True)
        label = "スコア" if search_method == "自然言語全文検索" else "距離"
        gallery_images = [(img, f"{label}: {round(float(info['vector_distance']), 3)}" if info['vector_distance'] != 'N/A' else None) for img, info in zip(images, image_info)]
        return gallery_images, image_info, text_input_update, image_input_update, gr.update(interactive=True), gr.update(selected_index=None), prev_button_update, next_button_update


    def clear_components(search_method):
        return (
            gr.update(value="", interactive=search_method != "画像ベクトル検索"),  # text_input
            gr.update(value=None, interactive=search_method == "画像ベクトル検索"),  # image_input
            gr.update(interactive=True),  # search_button
            gr.update(value=None, selected_index=None),  # gallery
            [],  # image_info_state
            gr.update(value=""),  # file_name
            gr.update(value=""),  # distance
            gr.update(value=""),  # generation_prompt
            gr.update(value="")   # caption
        )

    def update_text_input(search_method):
        if search_method == "画像ベクトル検索":
            return gr.update(value=None, interactive=False)
        else:
            return gr.update(interactive=True)
        
    def update_image_input(search_method):
        if search_method == "画像ベクトル検索":
            return gr.update(interactive=True)
        else:
            return gr.update(value=None, interactive=False)
        
    def update_search_target(search_method):
        if search_method == "自然言語全文検索":
            return gr.update(choices=["キャプション", "プロンプト"], value="キャプション")
        elif search_method == "画像ベクトル検索":
            return gr.update(choices=["画像"], value="画像")
        else:
            return gr.update(choices=["画像", "キャプション", "プロンプト"], value="画像")

    def on_select(evt: gr.SelectData, image_info):
        selected_index = evt.index
        if 0 <= selected_index < len(image_info):
            info = image_info[selected_index]
            vector_distance = info.get('vector_distance', 'N/A')
            
            # データベースからdescriptionを取得
            connection = oracledb.connect(user=username, password=password, dsn=dsn)
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT description
                FROM IMAGE_DESCRIPTIONS
                WHERE image_id = (SELECT image_id FROM IMAGES WHERE file_name = :file_name)
                ORDER BY description_id
            """, {'file_name': info['file_name']})
            
            descriptions = cursor.fetchall()
            
            # LOBオブジェクトを文字列に変換
            description_texts = [desc[0].read() if desc[0] else "" for desc in descriptions]
            
            cursor.close()
            connection.close()
            
            # descriptionを連結
            combined_description = "\n\n".join(description_texts) if description_texts else "説明なし"
            
            return info['file_name'], str(vector_distance), info['generation_prompt'], combined_description
        else:
            return "選択エラー", "N/A", "選択エラー", "選択エラー"
    
    search_button.click(search_wrapper, inputs=[text_input, image_input, search_method, search_target], outputs=[gallery, image_info_state, text_input, image_input, search_button, gallery, prev_button, next_button])
    clear_button.click(clear_components, inputs=[search_method], outputs=[text_input, image_input, search_button, gallery, image_info_state, file_name, distance, generation_prompt, caption])
    search_method.change(update_image_input, inputs=[search_method], outputs=[image_input])
    search_method.change(update_text_input, inputs=[search_method], outputs=[text_input])
    search_method.change(update_search_target, inputs=[search_method], outputs=[search_target])
    gallery.select(on_select, inputs=[image_info_state], outputs=[file_name, distance, generation_prompt, caption])
    next_button.click(next_page, inputs=[current_page], outputs=[gallery, image_info_state, prev_button, next_button, current_page, page_info])
    prev_button.click(prev_page, inputs=[current_page], outputs=[gallery, image_info_state, prev_button, next_button, current_page, page_info])

    # デモの起動時に初期画像を表示するための関数
    def load_initial_gallery():
        images, image_info = load_initial_images(1)
        total_pages = get_total_pages()
        page_info_text = f"1 / {total_pages}"
        return images, image_info, gr.update(interactive=False), gr.update(interactive=total_pages > 1), 1, page_info_text

    demo.load(load_initial_gallery, outputs=[gallery, image_info_state, prev_button, next_button, current_page, page_info])
    
    # デモの起動時に初期画像を表示
    demo.load(load_initial_gallery, outputs=[gallery, image_info_state])

if __name__ == "__main__":
    try:
        demo.queue()
        demo.launch(debug=True, share=True, server_port=8899)
    except Exception as e:
        print(e)
        demo.close()