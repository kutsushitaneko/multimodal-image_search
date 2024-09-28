import spaces
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
from huggingface_hub import login
import cohere
import re

_ = load_dotenv(find_dotenv())

IMAGES_PER_PAGE = 16

# データベース接続情報
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
dsn = os.getenv("DB_DSN")

# Japanese Stable CLIPモデルのロード
login(os.getenv('HF_TOKEN'))
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

# Cohere設定
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

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

def do_rerank(query, documents, top_n):
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=documents,
        rank_fields=["text"],
        top_n=top_n,
        return_documents=True
    )

    """
    print("Rerank Response:")
    for result in response.results:
        print(f"Index: {result.index}, Relevance Score: {result.relevance_score}")
        print(f"Document: {result.document}")
    """
    
    return [(result.document.id, result.relevance_score, result.document.text)
            for result in response.results]

def get_image_data(image_id):
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    cursor.execute("SELECT image_data FROM IMAGES WHERE image_id = :image_id", {'image_id': image_id})
    image_data = cursor.fetchone()[0].read()

    cursor.close()
    connection.close()

    return image_data

def get_multiple_image_data(image_ids):
    if not image_ids:
        return {}  # 画像IDが空の場合は空の辞書を返す

    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    placeholders = ','.join([':' + str(i+1) for i in range(len(image_ids))])
    query = f"SELECT image_id, image_data FROM IMAGES WHERE image_id IN ({placeholders})"
    
    # バインド変数を辞書形式で渡す
    bind_vars = {str(i+1): image_id for i, image_id in enumerate(image_ids)}
    
    cursor.execute(query, bind_vars)
    
    image_data_dict = {row[0]: row[1].read() for row in cursor}

    cursor.close()
    connection.close()

    return image_data_dict

def load_initial_images(page=1):
    offset = (page - 1) * IMAGES_PER_PAGE
    results = get_latest_images(limit=IMAGES_PER_PAGE, offset=offset)
    
    # 画像IDのリストを作成
    image_ids = [result[0] for result in results]
    
    # 一度にすべての画像データを取得
    image_data_dict = get_multiple_image_data(image_ids)
    
    images = []
    image_info = []
    for image_id, file_name, generation_prompt, combined_description in results:
        image_data = image_data_dict.get(image_id)
        if image_data:
            images.append(Image.open(io.BytesIO(image_data)))
            image_info.append({
                'file_name': file_name,
                'generation_prompt': generation_prompt,
                'combined_description': combined_description,
                'vector_distance': 'N/A'
            })

    return images, image_info

def get_latest_images(limit=16, offset=0):
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT i.image_id, i.file_name, i.generation_prompt,
            id.description AS combined_description
        FROM IMAGES i
        LEFT JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
        ORDER BY i.upload_date DESC
        OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
    """, {'limit': limit, 'offset': offset})

    results = cursor.fetchall()
    
    # LOBオブジェクトを文字列に変換
    processed_results = []
    for row in results:
        image_id, file_name, generation_prompt, combined_description = row
        processed_results.append((
            image_id,
            file_name,
            generation_prompt.read() if generation_prompt else None,
            combined_description.read() if combined_description else "説明なし"
        ))

    cursor.close()
    connection.close()

    return processed_results

@spaces.GPU(duration=60)
def search_images(query, search_method, search_target, reranking, limit=16):
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    if search_method == "自然言語ベクトル検索":
        if search_target == "画像":
            embedding_json = json.dumps(compute_text_embeddings(query).tolist()[0])
            cursor.execute("""
                SELECT i.image_id, i.file_name, i.generation_prompt,
                    id.description as combined_description,
                    cie.embedding <#> :query_embedding as vector_distance,
                    'vector' as method
                FROM CURRENT_IMAGE_EMBEDDINGS cie
                JOIN IMAGES i ON cie.image_id = i.image_id
                LEFT JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
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
                    id.description as combined_description,
                    id.embedding <#> :query_embedding as vector_distance,
                    'vector' as method
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
                    id.description as combined_description,
                    i.prompt_embedding <#> :query_embedding as vector_distance,
                    'vector' as method
                FROM IMAGES i
                LEFT JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
                WHERE i.prompt_embedding IS NOT NULL
                ORDER BY vector_distance
                FETCH FIRST :limit ROWS ONLY
            """, {'query_embedding': embedding_json, 'limit': limit})
    elif search_method == "自然言語全文検索":
        # 検索テキストが複数のワードの場合、それらを " or " で連結
        if re.search(r"\s", query):
            words = query.split()
            query = " or ".join(words)

        if search_target == "キャプション":
            cursor.execute("""
                SELECT i.image_id, i.file_name, i.generation_prompt,
                    id.description as combined_description,
                    score(1) as relevance,
                    'text' as method
                FROM IMAGES i
                JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
                WHERE CONTAINS(id.description, :query, 1) > 0
                ORDER BY relevance DESC, i.image_id ASC
                FETCH FIRST :limit ROWS ONLY
            """, {'query': query, 'limit': limit})
        elif search_target == "プロンプト":
            cursor.execute("""
                SELECT i.image_id, i.file_name, i.generation_prompt,
                    id.description as combined_description,
                    score(1) as relevance,
                    'text' as method
                FROM IMAGES i
                LEFT JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
                WHERE i.generation_prompt IS NOT NULL
                AND CONTAINS(i.generation_prompt, :query, 1) > 0
                ORDER BY relevance DESC, i.image_id ASC
                FETCH FIRST :limit ROWS ONLY
            """, {'query': query, 'limit': limit})
    elif search_method == "画像ベクトル検索":
        embedding_json = json.dumps(compute_image_embeddings(query).tolist()[0])
        cursor.execute("""
            SELECT i.image_id, i.file_name, i.generation_prompt,
                id.description as combined_description,
                cie.embedding <#> :query_embedding as vector_distance,
                'vector' as method
            FROM CURRENT_IMAGE_EMBEDDINGS cie
            JOIN IMAGES i ON cie.image_id = i.image_id
            LEFT JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
            ORDER BY vector_distance
            FETCH FIRST :limit ROWS ONLY
        """, {'query_embedding': embedding_json, 'limit': limit})
    elif search_method == "ハイブリッド検索":
        sql_limit = limit * 2 if reranking == "リランキング" else limit
        text_query = query
        # 検索テキストが複数のワードの場合、それらを " or " で連結
        if re.search(r"\s", text_query):
            words = text_query.split()
            text_query = " or ".join(words)

        if search_target == "キャプション":
            # クエリの埋め込みベクトルを取得
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
                SELECT * FROM (
                    (SELECT i.image_id, i.file_name, i.generation_prompt,
                        id.description as combined_description,
                        id.embedding <#> :query_embedding AS score,
                        'vector' AS method
                    FROM IMAGE_DESCRIPTIONS id
                    JOIN IMAGES i ON id.image_id = i.image_id
                    ORDER BY score ASC, i.image_id ASC
                    FETCH FIRST :half_limit ROWS ONLY)
                    UNION ALL
                    (SELECT i.image_id, i.file_name, i.generation_prompt,
                        id.description as combined_description,
                        score(1) AS score,
                        'text' AS method
                    FROM IMAGES i
                    JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
                    WHERE CONTAINS(id.description, :text_query, 1) > 0
                    ORDER BY score DESC, i.image_id ASC
                    FETCH FIRST :half_limit ROWS ONLY)
                )
                ORDER BY method, score, image_id ASC
                FETCH FIRST :limit ROWS ONLY
            """, {
                'query_embedding': embedding_json,
                'text_query': text_query,
                'half_limit': int(sql_limit / 2),
                'limit': sql_limit
            })
        elif search_target == "プロンプト":
            # クエリの埋め込みベクトルを取得
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
                SELECT * FROM (
                    (SELECT i.image_id, i.file_name, i.generation_prompt,
                        id.description as combined_description,
                        i.prompt_embedding <#> :query_embedding as score, 
                        'vector' as method
                    FROM IMAGES i
                    LEFT JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
                    WHERE i.prompt_embedding IS NOT NULL
                    ORDER BY score ASC, i.image_id ASC
                    FETCH FIRST :half_limit ROWS ONLY)
                    UNION ALL
                    (SELECT i.image_id, i.file_name, i.generation_prompt,
                        id.description as combined_description,
                        score(1) as score,
                        'text' as method
                    FROM IMAGES i
                    LEFT JOIN IMAGE_DESCRIPTIONS id ON i.image_id = id.image_id
                    WHERE i.generation_prompt IS NOT NULL
                    AND CONTAINS(i.generation_prompt, :text_query, 1) > 0
                    ORDER BY score DESC, i.image_id ASC
                    FETCH FIRST :half_limit ROWS ONLY)
                )
                ORDER BY method, score, image_id ASC
                FETCH FIRST :limit ROWS ONLY
            """, {
                'query_embedding': embedding_json,
                'text_query': text_query,
                'half_limit': int(sql_limit / 2),
                'limit': sql_limit
            })
        else:
            raise ValueError("ハイブリッド検索での無効な検索対象です")
    else:
        raise ValueError("無効な検索方法です")

    results = cursor.fetchall()
    
    # LOBオブジェクトを文字列に変換
    processed_results = []
    for row in results:
        image_id, file_name, generation_prompt, combined_description, score, method = row
        processed_results.append((
            image_id,
            file_name,
            generation_prompt.read() if hasattr(generation_prompt, 'read') else generation_prompt,
            combined_description.read() if hasattr(combined_description, 'read') else combined_description,
            score,
            method
        ))

    cursor.close()
    connection.close()

    print(f"検索結果: {len(processed_results)}件")
    return processed_results

with gr.Blocks(title="画像検索") as demo:
    image_info_state = gr.State([])
    current_page = gr.State(1)

    gr.Markdown("# マルチモーダル画像検索")
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Row():
                with gr.Column(scale=2):
                    search_target = gr.Radio(
                        ["画像", "キャプション", "プロンプト"],
                        label="検索対象",
                        value="画像"
                    )
                with gr.Column(scale=3):
                    search_method = gr.Radio(
                        ["自然言語ベクトル検索", "画像ベクトル検索"],
                        label="検索方法",
                        value="自然言語ベクトル検索"
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    reranking = gr.Radio(
                        ["リランキング", "リランキングなし"],
                        label="リランキング",
                        value="リランキングなし",
                        interactive=False
                    )
            with gr.Row():
                text_input = gr.Textbox(label="検索テキスト", lines=2, show_copy_button=True)
            with gr.Row():  
                with gr.Column(scale=6):
                    search_button = gr.Button("検索")
                with gr.Column(scale=1):
                    clear_button = gr.Button("クリア")
        with gr.Column(scale=2):
            image_input = gr.Image(label="検索画像", type="pil", height=280, width=500, interactive=False)
    
    with gr.Row():    
        with gr.Column(scale=7):
            gallery = gr.Gallery(label="検索結果", show_label=False, elem_id="gallery", columns=[8], rows=[2], height=380, interactive=False, show_download_button=True)
    with gr.Row():
        prev_button = gr.Button("前")
        page_info = gr.Textbox(interactive=False, show_label=False, container=False)
        next_button = gr.Button("次")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_name = gr.Textbox(label="ファイル名", show_copy_button=True)
            distance = gr.Textbox(label="ベクトル距離（-1 x 内積) or 全文検索スコア", show_copy_button=True)
        with gr.Column(scale=2):        
            generation_prompt = gr.Textbox(label="画像生成プロンプト", lines=4, show_copy_button=True)
        with gr.Column(scale=2):
            caption = gr.Textbox(label="キャプション", lines=4, show_copy_button=True)

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


    def search_wrapper(text_query, image_query, search_method, search_target, reranking):
        if text_query or image_query is not None:
            results = search_images(text_query if text_query else image_query, search_method, search_target, reranking)

            if not results:  # 検索結果が空の場合
                return [], [], gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(selected_index=None), gr.update(interactive=False), gr.update(interactive=False), ""  
            # ベクトル検索結果と全文検索結果を分離
            vector_results = []
            text_results = []
            vector_info = []
            text_info = []
            
            # 画像IDのリストを作成
            image_ids = [result[0] for result in results]
            
            # 一度にすべての画像データを取得
            image_data_dict = get_multiple_image_data(image_ids)
            
            for result in results:
                image_id, file_name, generation_prompt, combined_description, score, method = result

                image_data = image_data_dict.get(image_id)
                if image_data:
                    img = Image.open(io.BytesIO(image_data))
                    img.load()
                else:
                    continue  # 画像データが見つからない場合はスキップ
                
                # 情報を構築
                info = {
                    'image_id': image_id,
                    'file_name': file_name,
                    'generation_prompt': generation_prompt,
                    'vector_distance': score if score is not None else 'N/A',
                    'method': method,
                    'combined_description': combined_description
                }


                if method == 'vector':
                    caption = f"ベクトル距離: {round(float(score), 3) if isinstance(score, (int, float)) else score}"
                    vector_results.append((img, caption))
                    vector_info.append(info)
                elif method == 'text':
                    caption = f"スコア: {round(float(score), 3) if isinstance(score, (int, float)) else score}"
                    text_results.append((img, caption))
                    text_info.append(info)
                else:
                    caption = f"{method.capitalize()} - スコア: {round(float(score), 3) if isinstance(score, (int, float)) else score}"
                    vector_results.append((img, caption))
                    vector_info.append(info)

            # ハイブリッド検索の場合、text_resultsとtext_infoをスコアの降順、image_idの昇順に並べ替える
            if search_method == "ハイブリッド検索":
                text_results_with_info = list(zip(text_results, text_info))
                text_results_with_info.sort(key=lambda x: (-float(x[1]['vector_distance']), x[1]['image_id']))
                if text_results_with_info:
                    text_results, text_info = map(list, zip(*text_results_with_info))
                else:
                    text_results, text_info = [], []

            # ギャラリーの画像と情報を統合
            gallery_images = vector_results + text_results
            image_info = vector_info + text_info

            print(f"search_target: {search_target}")
            print(f"search_method: {search_method}")
            print(f"reranking: {reranking}")
            if search_target in ["キャプション", "プロンプト"] and (search_method == "自然言語ベクトル検索" or search_method == "自然言語全文検索" or search_method == "ハイブリッド検索") and reranking == "リランキング":
                # リランキング用のドキュメントリストを作成
                if search_method == "ハイブリッド検索":
                    # 重複を排除するために、image_id をキーとした辞書を使用
                    unique_documents = {}
                    unique_indices = {}
                    for idx, info in enumerate(image_info):
                        image_id_str = str(info['image_id'])
                        if image_id_str not in unique_documents:
                            document_text = info['combined_description'] if search_target == "キャプション" else info['generation_prompt']
                            unique_documents[image_id_str] = {
                                'id': image_id_str,
                                'text': document_text
                            }
                            unique_indices[image_id_str] = idx  # オリジナルのインデックスを保持
                    documents = list(unique_documents.values())
                else:
                    documents = [
                        {
                            'id': str(info['image_id']),
                            'text': info['combined_description'] if search_target == "キャプション" else info['generation_prompt']
                        }
                        for info in image_info
                    ]
                
                # Cohere Rerankを使用してリランキング
                reranked_results = do_rerank(text_query, documents,top_n=16)
                #print(f"reranked_results: {reranked_results}")

                # リランキング結果に基づいて画像と情報を並べ替え
                reranked_image_info = []
                reranked_gallery_images = []
                for doc_id, score, _ in reranked_results:
                    index = next(i for i, info in enumerate(image_info) if str(info['image_id']) == doc_id)
                    updated_info = image_info[index].copy()
                    updated_info['vector_distance'] = score  # リランクスコアで更新
                    reranked_image_info.append(updated_info)
                    
                    img, _ = gallery_images[index]
                    new_caption = f"リランクスコア: {round(score, 3)}"
                    reranked_gallery_images.append((img, new_caption))
                
                gallery_images = reranked_gallery_images
                image_info = reranked_image_info

            page_info_text = ""

            # ギャラリーの更新と他のUI要素の更新
            return gallery_images, image_info, gr.update(interactive=not (search_target == "画像" and search_method == "画像ベクトル検索")), gr.update(interactive=search_target == "画像" and search_method == "画像ベクトル検索"), gr.update(interactive=True), gr.update(selected_index=None), gr.update(interactive=False), gr.update(interactive=False), page_info_text
        else:
            # 初期表示の処理
            images, image_info = load_initial_images()
            gallery_images = [(img, None) for img in images]
            page_info_text = f"1 / {get_total_pages()}"
            return gallery_images, image_info, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True), gr.update(selected_index=None), gr.update(interactive=True), gr.update(interactive=True), page_info_text


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

    def update_search_method(search_target):
        if search_target == "画像":
            return gr.update(choices=["自然言語ベクトル検索", "画像ベクトル検索"], value="自然言語ベクトル検索"), gr.update(interactive=False, value="リランキングなし")
        elif search_target == "キャプション" or search_target == "プロンプト":
            return gr.update(choices=["自然言語ベクトル検索", "自然言語全文検索", "ハイブリッド検索"], value="自然言語ベクトル検索"), gr.update(interactive=True)
        else:
            return gr.update(choices=[], value=None), gr.update(interactive=False)

        
    def update_text_input(search_method):
        if search_method == "画像ベクトル検索":
            return gr.update(value=None, interactive=False)
        else:
            return gr.update(interactive=True)
                
    def update_image_input(search_target, search_method):
        if search_target == "画像" and search_method == "画像ベクトル検索":
            return gr.update(interactive=True)
        else:
            return gr.update(value=None, interactive=False)
        
    def on_select(evt: gr.SelectData, image_info):
        selected_index = evt.index
        if 0 <= selected_index < len(image_info):
            info = image_info[selected_index]
            vector_distance = info.get('vector_distance', 'N/A')
            
            return info['file_name'], str(vector_distance), info['generation_prompt'], info['combined_description']
        else:
            return "選択エラー", "N/A", "選択エラー", "選択エラー"
    
    search_button.click(search_wrapper, inputs=[text_input, image_input, search_method, search_target, reranking], outputs=[gallery, image_info_state, text_input, image_input, search_button, gallery, prev_button, next_button, page_info])
    clear_button.click(clear_components, inputs=[search_method], outputs=[text_input, image_input, search_button, gallery, image_info_state, file_name, distance, generation_prompt, caption])
    search_target.change(update_search_method, inputs=[search_target], outputs=[search_method, reranking])
    search_target.change(update_image_input, inputs=[search_target, search_method], outputs=[image_input])
    search_method.change(update_text_input, inputs=[search_method], outputs=[text_input])
    search_method.change(update_image_input, inputs=[search_target, search_method], outputs=[image_input])

    gallery.select(on_select, inputs=[image_info_state], outputs=[file_name, distance, generation_prompt, caption])
    next_button.click(next_page, inputs=[current_page], outputs=[gallery, image_info_state, prev_button, next_button, current_page, page_info])
    prev_button.click(prev_page, inputs=[current_page], outputs=[gallery, image_info_state, prev_button, next_button, current_page, page_info])

    # デモの起動時に初期画像を表示するための関数
    def load_initial_gallery():
        images, image_info = load_initial_images(1)
        total_pages = get_total_pages()
        page_info_text = f"1 / {total_pages}"
        return images, image_info, gr.update(interactive=False), gr.update(interactive=total_pages > 1), 1, page_info_text

    # デモの起動時に初期画像を表示
    demo.load(load_initial_gallery, outputs=[gallery, image_info_state, prev_button, next_button, current_page, page_info])

if __name__ == "__main__":
    try:
        demo.queue()
        demo.launch(debug=True, share=True, server_port=8899)
    except Exception as e:
        print(e)
        demo.close()