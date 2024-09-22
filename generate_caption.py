from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
from PIL import Image
import time
from tqdm import tqdm

# モデルの選択（2B モデルか 7B モデルかを選択）
model_id = "Qwen/Qwen2-VL-7B-Instruct"


# モデルをロード
print(f"loading {model_id}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)


# プロセッサの初期化
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28
)

def process_image(image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open(image_path),
                },
                {"type": "text", "text": "画像を詳細に説明してください。 また、オブジェクト間の位置関係も説明してください。画像がテキストを主体としたものである場合は、テキストを抽出してください。"},
            ],
        }
    ]

    # プロンプトの準備
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # イメージの準備
    image_inputs, video_inputs = process_vision_info(messages)

    # モデル入力の準備
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 推論：出力の生成
    generated_ids = model.generate(**inputs, max_new_tokens=1000, repetition_penalty=1.1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

# 画像フォルダのパス
images_folder = "images"

# 処理対象の画像ファイルリストを取得
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
total_images = len(image_files)

# 結果を格納するJSONファイル
json_file = "captions.json"

# 既存の結果をロード（ファイルが存在する場合）
if os.path.exists(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}

# 画像ファイルを処理
for i, filename in enumerate(tqdm(image_files), 1):
    # ファイルが既に処理済みかチェック
    if filename in results:
        print(f"\nスキップ: {i}/{total_images} - {filename} (既に処理済み)")
        continue

    image_path = os.path.join(images_folder, filename)
    start_time = time.time()
    
    print(f"\n処理中: {i}/{total_images} - {filename}")
    
    caption = process_image(image_path)
    results[filename] = caption
    
    # 結果をJSONファイルに追記
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"処理時間: {processing_time:.2f}秒")

print(f"\n処理が完了しました。結果は{json_file}に保存されました。")





