from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
from PIL import Image

# モデルの選択（2B モデルか 7B モデルかを選択）
model_id = "Qwen/Qwen2-VL-7B-Instruct"


# モデルをロード
print(f"loading {model_id}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)


# プロセッサの初期化
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")



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
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

# 画像フォルダのパス
images_folder = "images"

# 結果を格納する辞書
results = {}

# 画像フォルダ内の全ての画像ファイルを処理
for filename in os.listdir(images_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        image_path = os.path.join(images_folder, filename)
        print(f"Processing {filename}...")
        caption = process_image(image_path)
        results[filename] = caption

# 結果をJSONファイルに出力
with open("captions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("処理が完了しました。結果はcaptions.jsonに保存されました。")





