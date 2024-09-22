import os
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
import oracledb
from dotenv import load_dotenv, find_dotenv
import json
import time

_ = load_dotenv(find_dotenv())

# データベース接続情報
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
dsn = os.getenv("DB_DSN")

# Japanese Stable CLIPモデルのロード
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "stabilityai/japanese-stable-clip-vit-l-16"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
processor = AutoImageProcessor.from_pretrained(model_path)

# 画像エンベディングを生成する関数
def compute_image_embeddings(image):
    image = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().detach().numpy()[0].tolist()

# Oracleデータベースに接続
connection = oracledb.connect(user=username, password=password, dsn=dsn)
cursor = connection.cursor()

# EMBEDDING_MODELSテーブルにモデル情報を挿入
cursor.execute("""
    SELECT model_id FROM EMBEDDING_MODELS WHERE model_name = :1
""", ("Japanese Stable CLIP",))

result = cursor.fetchone()

if result is None:
    cursor.execute("""
        INSERT INTO EMBEDDING_MODELS (model_name, model_version, is_current, vector_dimension)
        VALUES (:1, :2, :3, :4)
    """, ("Japanese Stable CLIP", "japanese-stable-clip-vit-l-16", 'Y', 768))
    model_id = cursor.lastrowid
    connection.commit()
    print("EMBEDDING_MODELSテーブルに新しいモデルを挿入しました。")
else:
    model_id = result[0]
    print("EMBEDDING_MODELSテーブルに既にモデルが存在します。")

# 処理開始時間を記録
start_time = time.time()

# 処理件数の初期化
total_images = 0
registered_count = 0
skipped_count = 0

# imagesディレクトリ内の画像を処理
images_dir = "images"
for filename in os.listdir(images_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        total_images += 1
        image_path = os.path.join(images_dir, filename)

        # ファイル名を小文字に変換
        filename_lower = filename.lower()

        # 既に同じファイル名の画像がテーブルに存在するかチェック（大文字小文字を区別しない）
        cursor.execute("""
            SELECT image_id FROM IMAGES WHERE LOWER(file_name) = :1
        """, (filename_lower,))
        image_exists = cursor.fetchone()

        if image_exists:
            print(f"{filename} は既にデータベースに存在するためスキップします。")
            skipped_count += 1
            continue  # 次の画像へ

        # 画像を開いてエンベディングを生成
        with Image.open(image_path) as img:
            embedding = compute_image_embeddings(img)

        # 画像データをバイナリとして読み込む
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        # IMAGESテーブルに画像情報を挿入（ファイル名と拡張子を小文字に統一）
        try:
            new_image_id = cursor.var(int)
            cursor.execute("""
                INSERT INTO IMAGES (image_data, file_name, file_type)
                VALUES (:1, :2, :3)
                RETURNING image_id INTO :4
            """, (image_data, filename_lower, os.path.splitext(filename_lower)[1], new_image_id))

            image_id = new_image_id.getvalue()[0]

            embedding_json = json.dumps(embedding)
            cursor.execute("""
                INSERT INTO IMAGE_EMBEDDINGS (image_id, model_id, embedding)
                VALUES (:image_id, :model_id, :embedding)
            """, {'image_id': image_id, 'model_id': model_id, 'embedding': embedding_json})

            connection.commit()
            registered_count += 1
            print(f"{filename} の登録が完了しました。")

        except oracledb.Error as e:
            print(f"データベース処理中にエラーが発生しました: {e}")
            if connection:
                connection.rollback()
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}")

# 接続を閉じる
cursor.close()
connection.close()

# 処理時間の計算
end_time = time.time()
elapsed_time = end_time - start_time

print("画像の処理とデータベースへの登録が完了しました。")
print(f"フォルダー内の画像数: {total_images}")
print(f"登録した画像数: {registered_count}")
print(f"スキップした画像数: {skipped_count}")
print(f"トータル処理時間: {elapsed_time:.2f}秒")