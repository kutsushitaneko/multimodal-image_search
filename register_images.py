# %pip install -U torch transformers oracledb pillow python-dotenv
import os
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
import oracledb
from dotenv import load_dotenv, find_dotenv
import json

_= load_dotenv(find_dotenv())

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
    #return image_features.cpu().detach().numpy()[0]
    return image_features.cpu().detach().numpy()[0].tolist()  # numpy配列に変換し、リストに変換

# Oracleデータベースに接続
connection = oracledb.connect(user=username, password=password, dsn=dsn)
cursor = connection.cursor()

# EMBEDDING_MODELSテーブルにモデル情報を挿入
#cursor.execute("""
#    INSERT INTO EMBEDDING_MODELS (model_name, model_version, is_current, vector_dimension)
#    VALUES (:1, :2, :3, :4)
#""", ("Japanese Stable CLIP", "japanese-stable-clip-vit-l-16", 'Y', 768))

#model_id = cursor.lastrowid
model_id = 1

# imagesディレクトリ内の画像を処理
images_dir = "images"
for filename in os.listdir(images_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        image_path = os.path.join(images_dir, filename)
        
        # 画像を開いてエンベディングを生成
        with Image.open(image_path) as img:
            embedding = compute_image_embeddings(img)
            #print(embedding)
        
        # 画像データをバイナリとして読み込む
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
        
        # IMAGESテーブルに画像情報を挿入
        try:   
            new_image_id = cursor.var(int)
            cursor.execute("""
                INSERT INTO IMAGES (image_data, file_name, file_type)
                VALUES (:1, :2, :3)
                RETURNING image_id INTO :4
            """, (image_data, filename, os.path.splitext(filename)[1], new_image_id))
        
            image_id = new_image_id.getvalue()[0]
            # IMAGE_EMBEDDINGSテーブルにエンベディングを挿入
            print(f"image_id: {image_id}")
            print(f"model_id: {model_id}")

            embedding_json = json.dumps(embedding)
            cursor.execute("""
                INSERT INTO IMAGE_EMBEDDINGS (image_id, model_id, embedding)
                VALUES (:image_id, :model_id, :embedding)
            """, {'image_id': image_id, 'model_id': model_id, 'embedding': embedding_json})

            print("IMAGE_EMBEDDINGSテーブルへの登録が完了しました。コミットします")
            connection.commit()
            print(f"{os.path.splitext(filename)[1]}のIMAGESテーブルと IMAGE_EMBEDDINGSテーブルへの登録が完了しました。")
            
        except oracledb.Error as e:
            print(f"データベース処理中にエラーが発生しました: {e}")
            if connection:
                connection.rollback()
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}")
# 接続を閉じる
cursor.close()
connection.close()

print("画像の処理とデータベースへの登録が完了しました。")