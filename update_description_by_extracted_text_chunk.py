from PIL import Image
import oracledb
import os
from dotenv import load_dotenv

_ = load_dotenv()

# データベース接続情報
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
dsn = os.getenv("DB_DSN")

def extract_text_chunks(png_file):
    with Image.open(png_file) as img:
        return {key: value for key, value in img.info.items() if isinstance(value, str)}

def format_chunks(chunks):
    return "\n".join(f"{key}:{value}" for key, value in chunks.items())

def process_image(image_path):
    if not image_path.lower().endswith('.png'):
        return

    chunks = extract_text_chunks(image_path)
    if not chunks:
        return

    formatted_chunks = format_chunks(chunks)

    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    try:
        # ファイル名からimage_idを取得
        file_name = os.path.basename(image_path)
        cursor.execute("""
            SELECT image_id FROM IMAGES WHERE file_name = :file_name
        """, [file_name])
        result = cursor.fetchone()
        if not result:
            print(f"画像 {file_name} がデータベースに見つかりません。")
            return
        image_id = result[0]

        if 'parameters' in chunks and any('Negative prompt' in chunk for chunk in chunks.values()):
            # この画像は、Stabile Diffusion Web UI AUTOMATIC1111 版 により生成されたものと推測されます
            # すべてのチャンクをgeneration_promptに登録
            cursor.execute("""
                UPDATE IMAGES
                SET generation_prompt = :prompt
                WHERE image_id = :image_id
            """, [formatted_chunks, image_id])
        elif 'prompt' in chunks and 'workflow' in chunks:
            # この画像は、ComfyUI により生成されたものと推測されます
            # すべてのチャンクをgeneration_promptに登録
            cursor.execute("""
                UPDATE IMAGES
                SET generation_prompt = :prompt
                WHERE image_id = :image_id
            """, [formatted_chunks, image_id])
        elif 'prompt' in chunks and 'prompt_3' in chunks:
            # この画像は、SD3 モデルを使ったカスタムアプリケーションにより生成されたものと推測されます
            # すべてのチャンクをgeneration_promptに登録
            cursor.execute("""
                UPDATE IMAGES
                SET generation_prompt = :prompt
                WHERE image_id = :image_id
            """, [formatted_chunks, image_id])
        else:
            # すべてのチャンクをdescriptionに登録
            cursor.execute("""
                INSERT INTO IMAGE_DESCRIPTIONS (image_id, description)
                VALUES (:image_id, :description)
            """, [image_id, formatted_chunks])

        connection.commit()
        print(f"画像 {file_name} の処理が完了しました。")
    except oracledb.Error as e:
        print(f"データベース処理中にエラーが発生しました: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()

def process_all_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            full_path = os.path.join(folder_path, filename)
            process_image(full_path)

# 使用例
images_folder = "images"  # imagesフォルダーのパス
process_all_images(images_folder)