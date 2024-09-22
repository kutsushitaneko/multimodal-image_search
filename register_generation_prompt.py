from PIL import Image
import oracledb
import os
from dotenv import load_dotenv
import time

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
        return False

    chunks = extract_text_chunks(image_path)
    if not chunks:
        return False

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
            return False
        image_id = result[0]

        # generation_promptが空かどうかを確認
        cursor.execute("""
            SELECT generation_prompt FROM IMAGES WHERE image_id = :image_id
        """, [image_id])
        existing_prompt = cursor.fetchone()[0]

        if existing_prompt is None or (isinstance(existing_prompt, str) and existing_prompt.strip() == '') or (hasattr(existing_prompt, 'read') and existing_prompt.read().strip() == ''):
            if 'parameters' in chunks and any('Negative prompt' in chunk for chunk in chunks.values()):
                # この画像は、Stable Diffusion Web UI AUTOMATIC1111 版 により生成されたものと推測されます
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

            connection.commit()
            print(f"画像 {file_name} の処理が完了しました。")
            return True
        else:
            print(f"画像 {file_name} のgeneration_promptは既に設定されています。スキップします。")
            return False

    except oracledb.Error as e:
        print(f"データベース処理中にエラーが発生しました: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()

def process_all_images(folder_path):
    start_time = time.time()
    total_images = 0
    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            total_images += 1
            full_path = os.path.join(folder_path, filename)
            if process_image(full_path):
                processed_count += 1
            else:
                skipped_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n画像の処理とデータベースへの登録が完了しました。")
    print(f"フォルダー内のPNG画像数: {total_images}")
    print(f"処理した画像数: {processed_count}")
    print(f"スキップした画像数: {skipped_count}")
    print(f"トータル処理時間: {elapsed_time:.2f}秒")

# 使用例
if __name__ == "__main__":
    images_folder = "images"  # imagesフォルダーのパス
    process_all_images(images_folder)