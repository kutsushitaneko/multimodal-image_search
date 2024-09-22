import json
import oracledb
from dotenv import load_dotenv
import os
import time

# 環境変数の読み込み
load_dotenv()

# データベース接続情報
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
dsn = os.getenv("DB_DSN")

# captions.json ファイルの読み込み
with open('captions.json', 'r', encoding='utf-8') as f:
    captions = json.load(f)

# データベース接続
connection = oracledb.connect(user=username, password=password, dsn=dsn)
cursor = connection.cursor()

start_time = time.time()
processed_count = 0
inserted_count = 0

try:
    for filename, caption in captions.items():
        # IMAGES テーブルから image_id を取得（大文字小文字を区別しない）
        cursor.execute("""
            SELECT image_id FROM IMAGES WHERE LOWER(file_name) = LOWER(:filename)
        """, filename=filename)
        result = cursor.fetchone()
        
        if result:
            image_id = result[0]
            
            # IMAGE_DESCRIPTIONS テーブルにレコードが存在するか確認
            cursor.execute("""
                SELECT COUNT(*) FROM IMAGE_DESCRIPTIONS WHERE image_id = :image_id
            """, image_id=image_id)
            description_count = cursor.fetchone()[0]
            
            if description_count == 0:
                # IMAGE_DESCRIPTIONS テーブルに挿入
                cursor.execute("""
                    INSERT INTO IMAGE_DESCRIPTIONS (image_id, description)
                    VALUES (:image_id, :description)
                """, image_id=image_id, description=caption)
                inserted_count += 1
            
            processed_count += 1
        else:
            print(f"画像 {filename} が IMAGES テーブルに見つかりません。")

    # 変更をコミット
    connection.commit()
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"キャプションの登録が完了しました。")
    print(f"処理件数: {processed_count}")
    print(f"挿入件数: {inserted_count}")
    print(f"処理時間: {processing_time:.2f}秒")

except oracledb.Error as e:
    print(f"エラーが発生しました: {e}")
    connection.rollback()

finally:
    cursor.close()
    connection.close()