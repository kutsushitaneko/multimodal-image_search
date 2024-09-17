import oci
import os
import json
import time
from dotenv import load_dotenv, find_dotenv
import oracledb

_ = load_dotenv(find_dotenv())

# OCI設定
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
compartment_id = os.getenv("OCI_COMPARTMENT_ID") 
model_id = "cohere.embed-multilingual-v3.0"
generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))

# データベース接続情報
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
dsn = os.getenv("DB_DSN")

# SQL 定義
sql_select_descriptions = """
    SELECT description_id, description
    FROM image_descriptions
    WHERE embedding IS NULL
"""

sql_update_embedding = """
    UPDATE image_descriptions
    SET embedding = :embedding
    WHERE description_id = :description_id
"""

def generate_embeddings(batch):
    embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
    embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id)
    embed_text_detail.inputs = batch
    embed_text_detail.truncate = "NONE"
    embed_text_detail.compartment_id = compartment_id
    embed_text_detail.is_echo = False
    embed_text_detail.input_type = "SEARCH_DOCUMENT"

    embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)
        
    return embed_text_response.data.embeddings

def process_descriptions(select_cursor, update_cursor, batch_size=96):
    select_cursor.execute(sql_select_descriptions)
    
    total_processed = 0
    total_time = 0
    batch_count = 0

    while True:
        batch_start_time = time.time()
        batch = select_cursor.fetchmany(batch_size)
        if not batch:
            break  # データがなくなったらループを終了

        # バッチサイズが0の場合はスキップ
        if len(batch) == 0:
            continue

        description_ids, descriptions = zip(*batch)
        # LOB型を文字列に変換
        descriptions = [desc.read() if hasattr(desc, 'read') else desc for desc in descriptions]
        embeddings = generate_embeddings(descriptions)

        for description_id, embedding in zip(description_ids, embeddings):
            update_cursor.execute(sql_update_embedding, {
                'embedding': json.dumps(embedding),
                'description_id': description_id
            })

        connection.commit()
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        batch_count += 1
        total_processed += len(batch)
        total_time += batch_time

        print(f"バッチ {batch_count}: {len(batch)} 件の embedding を更新しました。処理時間: {batch_time:.2f} 秒")

    return total_processed, total_time

if __name__ == "__main__":
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        select_cursor = connection.cursor()
        update_cursor = connection.cursor()
    
        start_time = time.time()
        total_processed, processing_time = process_descriptions(select_cursor, update_cursor)
        end_time = time.time()

        total_time = end_time - start_time

        print(f"\n処理が完了しました。")
        print(f"合計処理件数: {total_processed} 件")
        print(f"embedding 生成時間: {processing_time:.2f} 秒")
        print(f"総処理時間: {total_time:.2f} 秒")

    except oracledb.Error as e:
        print(f"データベース処理中にエラーが発生しました: {e}")
        if 'DPY-1003' not in str(e):  # DPY-1003エラーは無視
            if connection:
                connection.rollback()
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

    finally:
        if 'select_cursor' in locals():
            select_cursor.close()
            print("select用カーソルを閉じました")
        if 'update_cursor' in locals():
            update_cursor.close()
            print("update用カーソルを閉じました")
        if 'connection' in locals():
            connection.close()
            print("データベース接続を閉じました")

    print("画像説明の embedding 生成と更新が完了しました。")
