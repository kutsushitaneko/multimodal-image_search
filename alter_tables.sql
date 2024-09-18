-- IMAGE_DESCRIPTIONS テーブルに embedding 列を追加
ALTER TABLE IMAGE_DESCRIPTIONS ADD embedding VECTOR;

-- embedding 列にベクトルインデックスを作成
CREATE VECTOR INDEX idx_image_description_embedding ON IMAGE_DESCRIPTIONS(embedding) ORGANIZATION NEIGHBOR PARTITIONS
WITH DISTANCE DOT;