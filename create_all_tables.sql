-- IMAGES テーブル
CREATE TABLE IMAGES (
    image_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    image_data BLOB,
    file_name VARCHAR2(255),
    file_type VARCHAR2(50),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generation_prompt CLOB
);

-- EMBEDDING_MODELS テーブル
CREATE TABLE EMBEDDING_MODELS (
    model_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name VARCHAR2(255),
    model_version VARCHAR2(50),
    is_current CHAR(1) CHECK (is_current IN ('Y', 'N')),
    vector_dimension NUMBER
);

-- IMAGE_EMBEDDINGS テーブル
CREATE TABLE IMAGE_EMBEDDINGS (
    embedding_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    image_id NUMBER,
    model_id NUMBER,
    embedding VECTOR,
    CONSTRAINT fk_image_embedding FOREIGN KEY (image_id) REFERENCES IMAGES(image_id),
    CONSTRAINT fk_model_embedding FOREIGN KEY (model_id) REFERENCES EMBEDDING_MODELS(model_id)
);

-- Vector indexの作成
CREATE VECTOR INDEX image_embedding_idx ON IMAGE_EMBEDDINGS(embedding) ORGANIZATION NEIGHBOR PARTITIONS
WITH DISTANCE DOT;

-- IMAGE_DESCRIPTIONS テーブル
CREATE TABLE IMAGE_DESCRIPTIONS (
    description_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    image_id NUMBER,
    description CLOB,
    CONSTRAINT fk_image_description FOREIGN KEY (image_id) REFERENCES IMAGES(image_id)
);

-- 全文検索インデックスの作成
BEGIN
  ctx_ddl.create_preference('japanese_lexer', 'JAPANESE_LEXER');
END;
/

CREATE INDEX idx_image_prompt ON IMAGES(generation_prompt) 
INDEXTYPE IS CTXSYS.CONTEXT 
PARAMETERS ('LEXER japanese_lexer SYNC (ON COMMIT)');

CREATE INDEX idx_image_description ON IMAGE_DESCRIPTIONS(description) 
INDEXTYPE IS CTXSYS.CONTEXT 
PARAMETERS ('LEXER japanese_lexer SYNC (ON COMMIT)');

-- 現在使用中のモデルを取得するビュー
CREATE VIEW CURRENT_EMBEDDING_MODEL AS
SELECT model_id, model_name, model_version, vector_dimension
FROM EMBEDDING_MODELS
WHERE is_current = 'Y';

-- 現在のモデルによるエンベディングを取得するビュー
CREATE VIEW CURRENT_IMAGE_EMBEDDINGS AS
SELECT ie.embedding_id, ie.image_id, ie.embedding
FROM IMAGE_EMBEDDINGS ie
JOIN CURRENT_EMBEDDING_MODEL cem ON ie.model_id = cem.model_id;
