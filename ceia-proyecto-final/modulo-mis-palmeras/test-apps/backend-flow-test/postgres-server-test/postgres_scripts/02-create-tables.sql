-- Script de inicialización: Crear tabla imagenes
-- Se ejecuta después de 01-create-schemas.sql

SET search_path TO test_schema, public;

-- Crear tabla imagenes
CREATE TABLE IF NOT EXISTS test_schema.imagenes (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    storage_path VARCHAR(512) NOT NULL,
    file_original_name VARCHAR(255) NOT NULL,
    image_name UUID NOT NULL,
    hash_sha256 CHAR(64) NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT FALSE,
    date_created TIMESTAMP NOT NULL DEFAULT NOW(),
    date_deleted TIMESTAMP NULL,
    
    -- Constraints
    CONSTRAINT chk_hash_sha256_format CHECK (hash_sha256 ~ '^[a-f0-9]{64}$'),
    CONSTRAINT chk_deleted_date CHECK (
        (deleted = FALSE AND date_deleted IS NULL) OR
        (deleted = TRUE AND date_deleted IS NOT NULL)
    ),
    CONSTRAINT uk_hash_sha256 UNIQUE (hash_sha256)
);

-- Índices para mejorar el rendimiento
CREATE INDEX idx_imagenes_user_id ON test_schema.imagenes(user_id);

-- Comentarios de la tabla y columnas
COMMENT ON TABLE test_schema.imagenes IS 'Tabla para almacenar información de imágenes';
COMMENT ON COLUMN test_schema.imagenes.id IS 'Identificador único de la imagen';
COMMENT ON COLUMN test_schema.imagenes.user_id IS 'Identificador UUID del usuario (Keycloak sub)';
COMMENT ON COLUMN test_schema.imagenes.storage_path IS 'Ruta de almacenamiento de la imagen';
COMMENT ON COLUMN test_schema.imagenes.file_original_name IS 'Nombre original del archivo de la imagen';
COMMENT ON COLUMN test_schema.imagenes.image_name IS 'Nombre UUID de la imagen';
COMMENT ON COLUMN test_schema.imagenes.hash_sha256 IS 'Hash SHA256 de la imagen (64 caracteres hexadecimales)';
COMMENT ON COLUMN test_schema.imagenes.deleted IS 'Indica si la imagen ha sido eliminada (soft delete)';
COMMENT ON COLUMN test_schema.imagenes.date_created IS 'Fecha y hora de creación del registro';
COMMENT ON COLUMN test_schema.imagenes.date_deleted IS 'Fecha y hora de eliminación (soft delete)';

-- Crear tabla requests
CREATE TABLE IF NOT EXISTS test_schema.requests (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    endpoint VARCHAR(512) NOT NULL,
    date_created TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Índices para mejorar el rendimiento
CREATE INDEX idx_requests_user_id ON test_schema.requests(user_id);

-- Comentarios de la tabla y columnas
COMMENT ON TABLE test_schema.requests IS 'Tabla para almacenar información de requests/peticiones de usuarios';
COMMENT ON COLUMN test_schema.requests.id IS 'Identificador único del request';
COMMENT ON COLUMN test_schema.requests.user_id IS 'Identificador UUID del usuario que realizó el request (Keycloak sub)';
COMMENT ON COLUMN test_schema.requests.endpoint IS 'Endpoint/ruta del API que fue llamado';
COMMENT ON COLUMN test_schema.requests.date_created IS 'Fecha y hora en que se realizó el request';
