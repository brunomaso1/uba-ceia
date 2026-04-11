-- Script de inicialización: Crear esquema de prueba
-- Se ejecuta como el usuario POSTGRES_USER definido en el compose.yaml

-- Crear esquema de prueba
CREATE SCHEMA IF NOT EXISTS test_schema;

-- Establecer el esquema como parte del search_path
ALTER DATABASE postgresdb SET search_path TO test_schema, public;

-- Comentario del esquema
COMMENT ON SCHEMA test_schema IS 'Esquema de prueba para desarrollo y testing';
