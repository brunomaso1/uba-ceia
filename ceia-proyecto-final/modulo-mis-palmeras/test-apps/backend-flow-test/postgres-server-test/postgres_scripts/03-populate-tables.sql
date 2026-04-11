-- -- Script de inicialización: Poblar tablas con datos de prueba
-- -- Se ejecuta después de 02-create-tables.sql

-- SET search_path TO test_schema, public;

-- -- Insertar datos de prueba en la tabla imagenes
-- INSERT INTO test_schema.imagenes (storage_path, hash_sha256, deleted, date_created, date_deleted) VALUES
--     ('/storage/images/2024/01/img001.jpg', 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2', FALSE, '2024-01-15 10:30:00', NULL),
--     ('/storage/images/2024/01/img002.jpg', 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3', FALSE, '2024-01-16 14:20:00', NULL),
--     ('/storage/images/2024/02/img003.jpg', 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4', FALSE, '2024-02-10 09:15:00', NULL),
--     ('/storage/images/2024/02/img004.jpg', 'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5', TRUE, '2024-02-20 16:45:00', '2024-03-01 11:30:00'),
--     ('/storage/images/2024/03/img005.jpg', 'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6', TRUE, '2024-03-05 08:00:00', '2024-03-06 18:20:00');

-- -- Verificar los datos insertados
-- SELECT COUNT(*) as total_imagenes FROM test_schema.imagenes;
-- SELECT COUNT(*) as imagenes_activas FROM test_schema.imagenes WHERE deleted = FALSE;
-- SELECT COUNT(*) as imagenes_eliminadas FROM test_schema.imagenes WHERE deleted = TRUE;

-- -- Insertar datos de prueba en la tabla requests
-- INSERT INTO test_schema.requests (user_id, endpoint, date_created) VALUES
--     ('user_001', '/api/v1/images', '2024-01-15 10:30:15'),
--     ('user_001', '/api/v1/images/upload', '2024-01-15 10:32:20'),
--     ('user_002', '/api/v1/images', '2024-01-16 14:20:45'),
--     ('user_003', '/api/v1/images/search', '2024-02-10 09:15:30'),
--     ('user_002', '/api/v1/images/delete', '2024-02-20 16:45:10'),
--     ('user_001', '/api/v1/images', '2024-03-05 08:00:25'),
--     ('user_003', '/api/v1/images/upload', '2024-03-06 18:20:50'),
--     ('user_002', '/api/v1/images/search', '2024-03-07 12:10:00');

-- -- Verificar los datos insertados en requests
-- SELECT COUNT(*) as total_requests FROM test_schema.requests;
-- SELECT user_id, COUNT(*) as num_requests FROM test_schema.requests GROUP BY user_id ORDER BY num_requests DESC;
-- SELECT endpoint, COUNT(*) as num_calls FROM test_schema.requests GROUP BY endpoint ORDER BY num_calls DESC;
