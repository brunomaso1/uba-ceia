-- Selecciona todos los registros de la tabla Albums.
SELECT * FROM album;

-- Selecciona todos los géneros únicos de la tabla Genres.
SELECT DISTINCT (name) FROM genre;

-- Cuenta el número de pistas por género.
SELECT g.name, (SELECT COUNT(*) FROM track t WHERE t.genre_id = g.genre_id) as cantidad FROM genre g;

-- Encuentra la longitud total (en milisegundos) de todas las pistas para cada álbum.
SELECT  a.title, (SELECT SUM(t.milliseconds) FROM track t WHERE t.album_id = a.album_id) as "longitud acumulada" FROM album a;

-- Lista los 10 álbumes con más pistas.
SELECT a.title, COUNT(t.track_id) as cantidad FROM album a JOIN track t ON a.album_id = t.album_id GROUP BY a.album_id ORDER BY COUNT(t.track_id) DESC LIMIT 10;

-- Encuentra la longitud promedio de la pista para cada género.
SELECT g.name, (SELECT ROUND(AVG(t.milliseconds)) AS promedio FROM track t WHERE t.genre_id = g.genre_id) FROM genre g;

-- Para cada cliente, encuentra la cantidad total que han gastado.
SELECT c.customer_id, CONCAT(c.first_name, ' ', c.last_name), (SELECT SUM(i.total) FROM invoice i WHERE i.customer_id = c.customer_id) as "total factura" FROM customer c;

-- Para cada país, encuentra la cantidad total gastada por los clientes.
SELECT c.country, SUM(i.total) AS total FROM customer c JOIN invoice i ON c.customer_id = i.customer_id GROUP BY c.country;

-- Clasifica a los clientes en cada país por la cantidad total que han gastado.
SELECT c.country, c.customer_id, CONCAT(c.first_name, ' ', c.last_name) AS nombre, NTILE(4) OVER (PARTITION BY c.country ORDER BY s.total_gastado DESC) AS cuartil FROM customer c 
    JOIN (SELECT i.customer_id, SUM(total) AS total_gastado FROM invoice i GROUP BY i.customer_id) AS s
        ON s.customer_id = c.customer_id;

-- Para cada artista, encuentra el álbum con más pistas y clasifica a los artistas por este número.
WITH s1 AS (SELECT t.album_id, COUNT(t.track_id) AS cant_pistas
                            FROM track t 
                            GROUP BY t.album_id),
    s2 AS (SELECT a.artist_id, MAX(s1.cant_pistas) AS max_pistas FROM album a JOIN s1
        ON s1.album_id = a.album_id
        GROUP BY a.artist_id)
SELECT a.album_id, a.title, a.artist_id, s2.max_pistas, DENSE_RANK() OVER (ORDER BY max_pistas DESC) as ranking
    FROM album a 
        JOIN s1 ON a.album_id = s1.album_id 
        JOIN s2 ON s2.artist_id = a.artist_id 
    WHERE s1.cant_pistas = s2.max_pistas 
    ORDER BY ranking;

-- Selecciona todas las pistas que tienen la palabra "love" en su título.
SELECT * FROM track WHERE LOWER(name) LIKE '%love%';

-- Selecciona a todos los clientes cuyo primer nombre comienza con 'A'.
SELECT * FROM customer WHERE first_name LIKE 'A%';

-- Calcula el porcentaje del total de la factura que representa cada factura.
SELECT invoice_id, CONCAT(ROUND((total / (SELECT SUM(total) FROM invoice)) * 100, 2), ' %')  AS porcentaje FROM invoice;

-- Calcula el porcentaje de pistas que representa cada género.
SELECT g.name, CONCAT(ROUND((SELECT COUNT(*) FROM track t WHERE t.genre_id = g.genre_id)::DECIMAL / (SELECT COUNT(*) FROM track) * 100, 2), ' %') AS porcentaje FROM genre g;

-- Para cada cliente, compara su gasto total con el del cliente que gastó más.
SELECT customer_id, CONCAT(first_name, ' ', last_name),
    CONCAT(ROUND(
        (SELECT SUM(i.total) FROM invoice i WHERE i.customer_id = c.customer_id) / 
        (SELECT SUM(i.total) FROM invoice i GROUP BY i.customer_id ORDER BY SUM(i.total) DESC LIMIT 1) * 100), ' %')
        AS "porcentaje gasto total"
    FROM customer c;

-- Para cada factura, calcula la diferencia en el gasto total entre ella y la factura anterior.
SELECT i.*, (i.total - LAG(i.total, 1) OVER ()) AS diferencia FROM invoice i;

-- Para cada factura, calcula la diferencia en el gasto total entre ella y la próxima factura.
SELECT i.*, (LEAD(i.total, 1) OVER () - i.total) AS diferencia FROM invoice i;

-- Encuentra al artista con el mayor número de pistas para cada género.
WITH t AS (SELECT COUNT(t.track_id) AS "cant_pistas", t.genre_id, a.artist_id 
                FROM track t 
                    JOIN album b ON t.album_id = b.album_id
                    JOIN artist a ON a.artist_id = b.artist_id
                GROUP BY t.genre_id, a.artist_id),
    t2 AS (SELECT t.genre_id, t.artist_id, t.cant_pistas, ROW_NUMBER() OVER (PARTITION BY t.genre_id ORDER BY cant_pistas DESC) FROM t)
SELECT (SELECT name FROM genre WHERE genre_id = t2.genre_id), genre_id, 
        (SELECT name FROM artist WHERE artist_id = t2.artist_id),
        t2.artist_id, 
        t2.cant_pistas FROM t2 WHERE row_number = 1;

/* SELECT (SELECT name FROM genre WHERE genre_id = s3.genre_id), 
        s3.genre_id, 
        (SELECT name FROM artist WHERE artist_id = s3.artist_id),
        s3.artist_id, 
        s3.cant_pistas
    FROM (SELECT s.genre_id, max(cant_pistas) AS cant_pistas 
            FROM (SELECT COUNT(t.track_id) AS "cant_pistas", t.genre_id, a.artist_id 
                FROM track t 
                    JOIN album b ON t.album_id = b.album_id
                    JOIN artist a ON a.artist_id = b.artist_id
                GROUP BY t.genre_id, a.artist_id) s -- Cuenta cantidad agrupado por genero y artista.
            GROUP BY s.genre_id) s2 
        JOIN (SELECT COUNT(t.track_id) AS "cant_pistas", t.genre_id, a.artist_id 
            FROM track t 
                JOIN album b ON t.album_id = b.album_id
                JOIN artist a ON a.artist_id = b.artist_id
            GROUP BY t.genre_id, a.artist_id) s3
        ON s2.genre_id = s3.genre_id AND s2.cant_pistas = s3.cant_pistas
    ORDER BY s3.genre_id; */

-- Compara el total de la última factura de cada cliente con el total de su factura anterior.
WITH t1 AS (SELECT i.* FROM invoice i 
            JOIN (SELECT ROW_NUMBER() OVER 
                        (PARTITION BY customer_id ORDER BY invoice_date DESC) AS row_id, 
                        invoice_id 
                    FROM invoice) s 
                ON s.invoice_id = i.invoice_id WHERE s.row_id = 1),
    t2 AS (SELECT i.* FROM invoice i 
            JOIN (SELECT ROW_NUMBER() OVER 
                        (PARTITION BY customer_id ORDER BY invoice_date DESC) AS row_id, 
                        invoice_id 
                    FROM invoice) s 
                ON s.invoice_id = i.invoice_id WHERE s.row_id = 2)
SELECT t1.customer_id, (SELECT CONCAT(first_name, ' ', last_name) AS nombre FROM customer WHERE customer_id = t1.customer_id), t1.total AS last_invoice, t2.total AS prelast_invoice, t1.total - t2.total AS diferencia 
    FROM t1 JOIN t2 ON t1.customer_id = t2.customer_id;

-- Encuentra cuántas pistas de más de 3 minutos tiene cada álbum.
SELECT t.album_id, (SELECT title FROM album WHERE album_id = t.album_id), COUNT(t.track_id) 
    FROM track t WHERE t.milliseconds > 3*60000 
    GROUP BY t.album_id;