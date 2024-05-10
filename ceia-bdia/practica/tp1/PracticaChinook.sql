-- Selecciona todos los registros de la tabla Albums.
--SELECT * FROM album;

-- Selecciona todos los géneros únicos de la tabla Genres.
--SELECT DISTINCT (name) FROM genre;

-- Cuenta el número de pistas por género.
--SELECT g.name, (SELECT COUNT(*) FROM track t WHERE t.genre_id = g.genre_id) as cantidad FROM genre g;

-- Encuentra la longitud total (en milisegundos) de todas las pistas para cada álbum.
--SELECT  a.title, (SELECT SUM(t.milliseconds) FROM track t WHERE t.album_id = a.album_id) as "longitud acumulada" FROM album a;

-- Lista los 10 álbumes con más pistas.
--SELECT a.title, COUNT(t.track_id) as cantidad FROM album a JOIN track t ON a.album_id = t.album_id GROUP BY a.album_id ORDER BY COUNT(t.track_id) DESC LIMIT 10;

-- Encuentra la longitud promedio de la pista para cada género.
--SELECT g.name, (SELECT ROUND(AVG(t.milliseconds)) AS promedio FROM track t WHERE t.genre_id = g.genre_id) FROM genre g;

-- Para cada cliente, encuentra la cantidad total que han gastado.
--SELECT c.customer_id, CONCAT(c.first_name, ' ', c.last_name), (SELECT SUM(i.total) FROM invoice i WHERE i.customer_id = c.customer_id) as "total factura" FROM customer c;

-- Para cada país, encuentra la cantidad total gastada por los clientes.
--SELECT c.country, SUM(i.total) AS total FROM customer c JOIN invoice i ON c.customer_id = i.customer_id GROUP BY c.country;

-- Clasifica a los clientes en cada país por la cantidad total que han gastado.
/* SELECT c.country, c.customer_id, CONCAT(c.first_name, ' ', c.last_name), SUM(i.total) AS total 
    FROM customer c JOIN invoice i ON c.customer_id = i.customer_id 
    GROUP BY c.country, c.customer_id ORDER BY c.country, SUM(i.total) DESC; */

-- Para cada artista, encuentra el álbum con más pistas y clasifica a los artistas por este número.

-- Selecciona todas las pistas que tienen la palabra "love" en su título.
--SELECT * FROM track WHERE LOWER(name) LIKE '%love%';

-- Selecciona a todos los clientes cuyo primer nombre comienza con 'A'.
--SELECT * FROM customer WHERE first_name LIKE 'A%';

-- Calcula el porcentaje del total de la factura que representa cada factura.
--SELECT invoice_id, CONCAT(ROUND((total / (SELECT SUM(total) FROM invoice)) * 100, 2), ' %')  AS porcentaje FROM invoice;

-- Calcula el porcentaje de pistas que representa cada género.
--SELECT g.name, CONCAT(ROUND((SELECT COUNT(*) FROM track t WHERE t.genre_id = g.genre_id)::DECIMAL / (SELECT COUNT(*) FROM track) * 100, 2), ' %') AS porcentaje FROM genre g;

-- Para cada cliente, compara su gasto total con el del cliente que gastó más.
SELECT customer_id, CONCAT(first_name, ' ', last_name),
    CONCAT(ROUND(
        (SELECT SUM(i.total) FROM invoice i WHERE i.customer_id = c.customer_id) / 
        (SELECT SUM(i.total) FROM invoice i GROUP BY i.customer_id ORDER BY SUM(i.total) DESC LIMIT 1) * 100), ' %')
        AS "porcentaje gasto total"
    FROM customer c;

-- Para cada factura, calcula la diferencia en el gasto total entre ella y la factura anterior.

-- Para cada factura, calcula la diferencia en el gasto total entre ella y la próxima factura.

-- Encuentra al artista con el mayor número de pistas para cada género.

-- Compara el total de la última factura de cada cliente con el total de su factura anterior.

-- Encuentra cuántas pistas de más de 3 minutos tiene cada álbum.
