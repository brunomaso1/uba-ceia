docker pull postgres
docker run --name bdia-postgres-container -e POSTGRES_PASSWORD=postgres -p 5435:5432 -v pgdata:/var/lib/postgresql/data -d postgres
docker run --name bdia-postgres-container -e POSTGRES_PASSWORD=postgres -p 5435:5432 -d postgres
docker exec -it bdia-postgres-container bash

# docker run --name bdia-postgres-container -e POSTGRES_PASSWORD=docker_user -e POSTGRES_USER=docker_user -p 5432:5432 -d postgres
# user=postgres

docker run --name bdia-postgres-container -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=docker run --name learn_postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=docker_user -p 5433:5432 -v pgdata:/var/lib/postgresql/data -d postgres
 -p 5433:5432 -v pgdata:/var/lib/postgresql/data -d postgres

docker run --name bdia-postgres-container -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres -p 5433:5432 -v pgdata:/var/lib/postgresql/data -d postgres