#!/bin/bash
set -e

# Usar las variables de entorno correctas
echo "Initializing MongoDB with following parameters:"
echo "Admin Username: $MONGO_INITDB_ROOT_USERNAME"
echo "Database: $MONGODB_INITDB_DATABASE"
echo "User: $MONGODB_USER"

mongosh --quiet -u "$MONGO_INITDB_ROOT_USERNAME" -p "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin <<EOF
// Crear la base de datos y colección
use $MONGODB_INITDB_DATABASE
db.createCollection("imagenes")

// Crear usuario regular
db.createUser({
  user: "$MONGODB_USER",
  pwd: "$MONGODB_PASSWORD",
  roles: [{
    role: "readWrite",
    db: "$MONGODB_INITDB_DATABASE"
  }]
})
EOF