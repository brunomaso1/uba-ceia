#!/bin/bash
set -euo pipefail

# ==========================================================================================
#                                   Configuración
# ==========================================================================================
root_path="/vagrant"

# Función para loguear mensajes con timestamp
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Función para iniciar un servicio usando docker compose en un directorio específico
start_service() {
  local path="$1"
  local compose_args="${2:-}"
  log "Iniciando servicio en ${path}..."
  # Se ejecuta en un subshell para no modificar el directorio actual del script
  (cd "$path" && docker compose $compose_args up -d) || {
    log "Error al iniciar el servicio en ${path}"
    exit 1
  }
  log "Servicio en ${path} iniciado correctamente."
}

# Esperar hasta que docker esté disponible
log "Esperando a que Docker esté disponible..."
until docker info > /dev/null 2>&1; do
  log "Docker no está disponible aún. Reintentando en 2 segundos..."
  sleep 2
done

log "Docker está disponible. Iniciando servicios de desarrollo..."

# ==========================================================================================
#                                   Iniciar servicios
# ==========================================================================================

# MODULO REPOSITORIO OBJETOS
# MinIO
start_service "${root_path}/modulo-repositorio-objetos/minio" "--env-file .env.dev"

# MODULO SEGURIDAD
# Lldap
# Copiar lldap_data a ubicación local para evitar conflictos con SQLite en /vagrant
log "Preparando directorio local para LLDAP..."
mkdir -p /home/vagrant/lldap_local_data
cp -r "${root_path}/lldap_data/"* /home/vagrant/lldap_local_data/ 2>/dev/null || true
chown -R 1000:1000 /home/vagrant/lldap_local_data
chmod -R 755 /home/vagrant/lldap_local_data
log "Datos de LLDAP copiados a /home/vagrant/lldap_local_data con permisos configurados"
start_service "${root_path}/modulo-seguridad/lldap" "--env-file .env.dev"

# Keycloak
log "Preparando directorio local para Keycloak..."
mkdir -p /home/vagrant/postgres_data
cp -r "${root_path}/modulo-seguridad/keycloak/postgres_data/"* /home/vagrant/postgres_data/ 2>/dev/null || true
chown -R 1000:1000 /home/vagrant/postgres_data
chmod -R 755 /home/vagrant/postgres_data
log "Datos de Keycloak copiados a /home/vagrant/postgres_data con permisos configurados"
start_service "${root_path}/modulo-seguridad/keycloak" "--env-file .env.dev"

# SSP
start_service "${root_path}/modulo-seguridad/ldap-self-service-password" ""

# Entrypoint
start_service "${root_path}/modulo-seguridad/entrypoint" "--env-file .env.dev"

# MODULO MIS PALMERAS
# Mis palmeras APP
start_service "${root_path}/modulo-mis-palmeras/mis-palmeras-app" "--env-file .env.dev"

# Landing page
start_service "${root_path}/modulo-mis-palmeras/mis-palmeras-landing-page" ""

# Maintenance page
start_service "${root_path}/modulo-mis-palmeras/mis-palmeras-maintenance" ""

# MODULO ETIQUETADO DATOS
# CVAT
start_service "${root_path}/modulo-etiquetado-datos/cvat" "--env-file .env.dev -f docker-compose.yml -f docker-compose.custom.yml"

# FiftyOne
# start_service "${root_path}/modulo-calidad-datos/fiftyone" ""

# MongoDB
# start_service "${root_path}/modulo-gestor-datos/mongodb" "" # Se tiene que levantar local por problemas de virtualizacion de la VM

# Mlflow
start_service "${root_path}/modulo-reportes/mlflow" ""