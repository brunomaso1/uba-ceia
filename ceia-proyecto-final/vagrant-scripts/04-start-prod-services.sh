#!/bin/bash
set -euo pipefail

# ==============================
# Configuración
# ==============================
root_path="/opt/ceia-proyecto-final"

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

log "Docker está disponible. Iniciando servicios de producción..."

# ==============================
# Iniciar servicios
# ==============================

# MODULO REPOSITORIO OBJETOS
# MinIO
start_service "${root_path}/modulo-repositorio-objetos/minio" "--env-file .env.prod"

# MODULO SEGURIDAD
# Lldap
start_service "${root_path}/modulo-seguridad/lldap" "--env-file .env.prod"

# Keycloak
start_service "${root_path}/modulo-seguridad/keycloak" "--env-file .env.prod"

# SSP
start_service "${root_path}/modulo-seguridad/ldap-self-service-password" ""

# Entrypoint
start_service "${root_path}/modulo-seguridad/entrypoint" "--env-file .env.prod"

# MODULO MIS PALMERAS
# Mis palmeras APP
start_service "${root_path}/modulo-mis-palmeras/mis-palmeras-app" "--env-file .env.prod"

# Landing page
start_service "${root_path}/modulo-mis-palmeras/mis-palmeras-landing-page" ""

# Maintenance landing page
start_service "${root_path}/modulo-mis-palmeras/mis-palmeras-maintenance" ""

# MODULO ETIQUETADO DATOS
# CVAT
start_service "${root_path}/modulo-etiquetado-datos/cvat" "--env-file .env.prod -f docker-compose.yml -f docker-compose.custom.yml"