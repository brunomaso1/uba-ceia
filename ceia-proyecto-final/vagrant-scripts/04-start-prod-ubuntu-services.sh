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

# ==============================
# Iniciar servicios
# ==============================

# MinIO
start_service "$root_path/modulo-repositorio-objetos/minio" ""

# Lldap
start_service "$root_path/modulo-seguridad/lldap" "--env-file .env.prod"

# SSP
start_service "$root_path/modulo-seguridad/ldap-self-service-password" ""

# Landing page
start_service "$root_path/modulo-aplicaciones-web/landing-page" ""

# CVAT
start_service "$root_path/modulo-etiquetado-datos/cvat" "--env-file .env.prod -f docker-compose.yml -f docker-compose.custom.yml"

# Entrypoint
start_service "$root_path/modulo-aplicaciones-web/entrypoint" "-f docker-compose.traefik.prod.yml"
