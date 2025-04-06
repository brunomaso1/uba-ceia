#!/bin/bash
set -euo pipefail

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

# Inicia cada servicio

# MinIO
start_service "/vagrant/modulo-repositorio-objetos/minio" ""

# Lldap
start_service "/vagrant/modulo-seguridad/lldap" "--env-file .env.dev"

# SSP
start_service "/vagrant/modulo-seguridad/ldap-self-service-password" ""

# Landing page
start_service "/vagrant/modulo-aplicaciones-web/landing-page" ""

# CVAT
start_service "/vagrant/modulo-etiquetado-datos/cvat" "--env-file .env.dev -f docker-compose.yml -f docker-compose.custom.yml"

# Entrypoint
start_service "/vagrant/modulo-aplicaciones-web/entrypoint" "-f docker-compose.traefik.dev.yml"