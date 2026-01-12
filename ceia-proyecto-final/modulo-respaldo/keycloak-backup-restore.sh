#!/bin/bash
set -euo pipefail

# =====================
# Configuración
# =====================
DATE_FORMAT="%Y%m%d%H%M%S"
SERVICE_BASE_DIR="../modulo-seguridad/keycloak"
DATA_DIR="postgres_data"

log() {
    echo "[$(date +"$DATE_FORMAT")] $*"
}

ensure_backup_dir() {
    local backup_dir="$1"
    mkdir -p "$backup_dir"
}

# =====================
# Backup
# =====================
backup() {
    local backup_dir="$1"
    local compose_args="$2"

    ensure_backup_dir "$backup_dir"
    local timestamp=$(date +"$DATE_FORMAT")
    local backup_file="$backup_dir/keycloak_backup_$timestamp.tar.gz"

    log "=== Iniciando backup de Keycloak (PostgreSQL) ==="

    # Detener Keycloak y Postgres
    log "[Paso 1/3] Deteniendo servicios Keycloak y PostgreSQL..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args stop)

    # Crear backup
    log "[Paso 2/3] Creando backup comprimido..."
    tar czvf "$backup_file" -C "$SERVICE_BASE_DIR" "$DATA_DIR"

    # Reiniciar servicios
    log "[Paso 3/3] Reiniciando servicios..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args up -d)

    log "=== Backup completado: $(du -h "$backup_file" | cut -f1) ==="
}

# =====================
# Restore
# =====================
restore() {
    local backup_file="$1"
    local compose_args="$2"

    log "=== Iniciando restauración de Keycloak (PostgreSQL) ==="

    if [[ ! -f "$backup_file" ]]; then
        log "Error: Archivo de backup no encontrado: $backup_file"
        exit 1
    fi

    # Detener servicios
    log "[Paso 1/4] Deteniendo servicios..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args down)

    # Eliminar datos actuales
    log "[Paso 2/4] Eliminando datos actuales de PostgreSQL..."
    rm -rf "${SERVICE_BASE_DIR:?}/${DATA_DIR}"

    # Restaurar backup
    log "[Paso 3/4] Restaurando backup..."
    tar xzvf "$backup_file" -C "$SERVICE_BASE_DIR"

    # Permisos y arranque
    log "[Paso 4/4] Aplicando permisos y levantando servicios..."
    chmod -R 777 "${SERVICE_BASE_DIR}/${DATA_DIR}"
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args up -d)

    log "=== Restauración completada ==="
}

# =====================
# Dispatcher
# =====================
case "${1:-}" in
backup)
    shift
    backup "$1" "${2:-}"
    ;;
restore)
    shift
    restore "$1" "${2:-}"
    ;;
*)
    echo "Uso: $0 {backup|restore} <backup_dir|backup_file> [compose_args]"
    exit 1
    ;;
esac