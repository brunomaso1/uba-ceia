#!/bin/bash
set -euo pipefail

# Configuración
DATE_FORMAT="%Y%m%d%H%M%S"
# SERVICE_BASE_DIR="/vagrant/modulo-gestor-datos/mongo" # Desde dentro de Vagrant (VM)
SERVICE_BASE_DIR="../modulo-gestor-datos/mongo" # Desde fuera de Vagrant (host)

log() {
    echo "[$(date +"$DATE_FORMAT")] $*"
}

ensure_backup_dir() {
    local backup_dir="$1"
    mkdir -p "$backup_dir"
}

backup() {
    local backup_dir="$1"
    local compose_args="$2"

    ensure_backup_dir "$backup_dir"
    local timestamp=$(date +"$DATE_FORMAT")
    local backup_file="$backup_dir/mongodb_backup_$timestamp.tar.gz"

    log "=== Iniciando backup de MongoDB ==="

    # Detener MongoDB para consistencia
    log "[Paso 1/3] Deteniendo MongoDB..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args stop mongodb)

    # Backup solo de los datos
    log "[Paso 2/3] Creando backup comprimido..."
    tar czvf "$backup_file" -C "$SERVICE_BASE_DIR" mongodb_data

    # Reiniciar servicio
    log "[Paso 3/3] Iniciando MongoDB..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args start mongodb)

    log "=== Backup completado: $(du -h "$backup_file" | cut -f1) ==="
}

restore() {
    local backup_file="$1"
    local compose_args="$2"

    log "=== Iniciando restauración de MongoDB ==="

    # Validaciones
    if [[ ! -f "$backup_file" ]]; then
        log "Error: Archivo de backup no encontrado: $backup_file"
        exit 1
    fi

    # Detener servicios
    log "[Paso 1/3] Deteniendo servicios..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args stop mongodb mongo-express)

    # Limpieza y restauración
    log "[Paso 2/3] Reemplazando datos..."
    rm -rf "${SERVICE_BASE_DIR}/mongodb_data"
    tar xzvf "$backup_file" -C "$SERVICE_BASE_DIR"

    # Reinicio
    log "[Paso 3/3] Reiniciando servicios..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args up -d)

    log "=== Restauración completada ==="
}

case "$1" in
backup)
    shift
    backup "$1" "$2"
    ;;
restore)
    shift
    restore "$1" "$2"
    ;;
*)
    echo "Uso: $0 {backup|restore} <backup_dir/backup_file> <compose_args>"
    exit 1
    ;;
esac
