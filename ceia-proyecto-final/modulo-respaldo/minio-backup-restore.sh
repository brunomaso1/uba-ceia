#!/bin/bash
set -euo pipefail

# Configuración
DATE_FORMAT="%Y%m%d%H%M%S"
SERVICE_BASE_DIR="/vagrant/modulo-repositorio-objetos/minio"

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
    local backup_file="$backup_dir/minio_backup_$timestamp.tar.gz"

    log "=== Iniciando backup de MinIO ==="

    # Detener MinIO temporalmente
    log "[Paso 1/3] Deteniendo MinIO..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args stop minio)

    # Backup solo de los datos de MinIO
    log "[Paso 2/3] Creando backup comprimido..."
    tar czvf "$backup_file" -C "$SERVICE_BASE_DIR" minio-data

    # Reiniciar servicio
    log "[Paso 3/3] Reiniciando MinIO..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args start minio)

    log "=== Backup completado: $(du -h "$backup_file" | cut -f1) ==="
}

restore() {
    local backup_file="$1"
    local compose_args="$2"

    log "=== Iniciando restauración de MinIO ==="

    # Validaciones
    if [[ ! -f "$backup_file" ]]; then
        log "Error: Archivo de backup no encontrado: $backup_file"
        exit 1
    fi

    # Detener servicios dependientes
    log "[Paso 1/4] Deteniendo servicios..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args stop minio mc)

    # Limpiar y restaurar
    log "[Paso 2/4] Eliminando datos actuales..."
    rm -rf "${SERVICE_BASE_DIR}/minio-data"

    log "[Paso 3/4] Restaurando backup..."
    tar xzvf "$backup_file" -C "$SERVICE_BASE_DIR"

    log "[Paso 5/5] Aplicando permisos y reiniciando..."
    chmod -R 777 "${SERVICE_BASE_DIR}/minio-data" # MinIO requiere permisos amplios
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args up -d minio mc)

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
