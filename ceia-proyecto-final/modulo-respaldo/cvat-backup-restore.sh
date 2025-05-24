#!/bin/bash
set -euo pipefail

# Configuración
DATE_FORMAT="%Y%m%d%H%M%S"
SERVICE_BASE_DIR="/vagrant/modulo-etiquetado-datos/cvat"

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
    local backup_file="$backup_dir/cvat_backup_$timestamp.tar.gz"
    local temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT

    log "=== Iniciando backup de CVAT ==="

    log "[Paso 1/5] Deteniendo servicios..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args stop)

    log "[Paso 2/5] Creando directorio temporal..."
    mkdir -p "$temp_dir"

    log "[Paso 3/5] Respaldando volúmenes..."
    declare -A volume_map=(
        ["picudo-rojo-project_cvat_db"]="/var/lib/postgresql/data"
        ["picudo-rojo-project_cvat_data"]="/home/django/data"
        ["picudo-rojo-project_cvat_events_db"]="/var/lib/clickhouse"
    )

    for volume in "${!volume_map[@]}"; do
        log "Respaldo de volumen: $volume"
        docker run --rm \
            -v "$volume":"${volume_map[$volume]}" \
            -v "$temp_dir":/backup \
            alpine tar czf "/backup/${volume}.tar.gz" -C "${volume_map[$volume]}" .
    done

    log "[Paso 4/5] Aplicando permisos..."
    docker run --rm -v picudo-rojo-project_cvat_data:/volume alpine chmod -R 777 /volume

    log "[Paso 5/5] Reiniciando servicios y creando archivo único..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args up -d)

    log "Empaquetando backup en un solo archivo..."
    tar czf "$backup_file" -C "$temp_dir" .

    log "=== Backup completado en: $backup_file ==="
}

restore() {
    local backup_file="$1"
    local compose_args="$2"

    log "=== Iniciando restauración de CVAT ==="

    local temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT

    log "Extrayendo backup a directorio temporal..."
    tar xzf "$backup_file" -C "$temp_dir"

    log "[Paso 1/6] Validando archivos..."
    for volume in picudo-rojo-project_cvat_db picudo-rojo-project_cvat_data picudo-rojo-project_cvat_events_db; do
        if [[ ! -f "$temp_dir/${volume}.tar.gz" ]]; then
            log "Error: Falta archivo para volumen $volume"
            exit 1
        fi
    done

    log "[Paso 2/6] Deteniendo servicios..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args down)

    log "[Paso 3/6] Eliminando volúmenes antiguos..."
    docker volume rm picudo-rojo-project_cvat_db picudo-rojo-project_cvat_data picudo-rojo-project_cvat_events_db || true

    log "[Paso 4/6] Creando nuevos volúmenes..."
    docker volume create picudo-rojo-project_cvat_db
    docker volume create picudo-rojo-project_cvat_data
    docker volume create picudo-rojo-project_cvat_events_db

    log "[Paso 5/6] Restaurando datos..."
    for volume in picudo-rojo-project_cvat_db picudo-rojo-project_cvat_data picudo-rojo-project_cvat_events_db; do
        log "Restaurando: $volume"
        docker run --rm \
            -v "$volume":/restore \
            -v "$temp_dir":/backup \
            alpine sh -c "tar xzf /backup/${volume}.tar.gz -C /restore"
    done

    log "[Paso 6/6] Reiniciando servicios..."
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
    echo "Uso: $0 {backup|restore} <backup_dir/backup_file.tar.gz> <compose_args>"
    echo "Ejemplos:"
    echo "  Backup:  $0 backup \"/vagrant/backups/cvat\" \"--env-file .env.dev\""
    echo "  Restore: $0 restore \"/vagrant/backups/cvat/cvat_backup_2025-04-05_18-41-11.tar.gz\" \"--env-file .env.dev\""
    exit 1
    ;;
esac
