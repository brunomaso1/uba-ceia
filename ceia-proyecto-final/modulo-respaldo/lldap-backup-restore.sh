#!/bin/bash
set -euo pipefail

# Configuración
DATE_FORMAT="%Y-%m-%d_%H-%M-%S"
SERVICE_BASE_DIR="/vagrant/modulo-seguridad/lldap"

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
    local backup_file="$backup_dir/lldap_backup_$timestamp.tar.gz"

    log "=== Iniciando backup ==="

    # Paso 1: Detener y limpiar
    log "[Paso 1/3] Deteniendo servicio..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args rm -sf lldap || true)

    # Paso 2: Crear backup
    log "[Paso 2/3] Creando backup comprimido..."
    tar czvf "$backup_file" -C "$SERVICE_BASE_DIR" lldap_data

    # Paso 3: Reiniciar servicio
    log "[Paso 3/3] Reiniciando LLDAP..."
    # (cd "$SERVICE_BASE_DIR" && docker compose $compose_args up -d lldap)
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args up -d lldap)

    log "=== Backup completado: $(du -h "$backup_file" | cut -f1) ==="
}

restore() {
    local backup_file="$1"
    local compose_args="$2"

    log "=== Iniciando restauración ==="

    # Validar archivo de backup
    if [[ ! -f "$backup_file" ]]; then
        log "Error: Archivo de backup no encontrado: $backup_file"
        exit 1
    fi

    # Detener servicio
    log "[Paso 1/4] Deteniendo servicio..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args rm -sf lldap || true)

    # Eliminar datos actuales
    log "[Paso 2/4] Eliminando datos existentes..."
    rm -rf "${SERVICE_BASE_DIR}/lldap_data"

    # Restaurar backup
    log "[Paso 3/4] Restaurando archivo $backup_file..."
    tar xzvf "$backup_file" -C "$SERVICE_BASE_DIR"

    # Iniciar servicio
    log "[Paso 4/4] Reiniciando servicio..."
    (cd "$SERVICE_BASE_DIR" && docker compose $compose_args up -d lldap)

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
    echo "Ejemplos:"
    echo "  Backup:   $0 backup \"/vagrant/backups\" \"--env-file .env.dev\""
    echo "  Restore:  $0 restore \"/vagrant/backups/lldap_backup_2025-04-05.tar.gz\" \"--env-file .env.dev\""
    exit 1
    ;;
esac
