#!/bin/bash
# 00-resize-disk-hyperv.sh
# Script para redimensionar particiones LVM en Ubuntu automáticamente

set -e # Detener el script si hay un error crítico

LOG_PREFIX="[DISK RESIZE]"

echo "$LOG_PREFIX Iniciando verificación de espacio en disco..."

# 1. Instalar cloud-guest-utils si no tenemos 'growpart'
if ! command -v growpart &> /dev/null; then
    echo "$LOG_PREFIX Instalando utilidades necesarias (cloud-guest-utils)..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq cloud-guest-utils
fi

# 2. Extender la partición física (Asumimos partición 3 para LVM en estas boxes de Ubuntu)
# El '|| true' asegura que el script no falle si la partición ya tiene el tamaño máximo.
echo "$LOG_PREFIX Intentando extender partición /dev/sda 3..."
growpart /dev/sda 3 || true

# 3. Redimensionar el Physical Volume (PV) de LVM
echo "$LOG_PREFIX Actualizando Physical Volume (LVM)..."
pvresize /dev/sda3 || true

# 4. Extender el Logical Volume (LV) para usar todo el espacio libre
# Usamos la ruta específica que vimos en tu 'df -h'
LV_PATH="/dev/mapper/ubuntu--vg-ubuntu--lv"
echo "$LOG_PREFIX Extendiendo Logical Volume ($LV_PATH)..."
lvextend -l +100%FREE "$LV_PATH" || true

# 5. Redimensionar el sistema de archivos ext4
echo "$LOG_PREFIX Redimensionando sistema de archivos..."
resize2fs "$LV_PATH" || true

echo "$LOG_PREFIX ¡Proceso finalizado! Espacio actual:"
df -h / | grep /