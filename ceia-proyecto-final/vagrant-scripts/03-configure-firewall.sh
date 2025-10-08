#!/bin/bash
set -euo pipefail

# ========================
# Configuración de firewall UFW
# ========================

# Detectar subred local (ajusta si no es 192.168.0.0/24)
LAN_SUBNET="192.168.0.0/24"

echo "[INFO] Configurando firewall UFW..."

# Instalar UFW si no está
if ! command -v ufw &>/dev/null; then
  echo "[INFO] UFW no está instalado. Instalando..."
  sudo apt-get update && sudo apt-get install -y ufw
fi

# Resetear configuración previa
echo "[INFO] Reseteando configuración previa de UFW..."
sudo ufw --force reset

# Políticas por defecto: denegar entrada, permitir salida
sudo ufw default deny incoming
sudo ufw default allow outgoing

# -------------------------
# Reglas globales (Internet)
# -------------------------
echo "[INFO] Permitiendo SSH, HTTP y HTTPS desde cualquier lugar..."
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS

# -------------------------
# Reglas solo LAN
# -------------------------
echo "[INFO] Permitiendo acceso SOLO desde LAN (${LAN_SUBNET}) a Plex y NAS..."
sudo ufw allow from $LAN_SUBNET to any port 32400 proto tcp  # Plex
sudo ufw allow from $LAN_SUBNET to any port 445 proto tcp    # NAS/SMB
# Si usas NFS:
# sudo ufw allow from $LAN_SUBNET to any port 2049 proto tcp

# -------------------------
# Activar firewall
# -------------------------
echo "[INFO] Activando UFW..."
sudo ufw --force enable

echo "[INFO] Firewall configurado correctamente. Reglas activas:"
sudo ufw status verbose
