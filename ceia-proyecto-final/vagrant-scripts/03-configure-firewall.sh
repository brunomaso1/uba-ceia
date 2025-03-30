#!/bin/bash
set -e

# Verificar si ufw está instalado
if ! command -v ufw &> /dev/null; then
  echo "ufw no está instalado. Instalándolo..."
  sudo apt-get update && sudo apt-get install -y ufw
fi

# Permitir tráfico HTTP, HTTPS y SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 22/tcp    # SSH

# Activar el firewall de forma no interactiva
sudo ufw --force enable

echo "Firewall configurado y activado."