#!/bin/bash

# sudo apt-mark hold grub-pc  # Para evitar que grub-pc se actualice automáticamente
apt-mark hold openssh-server

# Para evitar interacciones durante instalación (por ejemplo, grub-pc)
export DEBIAN_FRONTEND=noninteractive

# Aceptar la configuración de grub-pc sin interacción
echo "grub-pc grub-pc/install_devices multiselect /dev/sda" | sudo debconf-set-selections
echo "grub-pc grub-pc/install_devices_disks_changed boolean true" | sudo debconf-set-selections
echo "grub-pc grub-pc/install_devices_empty boolean false" | sudo debconf-set-selections
echo "grub-pc grub-pc/overwrite_other_os boolean true" | sudo debconf-set-selections

# Actualizar e instalar sin interacción
sudo apt-get -q update -y && \
sudo apt-get -q -y \
  -o Dpkg::Options::="--force-confdef" \
  -o Dpkg::Options::="--force-confold" \
  upgrade && \
# sudo apt-get -q full-upgrade -y && \
sudo apt-get -q autoremove -y && \
sudo apt-get -q clean -y && \
sudo apt-get -q autoclean -y

# Install required base packages
sudo apt -q install -y ca-certificates curl gnupg git

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

# Update again to include Docker packages
sudo apt -q update

# Install Docker and related tools
sudo apt -q install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add vagrant user to docker group
sudo usermod -aG docker vagrant

# Activar nuevamente la actualización de los paquetes desactivados
sudo apt-mark unhold openssh-server
