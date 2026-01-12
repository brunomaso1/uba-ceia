#!/usr/bin/env bash
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run as root (sudo)."
  exit 1
fi

echo "[1/7] Updating APT index"
apt-get update -y

echo "[2/7] Installing base packages"
apt-get install -y curl gnupg2 lsb-release pciutils

echo "[3/7] Checking for NVIDIA GPU presence"
if ! lspci | grep -i -q 'nvidia'; then
  echo "No NVIDIA GPU detected (lspci). Continuing (driver install will likely be useless)."
fi

echo "[4/7] Installing NVIDIA driver (server variant)"
if command -v nvidia-smi &>/dev/null; then
  echo "Driver already present:"
  nvidia-smi || true
else
  # Fixed version (adjust if needed)
  ubuntu-drivers list --gpgpu || true
  ubuntu-drivers install --gpgpu nvidia:580-server
  apt-get install -y nvidia-utils-580-server
fi

echo "[5/7] Adding NVIDIA Container Toolkit repository (official secure method)"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y

echo "[6/7] Installing NVIDIA Container Toolkit (pinned versions)"
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1
apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

echo "[7/7] Configuring Docker runtime"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "Testing: nvidia-smi"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi || echo "nvidia-smi executed but driver may not be active."
else
  echo "nvidia-smi not found."
fi

echo "Testing Docker GPU access"
if command -v docker &>/dev/null; then
  docker run --rm --gpus all --pull always nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi || echo "Docker GPU test failed."
else
  echo "Docker not installed."
fi

echo "Done."