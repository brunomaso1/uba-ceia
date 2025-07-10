#!/bin/bash

# Configura el enrutamiento para priorizar eth1 (bridge)
cat <<EOF > /etc/netplan/60-custom.yaml
network:
  version: 2
  ethernets:
    eth1:
      dhcp4: true
      dhcp4-overrides:
        route-metric: 50  # Prioridad más alta que eth0
EOF

# Aplica la configuración
netplan apply

sleep 5