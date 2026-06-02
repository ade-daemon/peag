#!/bin/bash
set -e

# PEAG k3s installer
# Installs k3s with the right config for PEAG

echo "Installing k3s..."

curl -sfL https://get.k3s.io | sh -s - \
  --disable traefik \
  --disable servicelb \
  --write-kubeconfig-mode 644 \
  --node-label "node-type=cpu" \
  --node-label "peag=true"

echo "Waiting for k3s to be ready..."
sleep 10
until kubectl get nodes | grep -q "Ready"; do
  sleep 2
done

echo "k3s is ready."

# Copy kubeconfig to standard location
mkdir -p ~/.kube
cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
chmod 600 ~/.kube/config

echo "kubectl configured."
