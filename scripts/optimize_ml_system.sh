#!/bin/bash

echo "[INFO] Removing any existing swap files..."
sudo swapoff /home/user/swapfile 2>/dev/null
sudo rm -f /home/user/swapfile 2>/dev/null
sudo sed -i '/\/home\/user\/swapfile/d' /etc/fstab

echo "[INFO] Enabling dynamic swap usage (no fixed limit)..."
sudo swapon -a

echo "[INFO] Setting swappiness to 10 (use RAM first, swap only when necessary)..."
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

echo "[INFO] Optimizing GPU memory allocation for PyTorch..."
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

echo "[INFO] Installing and enabling ZRAM (faster swap in RAM)..."
sudo apt update && sudo apt install -y zram-tools
echo 'ALGO=lz4' | sudo tee /etc/default/zramswap
echo 'PERCENTAGE=100' | sudo tee -a /etc/default/zramswap
sudo systemctl restart zramswap

echo "[INFO] System optimization complete! Checking system swap settings..."
free -h
sysctl vm.swappiness
