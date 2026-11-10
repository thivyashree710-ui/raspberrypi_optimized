#!/bin/bash
# Run on the Pi to remove the auto-start:   bash /mnt/raspi-ssd/New/service/uninstall_service.sh
set -e
sudo systemctl disable --now new-optimized-web.service || true
sudo rm -f /etc/systemd/system/new-optimized-web.service
sudo systemctl daemon-reload
echo "service removed"
