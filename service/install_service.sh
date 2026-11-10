#!/bin/bash
# Run this ON THE PI, once:   bash /mnt/raspi-ssd/New/service/install_service.sh
# It copies the unit file to /etc/systemd/system (a tiny write on the SD card, which systemd requires) and starts the service. It asks for your sudo password.
set -e
sudo cp /mnt/raspi-ssd/New/service/new-optimized-web.service /etc/systemd/system/new-optimized-web.service
sudo systemctl daemon-reload
sudo systemctl enable --now new-optimized-web.service
sleep 3; systemctl --no-pager --lines=0 status new-optimized-web.service | head -8
echo; echo "Open  http://$(hostname -I | awk '{print $1}'):8000  in a browser (the first start can take about 30 seconds)."
