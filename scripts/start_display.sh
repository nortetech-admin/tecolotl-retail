#bash /home/filifloresb/Documents/tecolotl/retail/scripts/start_display.sh
#!/bin/bash
rpicam-hello -t 0s \
  --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json \
  --viewfinder-width 1920 \
  --viewfinder-height 1080 \
  --width 1920 \
  --height 1080 \
  --framerate 30 \
  --fullscreen
