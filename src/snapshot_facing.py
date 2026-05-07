"""
snapshot_facing.py — Tecolotl Retail
======================================
Detecta personas DE ESPALDAS (mirando el anaquel) y guarda
una foto por cada detección. Espera 2 segundos entre fotos
para no disparar múltiples capturas del mismo momento.

Presiona Ctrl+C para terminar.
"""

import time
import cv2
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from pose_detector import get_poses
from shelf_attention import analyze_pose, get_zone_boundaries, IMAGE_WIDTH

MODEL_PATH = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
IMG_H      = 480
FONT       = cv2.FONT_HERSHEY_SIMPLEX

imx500 = IMX500(MODEL_PATH)
picam2 = Picamera2(imx500.camera_num)
picam2.configure(picam2.create_preview_configuration(
    main={"size": (IMAGE_WIDTH, IMG_H), "format": "RGB888"}
))
picam2.start()

print("Esperando personas mirando el anaquel... (Ctrl+C para terminar)\n")

count      = 1
last_saved = 0   # timestamp de la última foto

try:
    while True:
        frame_rgb = picam2.capture_array("main")
        metadata  = picam2.capture_metadata()

        poses = get_poses(metadata, imx500)
        poses = [p for p in poses if p.score > 0.2]

        for pose in poses:
            analysis = analyze_pose(pose)
            if analysis["facing_shelf"] is not True:
                continue

            now = time.time()
            if now - last_saved < 2:   # cooldown de 2 segundos
                continue

            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Hombros
            ls, rs = pose.left_shoulder, pose.right_shoulder
            if ls.is_valid() and rs.is_valid():
                cv2.circle(frame, (int(ls.x), int(ls.y)), 6, (86, 199, 29), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(rs.x), int(rs.y)), 6, (86, 199, 29), -1, cv2.LINE_AA)
                cv2.line(frame, (int(ls.x), int(ls.y)), (int(rs.x), int(rs.y)),
                         (86, 199, 29), 2, cv2.LINE_AA)

            # Líneas de zona
            for bx in [int(b) for b in get_zone_boundaries()]:
                cv2.line(frame, (bx, 0), (bx, IMG_H), (255, 255, 255), 2)

            # Badge
            zone = analysis["zone"]
            cv2.putText(frame, f"MIRANDO ANAQUEL | Zona {zone}",
                        (10, 30), FONT, 0.7, (86, 199, 29), 2, cv2.LINE_AA)

            fname = f"snapshot_{count:03d}_zona{zone}.png"
            cv2.imwrite(fname, frame)
            print(f"[{count}] {fname}")
            count     += 1
            last_saved = now
            break   # una foto por ciclo aunque haya varias personas

        time.sleep(0.1)

except KeyboardInterrupt:
    print(f"\nTerminado. {count - 1} fotos guardadas.")
finally:
    picam2.stop()