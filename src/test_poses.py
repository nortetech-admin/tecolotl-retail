import time
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from pose_detector import get_poses, print_pose

MODEL_PATH = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

imx500 = IMX500(MODEL_PATH)
picam2 = Picamera2(imx500.camera_num)
picam2.start(show_preview=True)

print("Cámara iniciada. Buscando personas...\n")

try:
    while True:
        metadata = picam2.capture_metadata()
        poses = get_poses(metadata, imx500)
        poses = [p for p in poses if p.score > 0.3]
        if poses:
            print(f"--- {len(poses)} persona(s) ---")
            for i, pose in enumerate(poses):
                print(f"\nPersona {i+1}:")
                print_pose(pose)
        else:
            print("Sin detecciones...")

        time.sleep(1)

except KeyboardInterrupt:
    print("\nDetenido.")
    picam2.stop()
