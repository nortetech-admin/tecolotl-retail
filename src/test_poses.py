import time
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from pose_detector import get_poses, Pose

MODEL_PATH = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

def is_facing_shelf(pose: Pose, min_shoulder_width: int = 40) -> bool:
    ls = pose.left_shoulder
    rs = pose.right_shoulder
    if not ls.is_valid() or not rs.is_valid():
        return None
    shoulder_width = abs(rs.x - ls.x)
    if shoulder_width < min_shoulder_width:
        return None
    return ls.x < rs.x

imx500 = IMX500(MODEL_PATH)
picam2 = Picamera2(imx500.camera_num)
picam2.start(show_preview=True)

print("Cámara iniciada.\n")

try:
    while True:
        metadata = picam2.capture_metadata()
        poses = get_poses(metadata, imx500)
        poses = [p for p in poses if p.score > 0.3]

        if poses:
            for i, pose in enumerate(poses):
                result = is_facing_shelf(pose)
                if result is True:
                    print(f"Persona {i+1}: DE ESPALDAS — mirando anaquel")
                elif result is False:
                    print(f"Persona {i+1}: DE FRENTE — no mira anaquel")
                else:
                    print(f"Persona {i+1}: DE LADO — inconclusivo")
        else:
            print("Sin detecciones...")

        time.sleep(1)

except KeyboardInterrupt:
    print("\nDetenido.")
    picam2.stop()