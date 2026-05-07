import time
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from pose_detector import get_poses, Pose, print_pose
from shelf_attention import (
    analyze_pose,
    get_zone_boundaries,
    ZONE_COUNT,
    IMAGE_WIDTH,
)
from config import SHOW_RULER, SHOW_ZONE_OVERLAY, SHOW_POSE_OVERLAY

MODEL_PATH = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

# ---------------------------------------------------------------------------
# Display / overlay config
# ---------------------------------------------------------------------------

WINDOW_NAME = "Tecolotl — Shelf Attention Debug"

# Zone fill colors (B, G, R) — cycles if ZONE_COUNT > len
ZONE_COLORS_BGR = [
    (219, 152,  52),  # azul
    (171, 148,  31),  # verde-azulado
    ( 36, 160, 250),  # amarillo
    (177, 130, 212),  # lila
    (106, 195, 139),  # verde
    ( 78, 186, 244),  # naranja
]

# Status colors (B, G, R)
COLOR_FACING  = ( 86, 199,  29)  # verde       — mirando anaquel
COLOR_AWAY    = ( 36, 112, 237)  # azul        — de frente (no mira)
COLOR_UNKNOWN = (130, 130, 130)  # gris        — inconclusivo

ALPHA_ZONE     = 0.18   # transparencia del relleno de zona (0–1)
LINE_COLOR     = (255, 255, 255)
LINE_THICKNESS = 2

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICK = 1


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_ruler(frame: np.ndarray) -> np.ndarray:
    """
    Dibuja una regla de píxeles en la parte inferior del frame.
    Marcas cada 10px, números cada 50px.
    """
    h, w = frame.shape[:2]
    ruler_y = h - 30

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, ruler_y - 4), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    for x in range(0, w + 1, 10):
        is_major = x % 50 == 0
        color    = (0, 255, 255) if is_major else (0, 160, 160)
        tick     = 14 if is_major else 7
        cv2.line(frame, (x, ruler_y), (x, ruler_y + tick), color, 1)
        if is_major and x > 0:
            label = str(x)
            lw, _ = cv2.getTextSize(label, FONT, 0.35, 1)[0]
            cv2.putText(frame, label, (x - lw // 2, h - 2),
                        FONT, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_zones(frame: np.ndarray, zone_count: int, img_width: int) -> np.ndarray:
    """
    Dibuja rellenos semi-transparentes por zona y líneas divisoras verticales.
    Retorna el frame con el overlay aplicado.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    boundaries    = get_zone_boundaries(zone_count, img_width)
    boundaries_px = [int(b * w / img_width) for b in boundaries]
    zone_edges    = [0] + boundaries_px + [w]

    for i in range(zone_count):
        x1    = zone_edges[i]
        x2    = zone_edges[i + 1]
        color = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]

        # Relleno de zona
        cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)

        # Etiqueta de zona (arriba, centrada)
        label          = f"Zona {i + 1}"
        lw, _          = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)[0]
        lx             = x1 + (x2 - x1 - lw) // 2
        cv2.putText(overlay, label, (lx, 28),
                    FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)

        # Rango en píxeles
        px_start = int(i * img_width / zone_count)
        px_end   = int((i + 1) * img_width / zone_count)
        sub      = f"{px_start}-{px_end}px"
        sw2, _   = cv2.getTextSize(sub, FONT, 0.38, 1)[0]
        cv2.putText(overlay, sub, (x1 + (x2 - x1 - sw2) // 2, 46),
                    FONT, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # Mezclar relleno con el frame original
    frame = cv2.addWeighted(overlay, ALPHA_ZONE, frame, 1 - ALPHA_ZONE, 0)

    # Líneas divisoras (después del blend para que queden nítidas)
    for bx in boundaries_px:
        cv2.line(frame, (bx, 0), (bx, h), LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

    return frame


def draw_person(
    frame: np.ndarray,
    pose: Pose,
    analysis: dict,
    person_idx: int,
    img_width: int,
    img_height: int,
) -> None:
    """
    Dibuja bounding box, hombros, línea central y badge de estado para una persona.
    Modifica el frame en su lugar.
    """
    h, w = frame.shape[:2]
    sx   = w / img_width
    sy   = h / img_height

    facing = analysis["facing_shelf"]
    zone   = analysis["zone"]
    cx     = analysis["shoulder_center_x"]

    if facing is True:
        color  = COLOR_FACING
        status = "mirando anaquel"
    elif facing is False:
        color  = COLOR_AWAY
        status = "de frente"
    else:
        color  = COLOR_UNKNOWN
        status = "inconclusivo"

    zone_str = f"Z{zone}" if zone is not None else "Z?"

    # Bounding box
    box = pose.box  # [x1, y1, x2, y2]
    bx1, by1 = int(box[0] * sx), int(box[1] * sy)
    bx2, by2 = int(box[2] * sx), int(box[3] * sy)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)

    # Keypoints de hombros y línea que los une
    ls = pose.left_shoulder
    rs = pose.right_shoulder
    if ls.is_valid() and rs.is_valid():
        lsp = (int(ls.x * sx), int(ls.y * sy))
        rsp = (int(rs.x * sx), int(rs.y * sy))
        cv2.circle(frame, lsp, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, rsp, 5, color, -1, cv2.LINE_AA)
        cv2.line(frame, lsp, rsp, color, 2, cv2.LINE_AA)

        # Línea vertical desde el centro de hombros
        if cx is not None:
            cx_frame = int(cx * sx)
            cv2.line(frame, (cx_frame, by1), (cx_frame, by2),
                     color, 1, cv2.LINE_AA)

    # Badge de estado encima del bounding box
    badge = f"P{person_idx + 1} {zone_str} | {status}"
    (tw, th), baseline = cv2.getTextSize(badge, FONT, FONT_SCALE, FONT_THICK)
    pad  = 4
    bgy1 = max(0, by1 - th - pad * 2 - baseline)
    cv2.rectangle(frame, (bx1, bgy1), (bx1 + tw + pad * 2, by1), color, -1)
    cv2.putText(frame, badge, (bx1 + pad, by1 - baseline - pad),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

imx500 = IMX500(MODEL_PATH)
picam2 = Picamera2(imx500.camera_num)

config = picam2.create_preview_configuration(
    main={"size": (IMAGE_WIDTH, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

IMG_H = 480

print(f"Cámara iniciada. Zonas: {ZONE_COUNT}  |  Límites: {get_zone_boundaries()}")
print("Presiona 'q' para salir.\n")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

last_debug_time = 0
try:
    while True:
        # Capturar frame y metadata
        request = picam2.capture_request()
        frame_rgb = picam2.capture_array("main")
        metadata  = picam2.capture_metadata()
        request.release()

        # picam2 entrega RGB; OpenCV usa BGR
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Overlay de zonas y regla
        if SHOW_ZONE_OVERLAY:
            frame = draw_zones(frame, ZONE_COUNT, IMAGE_WIDTH)
        if SHOW_RULER:
            frame = draw_ruler(frame)

        
        # Detección de poses
        poses_raw = get_poses(metadata, imx500)
        poses = [p for p in poses_raw if p.score > 0.4]

        if time.monotonic() - last_debug_time > 5:
            last_debug_time = time.monotonic()
            print(f"[DEBUG] raw poses: {len(poses_raw)} | filtered poses: {len(poses)}")

        if poses:
            for i, pose in enumerate(poses):
                analysis = analyze_pose(pose)
                if SHOW_POSE_OVERLAY:
                    draw_person(frame, pose, analysis, i, IMAGE_WIDTH, IMG_H)

                # Salida en consola
                facing = analysis["facing_shelf"]
                zone   = analysis["zone"]
                if facing is True:
                    orient = "DE ESPALDAS — mirando anaquel"
                elif facing is False:
                    orient = "DE FRENTE — no mira anaquel"
                else:
                    orient = "DE LADO — inconclusivo"
                print(f"Persona {i+1}: {orient} | Zona {zone}")
        else:
            cv2.putText(frame, "Sin detecciones...", (12, IMG_H - 12),
                        FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)   # ~20 fps; quitar para velocidad máxima

except KeyboardInterrupt:
    print("\nDetenido.")
finally:
    cv2.destroyAllWindows()
    picam2.stop()