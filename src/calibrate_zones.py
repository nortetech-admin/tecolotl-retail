"""
calibrate_zones.py — Tecolotl Retail
======================================
Captura un frame de la cámara y muestra una regla interactiva.
- Mueve el mouse para ver la coordenada X en tiempo real
- Haz clic izquierdo para marcar un límite de zona
- Presiona 'u' para deshacer el último punto
- Presiona 's' para guardar una imagen con los límites marcados
- Presiona 'q' para salir e imprimir los valores para copiar en shelf_attention.py
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

IMAGE_WIDTH  = 640
IMAGE_HEIGHT = 480
WINDOW_NAME  = "Calibrar Zonas — clic para marcar límites"

FONT       = cv2.FONT_HERSHEY_SIMPLEX
LINE_COLOR = (0, 255, 255)    # amarillo-cyan
TEXT_COLOR = (255, 255, 255)
RULE_COLOR = (200, 200, 200)  # regla gris
CLICK_COLOR = (0, 200, 255)   # líneas marcadas

boundaries = []   # lista de x clicks
mouse_x    = 0


def draw_ruler(frame: np.ndarray) -> np.ndarray:
    """Dibuja una regla horizontal con marcas cada 10px y número cada 50px."""
    img = frame.copy()
    h, w = img.shape[:2]

    ruler_y = h - 40   # posición vertical de la regla

    # Fondo semi-transparente para la regla
    overlay = img.copy()
    cv2.rectangle(overlay, (0, ruler_y - 8), (w, h), (30, 30, 30), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # Marcas de la regla
    for x in range(0, w + 1, 10):
        tick_h = 12 if x % 50 == 0 else 6
        cv2.line(img, (x, ruler_y), (x, ruler_y + tick_h), RULE_COLOR, 1)
        if x % 50 == 0 and x > 0:
            label = str(x)
            lw, _ = cv2.getTextSize(label, FONT, 0.32, 1)[0]
            cv2.putText(img, label, (x - lw // 2, ruler_y + 26),
                        FONT, 0.32, RULE_COLOR, 1, cv2.LINE_AA)

    return img


def draw_state(frame: np.ndarray) -> np.ndarray:
    img = draw_ruler(frame)
    h, w = img.shape[:2]

    # Línea del cursor
    cv2.line(img, (mouse_x, 0), (mouse_x, h), (180, 180, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"x={mouse_x}", (mouse_x + 4, 20),
                FONT, 0.45, (180, 180, 0), 1, cv2.LINE_AA)

    # Líneas de límites marcados
    for i, bx in enumerate(boundaries):
        cv2.line(img, (bx, 0), (bx, h - 40), CLICK_COLOR, 2, cv2.LINE_AA)
        cv2.putText(img, f"L{i+1}={bx}", (bx + 4, 44 + i * 18),
                    FONT, 0.42, CLICK_COLOR, 1, cv2.LINE_AA)

    # Rellenos de zona
    edges = [0] + boundaries + [w]
    zone_colors = [
        (219, 152, 52), (171, 148, 31), (36, 160, 250),
        (177, 130, 212), (106, 195, 139), (78, 186, 244),
    ]
    overlay = img.copy()
    for i in range(len(edges) - 1):
        x1, x2 = edges[i], edges[i + 1]
        color   = zone_colors[i % len(zone_colors)]
        cv2.rectangle(overlay, (x1, 0), (x2, h - 40), color, -1)
        label  = f"Zona {i+1}"
        lw, _  = cv2.getTextSize(label, FONT, 0.5, 1)[0]
        cv2.putText(overlay, label, (x1 + (x2 - x1 - lw) // 2, 22),
                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)

    # Instrucciones
    tips = ["clic=marcar limite", "u=deshacer", "s=guardar img", "q=salir"]
    for i, t in enumerate(tips):
        cv2.putText(img, t, (8, h - 40 - 14 - i * 16),
                    FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return img


def on_mouse(event, x, y, flags, param):
    global mouse_x
    mouse_x = x
    if event == cv2.EVENT_LBUTTONDOWN:
        boundaries.append(x)
        boundaries.sort()
        print(f"  Límite marcado: x={x}  →  boundaries hasta ahora: {boundaries}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Iniciando cámara...")

imx500 = IMX500("/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT), "format": "RGB888"}
)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

picam2.configure(config)
picam2.start()

print("Cámara lista. Abre la ventana y haz clic en los separadores del anaquel.")



saved = False
try:
    while True:
        frame_rgb = picam2.capture_array("main")
        frame     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        display   = draw_state(frame)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('u') and boundaries:
            removed = boundaries.pop()
            print(f"  Deshecho: x={removed}")

        elif key == ord('s'):
            fname = "zonas_calibradas.png"
            cv2.imwrite(fname, display)
            print(f"  Imagen guardada: {fname}")
            saved = True

        elif key == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    picam2.stop()

# Resultado final
print("\n" + "="*50)
print("Copia esto en shelf_attention.py:")
print(f"  ZONE_BOUNDARIES_PX = {sorted(boundaries)}")
print(f"  ZONE_COUNT         = {len(boundaries) + 1}")
print("="*50)