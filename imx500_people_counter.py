#!/usr/bin/env python3
from functools import lru_cache

import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from libcamera import Transform

MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
THRESHOLD = 0.45
MAX_DETECTIONS = 20

last_detections = []


class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = int(category)
        self.conf = float(conf)
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def parse_detections(metadata):
    global last_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections

    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

    _, input_h = imx500.get_input_size()

    if intrinsics.bbox_normalization:
        boxes = boxes / input_h

    if intrinsics.bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]

    labels = get_labels()
    parsed = []

    for box, score, cls in zip(boxes, scores, classes):
        if score < THRESHOLD:
            continue

        cls = int(cls)
        if cls < 0 or cls >= len(labels):
            continue

        if labels[cls] != "person":
            continue

        parsed.append(Detection(box, cls, score, metadata))

    last_detections = parsed[:MAX_DETECTIONS]
    return last_detections


def draw_overlay(request, stream="main"):
    detections = parse_detections(request.get_metadata())

    with MappedArray(request, stream) as m:
        h, w = m.array.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = map(int, det.box)

            cv2.rectangle(
                m.array,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                3,
                cv2.LINE_8
            )


def show_status_window():
    status = np.zeros((140, 520, 3), dtype=np.uint8)

    cv2.putText(
        status,
        "AI MONITORING ACTIVE",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
    )

    cv2.putText(
        status,
        "Smile you are being recorded with AI",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow("AI Status", status)


if __name__ == "__main__":
    print("Loading IMX500 model...")
    imx500 = IMX500(MODEL_PATH)
    intrinsics = imx500.network_intrinsics

    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        raise RuntimeError("El modelo cargado no es de detección de objetos")

    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "XRGB8888"},
	transform = Transform(rotation=0),
        controls={"FrameRate": 30},
        buffer_count=12
    )

    picam2.pre_callback = draw_overlay
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    print("Running IMX500 detection. Ctrl+C para salir.")

    try:
        while True:
            picam2.capture_metadata()
            show_status_window()

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
