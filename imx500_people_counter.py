#!/usr/bin/env python3
from functools import lru_cache

import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from libcamera import Transform

MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
THRESHOLD = 0.45


class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.conf = conf
        self.category = category
        obj = imx500.convert_inference_coords(coords, metadata, picam2)
        self.box = (obj.x, obj.y, obj.width, obj.height)


@lru_cache
def get_labels():
    return intrinsics.labels


def draw_overlay(request):
    outputs = imx500.get_outputs(request.get_metadata(), add_batch=True)
    if outputs is None:
        return

    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    labels = get_labels()

    with MappedArray(request, "main") as m:
        for box, score, cls in zip(boxes, scores, classes):
            if score < THRESHOLD:
                continue
            if labels[int(cls)] != "person":
                continue

            det = Detection(box, cls, score, request.get_metadata())
            x, y, w, h = det.box

            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 3, cv2.LINE_8)


if __name__ == "__main__":
    imx500 = IMX500(MODEL_PATH)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "object detection"
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "XRGB8888"},
        transform=Transform(rotation=0),
        controls={"FrameRate": 30},
        buffer_count=12
    )

    picam2.pre_callback = draw_overlay
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    print("Detectando personas... Presiona Ctrl+C para salir.")

    try:
        while True:
            picam2.capture_metadata()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()