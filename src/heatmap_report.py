"""
heatmap_report.py - Tecolotl Retail
===================================
Captures pose detections in front of 4 shelf zones and creates a heatmap
report with attention time per shelf.

Metric:
    attention_seconds per shelf = sum of seconds where a detected person is
    facing the shelf in that zone. If two people face the same shelf for one
    second, that shelf receives two person-seconds.

Usage:
    python src/heatmap_report.py --duration 300
    python src/heatmap_report.py --duration 60 --preview
"""

from __future__ import annotations
#Proporciona una forma más fácil de generar interacción con la linea de comandos.
import argparse
import csv
import time

#Libreria para añadir métodos a clases definidas por el usuario
from dataclasses import dataclass, field
#Librería provee clases para manipular fechas y tiempo
from datetime import datetime

#Librería para manipular rutas en el proyecto en cuestión
from pathlib import Path


from typing import Iterable

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

from config import POSE_DETECTION_THRESHOLD, POSE_SCORE_FILTER
from pose_detector import Pose, get_poses
from shelf_attention import IMAGE_WIDTH, ZONE_COUNT, analyze_pose, get_zone_boundaries


MODEL_PATH = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
IMG_H = 480
WINDOW_NAME = "Tecolotl Heatmap Report"
DEFAULT_REPORT_DIR = Path("reports")

FONT = cv2.FONT_HERSHEY_SIMPLEX

ZONE_COLORS_BGR = [
    (219, 152, 52),
    (171, 148, 31),
    (36, 160, 250),
    (177, 130, 212),
]


@dataclass
class HeatmapEvent:
    timestamp: str
    elapsed_seconds: float
    person_index: int
    zone: int | None
    facing_shelf: bool | None
    pose_score: float
    shoulder_center_x: float | None

#Esta clase nos permite definir qué variables y métodos tiene heatmapacumulator, en resumen
#
@dataclass
class HeatmapAccumulator:
    zone_count: int = ZONE_COUNT
    attention_seconds: dict[int, float] = field(init=False)
    facing_detections: dict[int, int] = field(init=False)
    all_detections: int = 0
    inconclusive_detections: int = 0
    events: list[HeatmapEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.attention_seconds = {zone: 0.0 for zone in range(1, self.zone_count + 1)}
        self.facing_detections = {zone: 0 for zone in range(1, self.zone_count + 1)}

    def add_pose(
        self,
        pose: Pose,
        person_index: int,
        elapsed_seconds: float,
        delta_seconds: float,
    ) -> None:
        #We use the analyze_pose from shelf_attention.py, it returns an array determining
        #If the person is facing a shelf and which shelf it is facing. 
        
        analysis = analyze_pose(pose)
        facing = analysis["facing_shelf"]
        zone = analysis["zone"]
        center_x = analysis["shoulder_center_x"]

        self.all_detections += 1
        if facing is None:
            self.inconclusive_detections += 1

        if facing is True and zone is not None:
            self.attention_seconds[zone] += delta_seconds
            self.facing_detections[zone] += 1

        self.events.append(
            HeatmapEvent(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                elapsed_seconds=elapsed_seconds,
                person_index=person_index,
                zone=zone,
                facing_shelf=facing,
                pose_score=pose.score,
                shoulder_center_x=center_x,
            )
        )

    @property
    def total_attention_seconds(self) -> float:
        return sum(self.attention_seconds.values())

    def rows(self) -> list[dict[str, float | int]]:
        total = self.total_attention_seconds
        rows = []
        for zone in range(1, self.zone_count + 1):
            seconds = self.attention_seconds[zone]
            percent = (seconds / total * 100.0) if total > 0 else 0.0
            rows.append(
                {
                    "shelf": zone,
                    "attention_seconds": round(seconds, 3),
                    "attention_percent": round(percent, 2),
                    "facing_detections": self.facing_detections[zone],
                }
            )
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 4-shelf person-attention heatmap report."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Capture duration in seconds. Default: 300.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory where CSV and PNG reports are written.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live camera preview with shelf zones while capturing.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=POSE_SCORE_FILTER,
        help="Minimum pose score to include in the report.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=POSE_DETECTION_THRESHOLD,
        help="Threshold passed to the IMX500 pose postprocess step.",
    )
    return parser.parse_args()


def ensure_report_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_events_csv(path: Path, events: Iterable[HeatmapEvent]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "timestamp",
                "elapsed_seconds",
                "person_index",
                "zone",
                "facing_shelf",
                "pose_score",
                "shoulder_center_x",
            ]
        )
        for event in events:
            writer.writerow(
                [
                    event.timestamp,
                    f"{event.elapsed_seconds:.3f}",
                    event.person_index,
                    event.zone if event.zone is not None else "",
                    event.facing_shelf if event.facing_shelf is not None else "",
                    f"{event.pose_score:.4f}",
                    (
                        f"{event.shoulder_center_x:.2f}"
                        if event.shoulder_center_x is not None
                        else ""
                    ),
                ]
            )


def write_summary_csv(path: Path, accumulator: HeatmapAccumulator) -> None:
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "shelf",
            "attention_seconds",
            "attention_percent",
            "facing_detections",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accumulator.rows())


def draw_zones_preview(frame: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    h, w = frame.shape[:2]
    boundaries = [int(b * w / IMAGE_WIDTH) for b in get_zone_boundaries()]
    edges = [0] + boundaries + [w]

    for idx in range(ZONE_COUNT):
        x1 = edges[idx]
        x2 = edges[idx + 1]
        color = ZONE_COLORS_BGR[idx % len(ZONE_COLORS_BGR)]
        cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
        cv2.putText(
            overlay,
            f"Anaquel {idx + 1}",
            (x1 + 12, 30),
            FONT,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    for boundary in boundaries:
        cv2.line(frame, (boundary, 0), (boundary, h), (255, 255, 255), 2)
    return frame


def draw_live_status(
    frame: np.ndarray,
    poses: list[Pose],
    accumulator: HeatmapAccumulator,
    elapsed_seconds: float,
    duration_seconds: float,
) -> None:
    cv2.putText(
        frame,
        f"{elapsed_seconds:.1f}/{duration_seconds:.1f}s",
        (12, IMG_H - 42),
        FONT,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"poses: {len(poses)}  attention: {accumulator.total_attention_seconds:.1f}s",
        (12, IMG_H - 16),
        FONT,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def create_report_image(
    path: Path,
    accumulator: HeatmapAccumulator,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    width = 1200
    height = 760
    margin = 64
    chart_x = margin
    chart_y = 270
    chart_w = width - margin * 2
    chart_h = 150

    image = np.full((height, width, 3), (246, 247, 249), dtype=np.uint8)
    total = accumulator.total_attention_seconds
    max_seconds = max(accumulator.attention_seconds.values(), default=0.0)

    cv2.putText(
        image,
        "Heatmap de atencion por anaquel",
        (margin, 82),
        FONT,
        1.1,
        (34, 34, 34),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Inicio: {started_at.isoformat(timespec='seconds')}   Fin: {ended_at.isoformat(timespec='seconds')}",
        (margin, 128),
        FONT,
        0.55,
        (78, 78, 78),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Total atencion: {total:.1f} person-seconds   Detecciones: {accumulator.all_detections}",
        (margin, 164),
        FONT,
        0.62,
        (78, 78, 78),
        1,
        cv2.LINE_AA,
    )

    shelf_w = chart_w // accumulator.zone_count
    for idx, row in enumerate(accumulator.rows()):
        zone = int(row["shelf"])
        seconds = float(row["attention_seconds"])
        percent = float(row["attention_percent"])
        x1 = chart_x + idx * shelf_w
        x2 = chart_x + (idx + 1) * shelf_w - 10
        intensity = seconds / max_seconds if max_seconds > 0 else 0.0
        base_color = np.array(ZONE_COLORS_BGR[idx % len(ZONE_COLORS_BGR)], dtype=float)
        pale = np.array([230, 230, 230], dtype=float)
        color = tuple(int(v) for v in (pale * (1.0 - intensity) + base_color * intensity))

        cv2.rectangle(image, (x1, chart_y), (x2, chart_y + chart_h), color, -1)
        cv2.rectangle(image, (x1, chart_y), (x2, chart_y + chart_h), (210, 210, 210), 1)

        label = f"Anaquel {zone}"
        cv2.putText(image, label, (x1 + 18, chart_y + 42), FONT, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{seconds:.1f}s",
            (x1 + 18, chart_y + 86),
            FONT,
            0.9,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{percent:.1f}%",
            (x1 + 18, chart_y + 122),
            FONT,
            0.68,
            (65, 65, 65),
            1,
            cv2.LINE_AA,
        )

    bar_top = 500
    bar_h = 32
    max_bar_w = width - margin * 2 - 210
    for idx, row in enumerate(accumulator.rows()):
        zone = int(row["shelf"])
        seconds = float(row["attention_seconds"])
        bar_w = int((seconds / max_seconds) * max_bar_w) if max_seconds > 0 else 0
        y = bar_top + idx * 54
        color = ZONE_COLORS_BGR[idx % len(ZONE_COLORS_BGR)]

        cv2.putText(image, f"Anaquel {zone}", (margin, y + 23), FONT, 0.58, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.rectangle(image, (margin + 150, y), (margin + 150 + max_bar_w, y + bar_h), (226, 226, 226), -1)
        cv2.rectangle(image, (margin + 150, y), (margin + 150 + bar_w, y + bar_h), color, -1)
        cv2.putText(
            image,
            f"{seconds:.1f}s",
            (margin + 165 + max_bar_w, y + 23),
            FONT,
            0.56,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(path), image)


def setup_camera() -> tuple[IMX500, Picamera2]:
    imx500 = IMX500(MODEL_PATH)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    if intrinsics.inference_rate is None:
        intrinsics.inference_rate = 10
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (IMAGE_WIDTH, IMG_H), "format": "RGB888"},
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12,
    )
    imx500.show_network_fw_progress_bar()
    picam2.configure(config)
    picam2.start()
    imx500.set_auto_aspect_ratio()
    return imx500, picam2


def run_capture(args: argparse.Namespace) -> tuple[HeatmapAccumulator, datetime, datetime]:
    imx500, picam2 = setup_camera()
    accumulator = HeatmapAccumulator()
    started_at = datetime.now()
    start = time.monotonic()
    last_tick = start

    if args.preview:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print(f"Capturando heatmap por {args.duration:.1f}s. Ctrl+C para terminar antes.")
    print(f"Zonas: {ZONE_COUNT} | Limites: {get_zone_boundaries()}")

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= args.duration:
                break

            delta = max(0.0, now - last_tick)
            last_tick = now

            request = picam2.capture_request()
            frame_rgb = request.make_array("main")
            metadata = request.get_metadata()
            request.release()

            poses = get_poses(
                metadata,
                imx500,
                detection_threshold=args.detection_threshold,
            )
            poses = [pose for pose in poses if pose.score >= args.min_score]

            for person_index, pose in enumerate(poses, start=1):
                accumulator.add_pose(
                    pose=pose,
                    person_index=person_index,
                    elapsed_seconds=elapsed,
                    delta_seconds=delta,
                )

            if args.preview:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame = draw_zones_preview(frame)
                draw_live_status(frame, poses, accumulator, elapsed, args.duration)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nCaptura detenida por usuario.")
    finally:
        ended_at = datetime.now()
        if args.preview:
            cv2.destroyAllWindows()
        picam2.stop()

    return accumulator, started_at, ended_at


def main() -> None:
    args = parse_args()
    report_dir = ensure_report_dir(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    accumulator, started_at, ended_at = run_capture(args)

    events_path = report_dir / f"heatmap_events_{stamp}.csv"
    summary_path = report_dir / f"heatmap_summary_{stamp}.csv"
    image_path = report_dir / f"heatmap_report_{stamp}.png"

    write_events_csv(events_path, accumulator.events)
    write_summary_csv(summary_path, accumulator)
    create_report_image(image_path, accumulator, started_at, ended_at)

    print("\nReporte generado:")
    print(f"  Eventos: {events_path}")
    print(f"  Resumen: {summary_path}")
    print(f"  Imagen:  {image_path}")
    print("\nResumen por anaquel:")
    for row in accumulator.rows():
        print(
            f"  Anaquel {row['shelf']}: "
            f"{row['attention_seconds']}s ({row['attention_percent']}%)"
        )


if __name__ == "__main__":
    main()
