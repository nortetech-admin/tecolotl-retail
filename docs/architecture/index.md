# System Architecture

## Overview

Tecolotl Retail runs entirely on a Raspberry Pi 5 with a Sony IMX500 AI camera. The architecture prioritizes local inference and stable visualization before expanding into a full analytics pipeline.

---

## Current Flow

```text
IMX500 Camera
      ↓
On-sensor Inference
      ↓
rpicam-hello
      ↓
Hardware Overlay
      ↓
HDMI Display
```

1. IMX500 captures video and runs AI inference on-sensor
2. `rpicam-hello` manages the camera pipeline
3. Detections are rendered through the hardware overlay
4. Output is displayed on a connected monitor

---

## Status

Analytics, tracking, and reporting modules are planned but not yet implemented.