# Display Pipeline

## Overview

Tecolotl Retail supports an optional real-time display pipeline for operational monitoring and debugging.

The display pipeline renders detections directly through the Sony IMX500 hardware overlay system using `rpicam-hello`.

This approach provides stable on-screen rendering while keeping analytics execution isolated inside the Python pipeline.

---

## Architecture

```text
Sony IMX500 AI Camera
            ↓
      rpicam-hello
            ↓
 Hardware Overlay Rendering
            ↓
      HDMI Display Output
```

The display pipeline is independent from:

- analytics
- event generation
- tracking
- storage systems

Visualization is treated strictly as an observability layer.

## Rationale

Early prototypes used OpenCV-based overlays rendered in Python.

This approach introduced visible overlay instability caused by:

- frame-to-frame inference variation
- rendering latency
- asynchronous drawing operations

**Before — Python OpenCV overlay (unstable):**

![Before display 1](../../assets/images/before_display.png)
![Before display 2](../../assets/images/before_display2.png)
![Before display 3](../../assets/images/before_display3.png)

**After — IMX500 hardware overlay (stable):**

![After display](../../assets/images/after_display.png)

The IMX500 hardware overlay pipeline provides significantly more stable rendering because detections are rendered internally before the display stage.

## Launch Script

The display pipeline is started using:

```bash
./scripts/start_display.sh
```

Current implementation:

```bash
#!/bin/bash

rpicam-hello -t 0s \
  --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json \
  --viewfinder-width 1920 \
  --viewfinder-height 1080 \
  --width 1920 \
  --height 1080 \
  --framerate 30 \
  --fullscreen
```

---

## Current Configuration

| Parameter | Value |
|---|---|
| Resolution | 1920×1080 |
| Framerate | 30 FPS |
| Inference Model | MobileNet SSD |
| Display Mode | Fullscreen |

---

## Operational Usage

The display pipeline is primarily intended for:

- deployment validation
- live monitoring
- camera alignment
- debugging
- demonstration environments

It is not required for analytics execution.

The Python analytics pipeline can operate independently without a connected display.