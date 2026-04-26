# Getting Started

## Overview

Tecolotl Retail is an edge-based computer vision analytics system designed for retail environments using a Raspberry Pi 5 and Sony IMX500 AI camera.

The platform performs:

- real-time person detection
- customer movement analysis
- shelf attention estimation
- retail analytics generation

The system is optimized for embedded deployment and hardware-accelerated inference.

## Architecture Summary

Tecolotl Retail separates visualization from analytics.

- IMX500 hardware overlays handle real-time rendering
- Python handles analytics, tracking, and event processing

```text
Sony IMX500 AI Camera
            ↓
     rpicam-hello
            ↓
      Metadata Stream
            ↓
 Python Analytics Pipeline
```

## Hardware Stack

| Component | Purpose |
|---|---|
| Raspberry Pi 5 | Edge compute platform |
| Sony IMX500 AI Camera | On-sensor AI inference |
| Raspberry Pi OS | Operating system |

## Software Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3 |
| Vision Utilities | OpenCV |
| Camera Pipeline | rpicam-hello |
| Documentation | VitePress |

## Repository Structure

```
Retail/
├── docs/
├── src/
├── scripts/
├── config/
├── data/
├── models/
└── old/
```

## Design Principles

- Edge-first architecture
- Hardware-accelerated inference
- Modular analytics pipeline
- Deterministic runtime behavior
- Separation of visualization and analytics

## Current Status

Current system capabilities include:

- real-time person detection
- IMX500 hardware overlays
- Raspberry Pi deployment
- analytics experimentation

## Optional Display Pipeline

Optional real-time monitoring can be enabled through the IMX500 hardware overlay pipeline.

See:

- [hardware/display-pipeline.md](../hardware/display-pipeline.md)

## Next Sections

- Architecture
- Hardware
- Vision
- Analytics
- Infrastructure
- Experiments