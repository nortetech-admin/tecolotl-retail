# config.py — Tecolotl Retail
# ================================

# ---------------------------------------------------------------------------
# Debug visual
# ---------------------------------------------------------------------------

SHOW_RULER: bool = False
SHOW_ZONE_OVERLAY: bool = True
SHOW_POSE_OVERLAY: bool = True

# ---------------------------------------------------------------------------
# Debug consola
# ---------------------------------------------------------------------------

DEBUG_PIPELINE: bool = True

# Imprime debug cada N frames para no saturar la terminal
DEBUG_EVERY_N_FRAMES: int = 15

# Imprime los keypoints importantes de cada persona
DEBUG_KEYPOINTS: bool = True

# Threshold extra en test_poses.py
POSE_SCORE_FILTER: float = 0.2

# Threshold que se manda a pose_detector.py
POSE_DETECTION_THRESHOLD: float = 0.3