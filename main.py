"""
rPPG Heart Rate Monitor — FastAPI Backend
==========================================
Accepts a 10-second video file, isolates facial ROIs using MediaPipe Face Mesh,
extracts the Green channel signal from the forehead, applies a band-pass filter,
and returns BPM with a signal confidence score.

Dependencies:
    pip install fastapi uvicorn python-multipart opencv-python mediapipe numpy scipy
"""

import io
import os
import tempfile
import logging
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt, periodogram
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rppg")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="rPPG Heart Rate API",
    description="Calculates BPM from facial video using remote photoplethysmography.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict to your Flutter app domain in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# MediaPipe initialisation (module-level, reused across requests)
# ---------------------------------------------------------------------------
# Replace:

# With:

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)
FACE_MESH = vision.FaceLandmarker.create_from_options(options)
# ---------------------------------------------------------------------------
# MediaPipe landmark indices for ROI regions
# Forehead: landmarks surrounding the glabella and mid-brow region.
# Left/Right cheek: landmarks on the malar eminence.
# ---------------------------------------------------------------------------
FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                      361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                      176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                      162, 21, 54, 103, 67, 109]

LEFT_CHEEK_LANDMARKS  = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_CHEEK_LANDMARKS = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

# ---------------------------------------------------------------------------
# Signal processing constants
# ---------------------------------------------------------------------------
BPM_LOW_HZ   = 0.75   # 45 BPM
BPM_HIGH_HZ  = 3.00   # 180 BPM
BUTTER_ORDER = 4


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------
class HRResponse(BaseModel):
    bpm: float
    confidence: float          # 0.0 – 1.0
    frames_analysed: int
    fps: float
    message: str


# ---------------------------------------------------------------------------
# Utility: extract polygon mask from landmark indices
# ---------------------------------------------------------------------------
def _get_roi_mask(
    landmarks,
    indices: list,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Return a binary mask for the convex hull of the given landmark indices."""
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)
    hull  = cv2.convexHull(points)
    mask  = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _mean_green_in_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Average the Green channel intensity within the masked region."""
    green  = frame_bgr[:, :, 1].astype(np.float32)   # OpenCV: B=0, G=1, R=2
    pixels = green[mask == 255]
    return float(pixels.mean()) if pixels.size > 0 else 0.0


# ---------------------------------------------------------------------------
# Utility: Butterworth band-pass filter
# ---------------------------------------------------------------------------
def _bandpass_filter(signal: np.ndarray, fps: float) -> np.ndarray:
    nyq  = fps / 2.0
    low  = BPM_LOW_HZ  / nyq
    high = BPM_HIGH_HZ / nyq
    # Clamp to valid range for scipy
    low  = max(low,  1e-4)
    high = min(high, 0.999)
    b, a = butter(BUTTER_ORDER, [low, high], btype="band")  # type: ignore
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Core rPPG pipeline
# ---------------------------------------------------------------------------
def extract_bpm(video_path: str) -> HRResponse:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %.1f fps, %d frames", fps, frame_count)

    green_signal: list[float] = []
    frames_with_face           = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = FACE_MESH.process(rgb)

        if not result.multi_face_landmarks:
            # Skip frames where no face is detected
            continue

        landmarks = result.multi_face_landmarks[0].landmark

        # Build masks for all three ROIs
        forehead_mask    = _get_roi_mask(landmarks, FOREHEAD_LANDMARKS, h, w)
        left_cheek_mask  = _get_roi_mask(landmarks, LEFT_CHEEK_LANDMARKS, h, w)
        right_cheek_mask = _get_roi_mask(landmarks, RIGHT_CHEEK_LANDMARKS, h, w)

        # Combined mask (union)
        combined_mask = cv2.bitwise_or(forehead_mask, left_cheek_mask)
        combined_mask = cv2.bitwise_or(combined_mask, right_cheek_mask)

        mean_g = _mean_green_in_mask(frame, combined_mask)
        green_signal.append(mean_g)
        frames_with_face += 1

    cap.release()

    if len(green_signal) < int(fps * 3):
        raise ValueError(
            f"Insufficient face-detected frames ({len(green_signal)}). "
            "Ensure the face is visible and well-lit throughout the video."
        )

    signal_array = np.array(green_signal, dtype=np.float64)

    # De-trend: subtract a linear fit to remove slow illumination drift
    x        = np.arange(len(signal_array))
    coeffs   = np.polyfit(x, signal_array, 1)
    detrended = signal_array - np.polyval(coeffs, x)

    # Band-pass filter in the physiological BPM range
    filtered = _bandpass_filter(detrended, fps)

    # FFT-based frequency estimation
    freqs, power = periodogram(filtered, fs=fps, window="hann")

    # Restrict to physiological range
    band_mask    = (freqs >= BPM_LOW_HZ) & (freqs <= BPM_HIGH_HZ)
    band_freqs   = freqs[band_mask]
    band_power   = power[band_mask]

    if band_power.sum() == 0 or len(band_freqs) == 0:
        raise ValueError("Could not detect a valid heart rate signal.")

    dominant_freq = band_freqs[np.argmax(band_power)]
    bpm           = dominant_freq * 60.0

    # Signal confidence: ratio of dominant peak power to total band power
    peak_power  = float(np.max(band_power))
    total_power = float(band_power.sum())
    confidence  = min(peak_power / total_power, 1.0) if total_power > 0 else 0.0

    logger.info("BPM=%.1f  Confidence=%.3f  Frames=%d", bpm, confidence, frames_with_face)

    return HRResponse(
        bpm=round(bpm, 1),
        confidence=round(confidence, 3),
        frames_analysed=frames_with_face,
        fps=round(fps, 2),
        message="Analysis successful.",
    )


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------
@app.post("/analyze", response_model=HRResponse)
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a video file (MP4, MOV, AVI) of 8–15 seconds.
    Returns BPM and signal confidence.
    """
    allowed_types = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/mpeg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. "
                   "Please upload MP4, MOV, or AVI.",
        )

    # Write to a named temporary file so OpenCV can open it
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        response = extract_bpm(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during analysis.")
        raise HTTPException(status_code=500, detail="Internal analysis error.")
    finally:
        os.unlink(tmp_path)

    return response


@app.get("/health")
def health():
    return {"status": "ok"}
