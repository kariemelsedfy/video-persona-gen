"""Minimal face-crop extraction for preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass
class FaceCropRecord:
    frame_index: int
    source_frame_index: int
    timestamp_sec: float
    crop_path: str
    bbox_xywh: list[int]
    detector_hit: bool
    fallback_source: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _expand_box_to_square(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_width: int,
    frame_height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    side = max(w, h)
    side = int(round(side * (1.0 + max(margin, 0.0) * 2.0)))
    side = max(1, min(side, frame_width, frame_height))
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = min(max(left, 0), frame_width - side)
    top = min(max(top, 0), frame_height - side)
    return left, top, side, side


def _largest_center_square(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    side = min(frame_width, frame_height)
    left = max((frame_width - side) // 2, 0)
    top = max((frame_height - side) // 2, 0)
    return left, top, side, side


def _load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load OpenCV Haar cascade: {cascade_path}")
    return detector


def extract_face_crops(
    input_path: str | Path,
    output_dir: str | Path,
    target_fps: float | None = 25.0,
    face_margin: float = 0.2,
    allow_center_crop_fallback: bool = True,
) -> tuple[list[FaceCropRecord], float]:
    source_path = Path(input_path).expanduser().resolve()
    crop_dir = Path(output_dir).expanduser().resolve()
    crop_dir.mkdir(parents=True, exist_ok=True)

    detector = _load_face_detector()
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for face extraction: {source_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0.0:
            source_fps = target_fps or 25.0

        effective_fps = source_fps
        if target_fps and 0.0 < target_fps < source_fps:
            effective_fps = target_fps

        frame_records: list[FaceCropRecord] = []
        previous_bbox: tuple[int, int, int, int] | None = None
        next_output_time = 0.0
        frame_index = 0
        output_frame_index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            timestamp_sec = frame_index / source_fps if source_fps > 0 else float(output_frame_index)
            should_keep = True
            if effective_fps < source_fps:
                if timestamp_sec + 1e-6 < next_output_time:
                    should_keep = False
                else:
                    next_output_time += 1.0 / effective_fps

            if should_keep:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                min_face = max(24, int(round(min(frame.shape[0], frame.shape[1]) * 0.1)))
                detections = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(min_face, min_face),
                )

                detector_hit = len(detections) > 0
                fallback_source: str | None = None
                if detector_hit:
                    x, y, w, h = max(detections, key=lambda box: int(box[2]) * int(box[3]))
                    bbox = _expand_box_to_square(
                        int(x),
                        int(y),
                        int(w),
                        int(h),
                        frame.shape[1],
                        frame.shape[0],
                        face_margin,
                    )
                elif previous_bbox is not None:
                    bbox = previous_bbox
                    fallback_source = "previous_box"
                elif allow_center_crop_fallback:
                    bbox = _largest_center_square(frame.shape[1], frame.shape[0])
                    fallback_source = "center_crop"
                else:
                    raise RuntimeError(
                        f"Face detection failed on frame {frame_index} of {source_path} and fallback is disabled."
                    )

                previous_bbox = bbox
                left, top, width, height = bbox
                crop = frame[top : top + height, left : left + width]
                crop_path = crop_dir / f"{output_frame_index:06d}.png"
                if not cv2.imwrite(str(crop_path), crop):
                    raise RuntimeError(f"Failed to write crop image: {crop_path}")

                frame_records.append(
                    FaceCropRecord(
                        frame_index=output_frame_index,
                        source_frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        crop_path=str(crop_path),
                        bbox_xywh=[int(left), int(top), int(width), int(height)],
                        detector_hit=detector_hit,
                        fallback_source=fallback_source,
                    )
                )
                output_frame_index += 1

            frame_index += 1
    finally:
        capture.release()

    return frame_records, effective_fps
