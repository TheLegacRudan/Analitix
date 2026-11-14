import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
VIDEO_CONFIG = {
    "dwell_seconds": 3,
    "zones": {
        "skladiste-ljudi": [
            [0.30, 0.30],
            [0.85, 0.30],
            [0.85, 0.90],
            [0.30, 0.90],
        ],
        "skladiste-osoba": [
            [0.16, 0.55],
            [0.24, 0.55],
            [0.24, 1.00],
            [0.16, 1.00],
        ],
        "stanica-noc": [
            [0.15, 0.55],
            [0.80, 0.55],
            [0.80, 0.88],
            [0.15, 0.88],
        ],
        "bus-ljudi": [
            [0.42, 0.50],
            [0.98, 0.50],
            [0.98, 0.70],
            [0.42, 0.70],
        ],
    },
}


def scale_zone(zone, width, height):
    """Convert normalized zone coordinates to pixel coordinates."""
    points = np.asarray(
        [[x * (width - 1), y * (height - 1)] for x, y in zone],
        dtype=np.int32,
    )
    return points


def keep_detections_in_zone(result, polygon):
    """Keep detections whose bounding-box center falls inside the zone."""
    if result.boxes is None:
        return result

    box_coordinates = result.boxes.xyxy
    if hasattr(box_coordinates, "cpu"):
        boxes = box_coordinates.cpu().numpy()
    else:
        boxes = np.asarray(box_coordinates)
    indices = []
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if cv2.pointPolygonTest(polygon, center, False) >= 0:
            indices.append(index)
    return result[indices]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on a video and save an annotated copy."
    )
    parser.add_argument(
        "--video",
        choices=VIDEO_CONFIG["zones"],
        default="bus-ljudi",
        help="Input video name (default: bus-ljudi).",
    )
    # provjerit optimalno
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum detection confidence, from 0 to 1 (default: 0.3).",
    )
    # using default
    parser.add_argument(
        "--classes",
        type=int,
        default=0,
        nargs="+",
        help="Optional class IDs to detect, e.g. --classes 0 for people only.",
    )
    # nezz cemu ovo sluzi
    parser.add_argument(
        "--device",
        help="Optional inference device, e.g. cpu, 0, or 0,1.",
    )
    # nezz cemu ovo sluzi
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display detections while processing; press q to stop early.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = ROOT / "videos" / (args.video + ".mp4")
    # video_path = args.video.expanduser().resolve()
    output_path = ROOT / "output" / \
        (args.video + "-output" + time.strftime("-%d%m-%H%M") + ".mp4")
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")

    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if video_path == output_path:
        raise ValueError("The output path must not overwrite the input video.")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1.")

    model = YOLO("yolo11n.pt")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    writer = None
    frames_processed = 0
    try:
        # provjeri za onaj video koji ima 50 fpsa
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"Could not read video dimensions: {video_path}")
        zone_polygon = scale_zone(
            VIDEO_CONFIG["zones"][args.video], width, height)

        # output_path.parent.mkdir(parents=True, exist_ok=True) - ne triba jer folder svakako postoji
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

        while True:
            success, frame = capture.read()
            if not success:
                break

            predict_options = {
                "source": frame,
                "conf": args.confidence,
                "classes": args.classes,
                "verbose": False,
            }

            # if args.device:
            #     predict_options["device"] = args.device

            result = model.predict(**predict_options)[0]
            zone_result = keep_detections_in_zone(result, zone_polygon)
            annotated_frame = zone_result.plot()
            cv2.polylines(
                annotated_frame,
                [zone_polygon],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )
            writer.write(annotated_frame)
            frames_processed += 1

            if args.show:
                cv2.imshow("YOLO video detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frames_processed == 0:
        raise RuntimeError(
            f"No frames were read from input video: {video_path}")

    print(f"Processed {frames_processed} frames.")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    main()
