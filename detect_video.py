import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from alarm_handling import (
    draw_triggered_alarm_boxes,
    load_alarms,
    save_alarms,
    update_alarm_states,
)


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
TRACKER_CONFIGS = {
    "crowd": "crowd.yaml",
    "night": "night.yaml",
    "default": "default.yaml",
}
PROFILE_SAMPLE_COUNT = 5
PROFILE_SAMPLE_WINDOW_SECONDS = 4
CROWD_PERSON_THRESHOLD = 5
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 80
ALARM_LOG = ROOT / "alarms.json"


def expand_zone(zone):
    """Expand a normalized rectangular zone on each side, clamped to the frame."""
    xs = [point[0] for point in zone]
    ys = [point[1] for point in zone]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    diffX = right - left
    diffY = bottom - top
    margin_x = (0.15/(diffX+0.6))  # only one calculation per zone, not expensive
    margin_y = (0.15/(diffY+0.6))

    left = max(0.0, left - margin_x)
    right = min(1.0, right + margin_x)
    top = max(0.0, top - margin_y)
    bottom = min(1.0, bottom + margin_y)
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


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


def select_tracker_profile(brightness_samples, person_counts):
    if not brightness_samples:
        raise ValueError("At least one video frame is required to select a tracker.")
    if np.median(brightness_samples) < LOW_LIGHT_BRIGHTNESS_THRESHOLD:
        return "night"
    if max(person_counts, default=0) >= CROWD_PERSON_THRESHOLD:
        return "crowd"
    return "default"


def sample_tracker_profile(model, capture, confidence, fps):
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    last_sample_frame = max(0, round(fps * PROFILE_SAMPLE_WINDOW_SECONDS))
    if frame_count > 0:
        last_sample_frame = min(last_sample_frame, frame_count - 1)

    sample_frames = sorted(
        {
            round(
                last_sample_frame * index / (PROFILE_SAMPLE_COUNT - 1)
            )
            for index in range(PROFILE_SAMPLE_COUNT)
        }
    )
    brightness_samples = []
    person_counts = []
    for frame_index in sample_frames:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise RuntimeError(f"Could not seek to frame {frame_index} for profile detection.")
        success, frame = capture.read()
        if not success:
            continue

        brightness_samples.append(
            float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        )
        detections = model.predict(
            source=frame,
            conf=confidence,
            classes=[0],
            verbose=False,
        )[0]
        if detections.boxes is None:
            person_counts.append(0)
        else:
            person_counts.append(len(detections.boxes))

    if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
        raise RuntimeError("Could not rewind video after tracker profile detection.")
    return select_tracker_profile(brightness_samples, person_counts)


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
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum detection confidence, from 0 to 1 (default: 0.3).",
    )
    # just people
    parser.add_argument(
        "--classes",
        type=int,
        default=0,
        nargs="+",
        help="Optional class IDs to detect, e.g. --classes 0 for people only.",
    )
    # show video
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display detections while processing; press q to stop early.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = ROOT / "videos" / (args.video + ".mp4")
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

    alarms = load_alarms(ALARM_LOG)
    save_alarms(ALARM_LOG, alarms)
    alarm_states = {}
    writer = None
    frames_processed = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Could not read video dimensions: {video_path}")

        tracker_profile = sample_tracker_profile(
            model, capture, args.confidence, fps
        )
        tracker_config = ROOT / "trackers" / TRACKER_CONFIGS[tracker_profile]
        if not tracker_config.is_file():
            raise FileNotFoundError(
                f"Tracker configuration not found: {tracker_config}"
            )
        print(
            f"Selected tracker profile: {tracker_profile} "
            f"({tracker_config.name})"
        )

        zone = VIDEO_CONFIG["zones"][args.video]
        zone_polygon = scale_zone(zone, width, height)
        detection_polygon = scale_zone(expand_zone(zone), width, height)

        output_path.parent.mkdir(parents=True, exist_ok=True)
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

            result = model.track(
                persist=True, tracker=str(tracker_config), **predict_options
            )[0]
            # result = model.track(persist=True, **predict_options)[0]

            alarms_changed, triggered = update_alarm_states(
                result,
                args.video,
                zone_polygon,
                frames_processed / fps,
                VIDEO_CONFIG["dwell_seconds"],
                alarm_states,
                alarms,
            )
            for alarm in triggered:
                print(
                    f"ALARM: {args.video}, person track {alarm['track_id']}, "
                    f"started at {alarm['alarm_triggered_time']}"
                )
            if alarms_changed:
                save_alarms(ALARM_LOG, alarms)

            # zaokruzi samo one ljude kojima je centar u detection zoni
            zone_result = keep_detections_in_zone(result, detection_polygon)
            annotated_frame = zone_result.plot(conf=False, font_size=2, line_width=1)
            draw_triggered_alarm_boxes(annotated_frame, zone_result, alarm_states)

            # iscrtava detection zone, nije potrebno
            # cv2.polylines(
            #     annotated_frame,
            #     [detection_polygon],
            #     isClosed=True,
            #     color=(255, 0, 0),
            #     thickness=2,
            # )
            # #only text, not necessary
            # x, y = detection_polygon.reshape(-1, 2)[0]
            # cv2.putText(
            #     annotated_frame,
            #     "DETECTION ZONE",
            #     (int(x), int(y) - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (255, 0, 0),
            #     1,
            #     cv2.LINE_AA,
            # )

            cv2.polylines(
                annotated_frame,
                [zone_polygon],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )

            # only text, not necessary
            x, y = zone_polygon.reshape(-1, 2)[0]
            cv2.putText(
                annotated_frame,
                "ALARM ZONE",
                (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
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
    print(f"Alarm log saved to: {ALARM_LOG}")


if __name__ == "__main__":
    main()
