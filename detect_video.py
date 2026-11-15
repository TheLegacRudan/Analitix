import argparse
import json
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
ALARM_LOG = ROOT / "alarms.json"


def expand_zone(zone):
    """Expand a normalized rectangular zone on each side, clamped to the frame."""
    xs = [point[0] for point in zone]
    ys = [point[1] for point in zone]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    diffX = right - left
    diffY = bottom - top
    margin_x = (0.15/(diffX+0.6)) # only one calculation per zone, not expensive
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


def format_video_time(seconds):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{milliseconds:03}"


def load_alarms(path):
    if not path.exists():
        return []
    alarms = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(alarms, list) or not all(
        isinstance(alarm, dict) for alarm in alarms
    ):
        raise ValueError(f"Alarm log must contain a JSON list: {path}")
    return alarms


def save_alarms(path, alarms):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(alarms, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _segments_intersect(a, b, c, d):
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    ab_c, ab_d = cross(a, b, c), cross(a, b, d)
    cd_a, cd_b = cross(c, d, a), cross(c, d, b)
    return (
        (ab_c == 0 and on_segment(a, c, b))
        or (ab_d == 0 and on_segment(a, d, b))
        or (cd_a == 0 and on_segment(c, a, d))
        or (cd_b == 0 and on_segment(c, b, d))
        or ((ab_c > 0) != (ab_d > 0) and (cd_a > 0) != (cd_b > 0))
    )


def update_alarm_states(result, video_name, alarm_polygon, now, states, alarms):
    """Track people inside the alarm zone and record qualified dwell events."""
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return False, []

    coordinates = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int)
    classes = boxes.cls.cpu().numpy().astype(int)
    changed = False
    triggered = []

    for coordinates_for_box, track_id, class_id in zip(
        coordinates, track_ids, classes
    ):
        if class_id != 0:
            continue
        track_id = int(track_id)

        # checks if anything is inside alarm zone not just center of the box
        x1, y1, x2, y2 = coordinates_for_box
        box_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        alarm_points = alarm_polygon.reshape(-1, 2)
        inside_alarm_zone = any(
            cv2.pointPolygonTest(alarm_polygon, corner, False) >= 0
            for corner in box_corners
        ) or any(
            x1 <= point[0] <= x2 and y1 <= point[1] <= y2
            for point in alarm_points
        ) or any(
            _segments_intersect(
                box_corners[i],
                box_corners[(i + 1) % 4],
                alarm_points[j],
                alarm_points[(j + 1) % len(alarm_points)],
            )
            for i in range(4)
            for j in range(len(alarm_points))
        )

        if inside_alarm_zone:
            state = states.setdefault(
                track_id,
                {"entered_at": now, "alarm": None},
            )
            if (
                state["alarm"] is None
                and now - state["entered_at"] >= VIDEO_CONFIG["dwell_seconds"]
            ):
                alarm = {
                    "video": video_name,
                    "track_id": track_id,
                    "start_time": format_video_time(
                        state["entered_at"] + VIDEO_CONFIG["dwell_seconds"]
                    ),
                    "end_time": None,
                }
                state["alarm"] = alarm
                alarms.append(alarm)
                triggered.append(alarm)
                changed = True
        else:
            state = states.pop(track_id, None)
            if state is not None and state["alarm"] is not None:
                state["alarm"]["end_time"] = format_video_time(now)
                changed = True

    return changed, triggered


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

    alarms = load_alarms(ALARM_LOG)
    save_alarms(ALARM_LOG, alarms)
    alarm_states = {}
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
            raise RuntimeError(f"Could not read video dimensions: {video_path}")
        
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

            result = model.track(persist=True, **predict_options)[0]
            alarms_changed, triggered = update_alarm_states(
                result,
                args.video,
                zone_polygon,
                frames_processed / fps,
                alarm_states,
                alarms,
            )
            for alarm in triggered:
                print(
                    f"ALARM: {args.video}, person track {alarm['track_id']}, "
                    f"started at {alarm['start_time']}"
                )
            if alarms_changed:
                save_alarms(ALARM_LOG, alarms)

            # zaokruzi samo one ljude kojima je centar u detection zoni
            zone_result = keep_detections_in_zone(result, detection_polygon)
            annotated_frame = zone_result.plot()

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

            #only text, not necessary
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
