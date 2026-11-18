import json

import cv2


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


def update_alarm_states(
    result, video_name, alarm_polygon, now, dwell_seconds, states, alarms, grace_period=0.75  # 0.75 sec is 18 frames in 24fps video, maybe go lower?
):
    """Track people inside the alarm zone and record qualified dwell events,
    using a grace period to handle temporary occlusions and brief disappearances.
    """
    boxes = result.boxes
    changed = False
    triggered = []
    seen_track_ids = set()

    if boxes is not None and boxes.id is not None:
        coordinates = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)
        classes = boxes.cls.cpu().numpy().astype(int)

        for coordinates_for_box, track_id, class_id in zip(
            coordinates, track_ids, classes
        ):
            if class_id != 0:
                continue
            track_id = int(track_id)
            seen_track_ids.add(track_id)

            # Check box overlap with the alarm zone
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
                # Get or create state, initializing or updating last_seen_in_zone
                state = states.setdefault(
                    track_id,
                    {
                        "entered_at": now,
                        "last_seen_in_zone": now,
                        "alarm": None,
                    },
                )
                # Update last seen timestamp because they are actively in the zone right now
                state["last_seen_in_zone"] = now

                # Check if dwell time is met
                if (
                    state["alarm"] is None
                    and now - state["entered_at"] >= dwell_seconds
                ):
                    alarm = {
                        "video": video_name,
                        "track_id": track_id,
                        "person_in_frame_time": format_video_time(
                            state["entered_at"]
                        ),
                        "alarm_triggered_time": format_video_time(
                            state["entered_at"] + dwell_seconds
                        ),
                        "end_time": None,
                    }
                    state["alarm"] = alarm
                    alarms.append(alarm)
                    triggered.append(alarm)
                    changed = True

    # Evaluate all active states to see who has been missing or out of the zone past the grace period
    expired_track_ids = []
    for track_id, state in states.items():
        time_since_last_seen = now - state["last_seen_in_zone"]

        # If they haven't been seen in the alarm zone for longer than the grace period
        if time_since_last_seen > grace_period:
            expired_track_ids.append(track_id)
            if state["alarm"] is not None and state["alarm"]["end_time"] is None:
                state["alarm"]["end_time"] = format_video_time(state["last_seen_in_zone"])
                changed = True

    # Clean up expired states
    for track_id in expired_track_ids:
        states.pop(track_id, None)

    return changed, triggered


def draw_triggered_alarm_boxes(frame, result, states):
    """Draw triggered alarm tracks in red over the default detection boxes."""
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return

    coordinates = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int)
    for box, track_id in zip(coordinates, track_ids):
        state = states.get(int(track_id))
        if state is None or state["alarm"] is None:
            continue

        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
