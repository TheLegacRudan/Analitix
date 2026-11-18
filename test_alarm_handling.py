import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from alarm_handling import (
    format_video_time,
    load_alarms,
    save_alarms,
    update_alarm_states,
)


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def make_result(boxes=(), track_ids=(), classes=()):
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=FakeTensor(boxes),
            id=FakeTensor(track_ids),
            cls=FakeTensor(classes),
        )
    )


class FormatVideoTimeTests(unittest.TestCase):
    def test_formats_milliseconds_and_carries_into_next_second(self):
        self.assertEqual(format_video_time(1.234), "00:00:01.234")
        self.assertEqual(format_video_time(59.9996), "00:01:00.000")
        self.assertEqual(format_video_time(3661.2), "01:01:01.200")


class AlarmPersistenceTests(unittest.TestCase):
    def test_loads_missing_file_as_empty_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alarms.json"

            self.assertEqual(load_alarms(path), [])

    def test_saves_and_loads_alarms_without_leaving_temporary_file(self):
        alarms = [{"track_id": 7, "end_time": None}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alarms.json"

            save_alarms(path, alarms)

            self.assertEqual(load_alarms(path), alarms)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), alarms)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_rejects_non_list_alarm_logs_and_non_object_entries(self):
        invalid_logs = ('{"track_id": 7}', '[{"track_id": 7}, null]')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alarms.json"
            for content in invalid_logs:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "JSON list"):
                        load_alarms(path)


class UpdateAlarmStatesTests(unittest.TestCase):
    def setUp(self):
        self.polygon = np.array(
            [[10, 10], [20, 10], [20, 20], [10, 20]],
            dtype=np.int32,
        )

    def test_triggers_once_at_dwell_threshold_and_closes_when_person_leaves(self):
        states = {}
        alarms = []
        inside_person = make_result([[12.0, 12.0, 16.0, 16.0]], [42], [0])

        self.assertEqual(
            update_alarm_states(
                inside_person, "camera", self.polygon, 5, 3, states, alarms
            ),
            (False, []),
        )
        self.assertEqual(
            update_alarm_states(
                inside_person, "camera", self.polygon, 7.999, 3, states, alarms
            ),
            (False, []),
        )

        changed, triggered = update_alarm_states(
            inside_person, "camera", self.polygon, 8, 3, states, alarms
        )
        self.assertTrue(changed)
        self.assertEqual(triggered, alarms)
        self.assertEqual(
            alarms[0],
            {
                "video": "camera",
                "track_id": 42,
                "person_in_frame_time": "00:00:05.000",
                "alarm_triggered_time": "00:00:08.000",
                "end_time": None,
            },
        )
        self.assertEqual(
            update_alarm_states(
                inside_person, "camera", self.polygon, 9, 3, states, alarms
            ),
            (False, []),
        )

        outside_person = make_result([[30.0, 30.0, 40.0, 40.0]], [42], [0])
        changed, triggered = update_alarm_states(
            outside_person, "camera", self.polygon, 10, 3, states, alarms
        )
        self.assertTrue(changed)
        self.assertEqual(triggered, [])
        self.assertEqual(alarms[0]["end_time"], "00:00:10.000")
        self.assertEqual(states, {})

    def test_box_overlap_counts_even_when_its_center_is_outside(self):
        states = {}
        alarms = []
        overlaps_zone = make_result([[18.0, 12.0, 24.0, 18.0]], [3], [0])

        update_alarm_states(
            overlaps_zone, "camera", self.polygon, 2, 3, states, alarms
        )

        self.assertEqual(states[3]["entered_at"], 2)

    def test_disappearance_closes_a_triggered_alarm(self):
        states = {}
        alarms = []
        inside_person = make_result([[12.0, 12.0, 16.0, 16.0]], [5], [0])
        update_alarm_states(
            inside_person, "camera", self.polygon, 1, 1, states, alarms
        )
        update_alarm_states(
            inside_person, "camera", self.polygon, 2, 1, states, alarms
        )

        changed, triggered = update_alarm_states(
            make_result(), "camera", self.polygon, 4, 1, states, alarms
        )

        self.assertTrue(changed)
        self.assertEqual(triggered, [])
        self.assertEqual(alarms[0]["end_time"], "00:00:04.000")
        self.assertEqual(states, {})

    def test_ignores_non_people_and_people_outside_the_zone(self):
        states = {}
        alarms = []
        result = make_result(
            [[12.0, 12.0, 16.0, 16.0], [30.0, 30.0, 40.0, 40.0]],
            [1, 2],
            [2, 0],
        )

        changed, triggered = update_alarm_states(
            result, "camera", self.polygon, 0, 3, states, alarms
        )

        self.assertFalse(changed)
        self.assertEqual(triggered, [])
        self.assertEqual(states, {})


if __name__ == "__main__":
    unittest.main()
