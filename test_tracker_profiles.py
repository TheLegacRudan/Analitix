import unittest

from detect_video import select_tracker_profile


class TrackerProfileSelectionTests(unittest.TestCase):
    def test_selects_night_profile_before_crowd_profile(self):
        self.assertEqual(
            select_tracker_profile([40, 60, 70], [8, 10, 7]),
            "night",
        )

    def test_selects_crowd_profile_for_five_or_more_people(self):
        self.assertEqual(
            select_tracker_profile([100, 120, 110], [2, 5, 1]),
            "crowd",
        )

    def test_defaults_to_warehouse_profile(self):
        self.assertEqual(
            select_tracker_profile([100, 120, 110], [2, 3, 1]),
            "default",
        )

    def test_requires_at_least_one_sampled_frame(self):
        with self.assertRaisesRegex(ValueError, "At least one video frame"):
            select_tracker_profile([], [])


if __name__ == "__main__":
    unittest.main()
