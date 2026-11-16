import unittest

import cv2
import numpy as np

from app import enumerate_shots, top_down_table


class TopDownTableTests(unittest.TestCase):
    def test_shots_use_allowed_endpoints_for_geometry_and_photo_lines(self):
        rails = np.float32([[0, 0], [800, 0], [800, 400], [0, 400]])
        pockets = list(rails) + [(400, 0), (400, 400)]
        balls = [[300, 150, 10], [100, 12, 10]]
        overhead = top_down_table(balls, rails, pockets)
        shots = enumerate_shots(balls, pockets, overhead)
        for shot, allowed, pocket in zip(shots, overhead['allowed_pockets'], overhead['pockets']):
            endpoint = np.array([allowed['x'], allowed['y']])
            target = np.array(balls[1][:2])
            direction = endpoint - target
            distance = np.linalg.norm(direction)
            self.assertAlmostEqual(shot['target_distance'], distance)
            np.testing.assert_allclose([shot['px'], shot['py']], endpoint, atol=0.001)
            np.testing.assert_allclose([shot['ghost_x'], shot['ghost_y']],
                                       target - direction / distance * 2 * overhead['radius'])
            self.assertFalse(np.allclose(endpoint, [pocket['x'], pocket['y']]))

    def test_ball_diameter_uses_short_rail_ratio_not_detected_sizes(self):
        rails = np.float32([[0, 0], [800, 0], [800, 400], [0, 400]])
        pockets = list(rails) + [(400, 0), (400, 400)]
        for radii in ([5, 30], [50, 2]):
            result = top_down_table([[200, 100, radii[0]], [600, 300, radii[1]]], rails, pockets)
            self.assertAlmostEqual(2 * result['radius'], result['height'] / 19.55555)
            self.assertEqual(len(result['allowed_corners']), 4)
            for position, pocket in zip(result['allowed_corners'], result['pockets'][:4]):
                self.assertAlmostEqual(abs(position['x'] - pocket['x']), result['radius'])
                self.assertAlmostEqual(abs(position['y'] - pocket['y']), result['radius'])
            self.assertEqual(len(result['allowed_middle']), 2)
            for position, pocket in zip(result['allowed_middle'], result['pockets'][4:]):
                self.assertEqual(position['x'], pocket['x'])
                self.assertAlmostEqual(abs(position['y'] - pocket['y']),
                                       0.25 * result['height'] / 19.55555)
                self.assertTrue(0 < position['y'] < result['height'])

    def test_skewed_photo_recovers_table_coordinates(self):
        surface = np.float32([[0, 0], [800, 0], [800, 400], [0, 400]])
        rails = np.float32([[220, 60], [570, 100], [760, 600], [40, 550]])
        mapping = cv2.getPerspectiveTransform(surface, rails)
        expected = np.float32([[200, 100], [600, 300]])
        photo = cv2.perspectiveTransform(expected[None], mapping)[0]
        centers = cv2.perspectiveTransform(np.float32([[[400, 0], [400, 400]]]), mapping)[0]
        pockets = list(rails) + list(centers)
        result = top_down_table(np.column_stack((photo, [10, 10])), rails, pockets)
        np.testing.assert_allclose([[b['x'], b['y']] for b in result['balls']], expected, atol=0.001)
        self.assertEqual(result['width'] / result['height'], 2)
        self.assertEqual(result['pockets'][4:], [dict(x=400, y=0), dict(x=400, y=400)])

    def test_vertical_long_rails_rotate_to_horizontal(self):
        rails = np.float32([[0, 0], [400, 0], [400, 800], [0, 800]])
        pockets = list(rails) + [(400, 400), (0, 400)]
        result = top_down_table([[200, 400, 10]], rails, pockets)
        self.assertEqual(result['balls'], [dict(x=400, y=200)])
        self.assertEqual(result['pockets'][4:], [dict(x=400, y=0), dict(x=400, y=400)])

    def test_missing_geometry_does_not_invent_positions(self):
        self.assertIsNone(top_down_table([[200, 200, 10]], [], []))


if __name__ == '__main__':
    unittest.main()
