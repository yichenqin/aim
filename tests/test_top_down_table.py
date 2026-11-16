import unittest

import cv2
import numpy as np

from app import top_down_table


class TopDownTableTests(unittest.TestCase):
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
