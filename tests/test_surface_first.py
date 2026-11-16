import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import detect_balls, find_playing_surface


class SurfaceFirstTests(unittest.TestCase):
    def test_background_white_object_is_not_selected_as_cue(self):
        image = np.full((450, 600, 3), 35, np.uint8)
        corners = np.int32([[100, 100], [450, 80], [530, 350], [50, 370]])
        cv2.fillConvexPoly(image, corners, (150, 150, 150))
        cv2.circle(image, (30, 30), 20, (255, 255, 255), -1)
        cv2.circle(image, (300, 200), 12, (245, 245, 245), -1)
        cv2.circle(image, (400, 240), 12, (0, 0, 220), -1)
        mask = find_playing_surface(image)
        self.assertEqual(mask[30, 30], 0)
        self.assertEqual(mask[200, 300], 255)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'photo.png')
            cv2.imwrite(path, image)
            balls, _, table, _ = detect_balls(path, return_table=True)
        np.testing.assert_allclose(balls[0][:2], [300, 200], atol=2)
        distance = cv2.distanceTransform(table, cv2.DIST_L2, 5)
        for x, y, radius in balls:
            self.assertGreaterEqual(distance[y, x], radius - 1)

    def test_no_rails_produces_error_instead_of_whole_image_search(self):
        image = np.full((450, 600, 3), 150, np.uint8)
        cv2.circle(image, (300, 200), 12, (255, 255, 255), -1)
        with self.assertRaisesRegex(ValueError, 'four rails'):
            find_playing_surface(image)


if __name__ == '__main__':
    unittest.main()
