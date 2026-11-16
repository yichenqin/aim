import unittest

import cv2
import numpy as np

from app import ball_appearance_supported, ball_inside_surface


class BallAppearanceTests(unittest.TestCase):
    def test_touching_balls_survive_small_inset_rail_error(self):
        for corner in (False, True):
            with self.subTest(corner=corner):
                image = np.full((120, 120, 3), 150, np.uint8)
                cv2.circle(image, (60, 60), 16, (0, 0, 210), -1)
                table = np.zeros((120, 120), np.uint8)
                table[46 if corner else 0:, 46:] = 255
                distance = cv2.distanceTransform(table, cv2.DIST_L2, 5)
                self.assertTrue(ball_inside_surface(distance, (60, 60, 16)))
                self.assertTrue(ball_appearance_supported(
                    cv2.cvtColor(image, cv2.COLOR_BGR2LAB), table, (60, 60, 16)))
                self.assertFalse(ball_inside_surface(distance, (44, 60, 16)))
                self.assertFalse(ball_inside_surface(distance, (48, 60, 16)))

    def supported(self, image):
        return ball_appearance_supported(cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
                                         np.full(image.shape[:2], 255, np.uint8),
                                         (60, 60, 16))

    def test_plain_felt_and_straight_shadow_are_not_balls(self):
        image = np.full((120, 120, 3), 150, np.uint8)
        self.assertFalse(self.supported(image))
        image[:, 60:] = 110
        self.assertFalse(self.supported(image))

    def test_elongated_shadow_is_not_a_ball(self):
        image = np.full((120, 120, 3), 150, np.uint8)
        cv2.ellipse(image, (60, 60), (30, 5), 25, 0, 360, (110, 110, 110), -1)
        self.assertFalse(self.supported(cv2.GaussianBlur(image, (7, 7), 0)))

    def test_round_felt_shadows_do_not_borrow_adjacent_ball_evidence(self):
        for brightness in (90, 110, 125):
            with self.subTest(brightness=brightness):
                image = np.full((120, 120, 3), 150, np.uint8)
                cv2.circle(image, (60, 60), 16, (brightness,) * 3, -1)
                cv2.circle(image, (87, 36), 16, (0, 0, 220), -1)
                self.assertFalse(self.supported(image))

    def test_dark_white_colored_and_striped_balls_are_supported(self):
        for color in [(15, 15, 15), (240, 240, 240), (0, 0, 200), (160, 30, 30)]:
            with self.subTest(color=color):
                image = np.full((120, 120, 3), 150, np.uint8)
                cv2.circle(image, (60, 60), 16, color, -1)
                self.assertTrue(self.supported(image))
        cv2.rectangle(image, (45, 56), (75, 64), (240, 240, 240), -1)
        self.assertTrue(self.supported(image))


if __name__ == '__main__':
    unittest.main()
