import unittest

import numpy as np

from app import easiest_shot


class ShotPlannerTests(unittest.TestCase):
    def setUp(self):
        self.rails = np.float32([[0, 0], [1000, 0], [1000, 500], [0, 500]])
        self.pockets = [(0, 0), (1000, 0), (1000, 500), (0, 500), (500, 0), (500, 500)]
        self.balls = np.array([[500, 400, 12], [500, 200, 12]])

    def test_straight_pot_uses_ghost_contact_position(self):
        shot = easiest_shot(self.balls, self.rails, self.pockets)
        self.assertEqual((shot['ball_index'], shot['pocket_index']), (1, 4))
        self.assertEqual(shot['ghost'], (500, 224))
        self.assertAlmostEqual(shot['cut_angle'], 0)

    def test_blocked_paths_are_rejected(self):
        for obstacle in ([500, 100, 12], [500, 300, 12], [518, 100, 12]):
            with self.subTest(obstacle=obstacle):
                shot = easiest_shot(np.vstack((self.balls, obstacle)), self.rails, self.pockets)
                if shot is not None:
                    self.assertNotEqual((shot['ball_index'], shot['pocket_index']), (1, 4))

    def test_missing_table_produces_no_recommendation(self):
        self.assertIsNone(easiest_shot(self.balls, [], []))

    def test_image_scaling_preserves_shot_choice(self):
        shot = easiest_shot(self.balls * 2, self.rails * 2,
                            np.asarray(self.pockets) * 2)
        self.assertEqual((shot['ball_index'], shot['pocket_index']), (1, 4))
        self.assertEqual(shot['ghost'], (1000, 448))


if __name__ == '__main__':
    unittest.main()
