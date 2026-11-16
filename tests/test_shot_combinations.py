import unittest

import cv2
import numpy as np

from app import cue_view_overlap, enumerate_shots, top_down_table


class ShotCombinationTests(unittest.TestCase):
    def test_cue_view_overlap_examples(self):
        for angle, expected in [(0, 1), (30, 0.5), (45, 1 - np.sqrt(2) / 2), (90, 0)]:
            with self.subTest(angle=angle):
                geometry = cue_view_overlap(angle)
                self.assertAlmostEqual(geometry['overlap'], expected)
                self.assertAlmostEqual(geometry['offset'], 1 - expected)
        self.assertLess(cue_view_overlap(30, -1)['offset'], 0)
        self.assertFalse(cue_view_overlap(120)['feasible'])
        self.assertIsNone(cue_view_overlap(None))

    def test_direction_angles_and_sequential_numbers(self):
        shots = enumerate_shots([[0, 0, 1], [10, 0, 1], [20, 10, 1]],
                                [(20, 0), (10, 10), (0, 0)])
        self.assertEqual([shot['number'] for shot in shots], list(range(1, 7)))
        self.assertEqual([shot['angle'] for shot in shots[:3]], [0, 90, 180])
        self.assertEqual([(shot['target'], shot['pocket']) for shot in shots],
                         [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)])

    def test_coincident_points_have_undefined_angles(self):
        shots = enumerate_shots([[0, 0, 1], [10, 0, 1]], [(10, 0)])
        self.assertIsNone(shots[0]['angle'])
        self.assertIsNone(shots[0]['ghost_angle'])
        self.assertIsNone(shots[0]['ghost_x'])

    def test_ghost_tangency_direction_and_distinct_angle(self):
        overhead = dict(radius=10, balls=[dict(x=100, y=200), dict(x=300, y=100)],
                        allowed_pockets=[dict(x=700, y=100)],
                        photo_allowed_pockets=[dict(x=2, y=2)])
        shot = enumerate_shots([[0, 0, 1], [1, 1, 1]], [(2, 2)], overhead)[0]
        self.assertEqual((shot['ghost_x'], shot['ghost_y']), (280, 100))
        self.assertAlmostEqual(shot['angle'], np.degrees(np.arctan2(100, 200)))
        self.assertAlmostEqual(shot['ghost_angle'], np.degrees(np.arctan2(100, 180)))
        self.assertGreater(shot['ghost_angle'], shot['angle'])
        self.assertAlmostEqual(shot['cue_distance'], np.hypot(180, 100))
        self.assertAlmostEqual(shot['target_distance'], 400)
        self.assertAlmostEqual(shot['total_distance'], np.hypot(180, 100) + 400)

    def test_cue_at_ghost_has_undefined_ghost_angle(self):
        shot = enumerate_shots([[8, 0, 1], [10, 0, 1]], [(20, 0)])[0]
        self.assertEqual(shot['angle'], 0)
        self.assertIsNone(shot['ghost_angle'])

    def test_missing_pockets_produces_no_combinations(self):
        self.assertEqual(enumerate_shots([[0, 0, 1], [10, 0, 1]], []), [])

    def test_perspective_changes_photo_angle_but_not_top_down_angle(self):
        rectangle = np.float32([[0, 0], [800, 0], [800, 400], [0, 400]])
        rails = np.float32([[220, 60], [570, 100], [760, 600], [40, 550]])
        transform = cv2.getPerspectiveTransform(rectangle, rails)
        # (-100, 100) and (-100, -100) form a right angle toward P1.
        positions = np.float32([[[200, 0], [100, 100]]])
        photo = cv2.perspectiveTransform(positions, transform)[0]
        balls = np.column_stack((photo, [10, 10]))
        sides = cv2.perspectiveTransform(np.float32([[[400, 0], [400, 400]]]), transform)[0]
        pockets = list(rails) + list(sides)
        overhead = top_down_table(balls, rails, pockets)
        corrected = enumerate_shots(balls, pockets, overhead)
        original = enumerate_shots(balls, pockets)
        self.assertAlmostEqual(corrected[0]['angle'], 90, places=3)
        self.assertGreater(abs(original[0]['angle'] - 90), 1)
        self.assertEqual(corrected[0]['x'], original[0]['x'])
        self.assertNotEqual(corrected[0]['label_x'], original[0]['label_x'])


if __name__ == '__main__':
    unittest.main()
