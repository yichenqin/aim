import io
import re
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import app as module


class UploadIsolationTests(unittest.TestCase):
    def test_result_photo_is_downscaled_with_original_overlay_coordinates(self):
        image = np.full((1800, 3200, 3), 150, np.uint8)
        circles = np.array([[800, 450, 40], [2000, 900, 40]])
        _, encoded = cv2.imencode('.png', image)
        detection = (circles, image, np.full(image.shape[:2], 255, np.uint8), 1)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(module, 'UPLOAD_FOLDER', folder), \
                patch.object(module, 'detect_balls', return_value=detection):
            client = module.app.test_client()
            response = client.post('/upload', data={
                'image': (io.BytesIO(encoded.tobytes()), 'large.png')})
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn('viewBox="0 0 3200 1800"', html)
            name = re.search(r'/uploads/([0-9a-f]{32}_output\.jpg)', html).group(1)
            with client.get('/uploads/' + name) as result:
                preview = cv2.imdecode(np.frombuffer(result.data, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(preview.shape[:2], (450, 800))
            # The cue outline remains at its correctly scaled location.
            green = (preview[:, :, 1] > 180) & (preview[:, :, 0] < 100) & (preview[:, :, 2] < 100)
            yy, xx = np.nonzero(green)
            self.assertAlmostEqual(float(xx.mean()), 200, delta=2)
            self.assertAlmostEqual(float(yy.mean()), 112.5, delta=2)

    def test_simultaneous_uploads_keep_separate_results(self):
        barrier = threading.Barrier(2)
        paths = []

        def detect(path, return_table=False):
            paths.append(path)
            barrier.wait(timeout=10)
            image = cv2.imread(path)
            return (np.array([[30, 30, 8], [65, 65, 8]]), image,
                    np.full(image.shape[:2], 255, np.uint8), 1)

        def upload(value):
            image = np.full((100, 100, 3), value, np.uint8)
            _, encoded = cv2.imencode('.png', image)
            with module.app.test_client() as client:
                response = client.post('/upload', data={
                    'image': (io.BytesIO(encoded.tobytes()), 'same-name.png')})
                self.assertEqual(response.status_code, 200)
                name = re.search(r'/uploads/([0-9a-f]{32}_output\.jpg)',
                                 response.get_data(as_text=True)).group(1)
                with client.get('/uploads/' + name) as result:
                    self.assertEqual(result.status_code, 200)
                    return name, result.data

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(module, 'UPLOAD_FOLDER', folder), \
                patch.object(module, 'detect_balls', side_effect=detect):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(upload, [60, 180]))
            self.assertNotEqual(results[0][0], results[1][0])
            self.assertNotEqual(results[0][1], results[1][1])
            self.assertEqual(len(set(paths)), 2)
            self.assertTrue(all(not Path(path).exists() for path in paths))
            for name, data in results:
                with module.app.test_client().get('/uploads/' + name) as result:
                    self.assertEqual(result.data, data)
            self.assertEqual(len(list(Path(folder).iterdir())), 2)

    def test_failed_upload_removes_original(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(module, 'UPLOAD_FOLDER', folder):
            response = module.app.test_client().post('/upload', data={
                'image': (io.BytesIO(b'not an image'), 'bad.jpg')})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_raw_and_legacy_images_cannot_be_served(self):
        client = module.app.test_client()
        for name in ['input.jpg', 'output.jpg', 'a' * 32 + '_input.jpg']:
            self.assertEqual(client.get('/uploads/' + name).status_code, 404)


if __name__ == '__main__':
    unittest.main()
