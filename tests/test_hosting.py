import io
import unittest
from unittest.mock import patch

from app import app


class HostingTests(unittest.TestCase):
    def test_opening_upload_link_returns_to_form(self):
        client = app.test_client()
        response = client.get('/upload')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/')
        response = client.get('/upload', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Pool Shot Assistant', response.data)
        self.assertEqual(client.post('/upload').status_code, 400)

    def test_health_check(self):
        response = app.test_client().get('/healthz')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'status': 'ok'})

    def test_oversized_upload_is_rejected(self):
        with patch.dict(app.config, MAX_CONTENT_LENGTH=1024):
            response = app.test_client().post('/upload', data={
                'image': (io.BytesIO(b'x' * 2048), 'large.jpg')})
        self.assertEqual(response.status_code, 413)
        self.assertIn(b'20 MB', response.data)


if __name__ == '__main__':
    unittest.main()
