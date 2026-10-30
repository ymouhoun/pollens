import base64
import unittest

from remote_images import MAX_BYTES, RemoteImageError, inline_remote_images

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff" + b"\x00" * 32


class InlineRemoteImagesTests(unittest.TestCase):
    def test_url_entries_become_base64_and_others_stay(self):
        seen = []

        def fetch(url):
            seen.append(url)
            return (PNG, "image/png") if "a.png" in url else (JPEG, "application/octet-stream")

        job = {"input": {"images": [
            {"name": "pollen-edit-source.png", "url": "https://beta.pollens.ai/media/a.png?e=1&s=x"},
            {"name": "pollen-edit-product.png", "url": "https://beta.pollens.ai/media/b.jpg?e=1&s=x"},
            {"name": "face.png", "image": "data:image/png;base64,AAAA"},
        ]}}
        names = []
        self.assertEqual(inline_remote_images(job, fetch=fetch, status=names.append), 2)
        images = job["input"]["images"]
        self.assertEqual(images[0]["image"], "data:image/png;base64," + base64.b64encode(PNG).decode())
        # the file's own bytes decide its type, not the header
        self.assertTrue(images[1]["image"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("url", images[0])
        self.assertEqual(images[2]["image"], "data:image/png;base64,AAAA")
        self.assertEqual(len(seen), 2)
        self.assertEqual(names, ["pollen-edit-source.png", "pollen-edit-product.png"])

    def test_refusals(self):
        with self.assertRaises(RemoteImageError):
            inline_remote_images({"input": {"images": [{"name": "x", "url": "http://beta.pollens.ai/media/x"}]}}, fetch=lambda u: (PNG, ""))
        with self.assertRaises(RemoteImageError):
            inline_remote_images({"input": {"images": [{"name": "x", "url": "https://e/x"}]}}, fetch=lambda u: (b"<html>", "text/html"))
        with self.assertRaises(RemoteImageError):
            inline_remote_images({"input": {"images": [{"name": "x", "url": "https://e/x"}]}}, fetch=lambda u: (PNG + b"\x00" * MAX_BYTES, "image/png"))

        def expired(url):
            raise OSError("HTTP Error 403: Forbidden")

        with self.assertRaisesRegex(RemoteImageError, "403"):
            inline_remote_images({"input": {"images": [{"name": "x", "url": "https://e/x"}]}}, fetch=expired)

    def test_redirects_are_not_followed(self):
        import http.server
        import threading
        import remote_images

        class Redirect(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/secret")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
        threading.Thread(target=server.handle_request, daemon=True).start()
        try:
            with self.assertRaises(Exception) as raised:
                remote_images._fetch(f"http://127.0.0.1:{server.server_port}/x")
            self.assertIn("302", str(raised.exception))
        finally:
            server.server_close()

    def test_nothing_to_do(self):
        self.assertEqual(inline_remote_images({"input": {"workflow": {}}}), 0)
        self.assertEqual(inline_remote_images({}), 0)


if __name__ == "__main__":
    unittest.main()
