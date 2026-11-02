"""Serve the portable asset gallery on loopback and open it in a browser."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]

class GalleryHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/__gallery_status':
            body = json.dumps({'root': str(ROOT), 'service': 'adas-competition-gallery'}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *args):
        pass

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    # Try nearby ports if another application already uses the preferred port.
    for port in range(args.port, args.port + 20):
        url = f'http://127.0.0.1:{port}/'
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port),
                functools.partial(GalleryHandler, directory=str(ROOT)))
        except OSError:
            try:
                with urlopen(url + '__gallery_status', timeout=1) as response:
                    status = json.load(response)
                if status.get('service') == 'adas-competition-gallery' and status.get('root') == str(ROOT):
                    if not args.no_open:
                        webbrowser.open(url)
                    return
            except Exception:
                pass
            continue
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    raise RuntimeError('No available local gallery port')

if __name__ == '__main__':
    main()
