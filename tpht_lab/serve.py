"""A static file server for the built site with the right MIME types.

Python's http.server takes types from the OS (the Windows registry, or
/etc/mime.types), which often lacks .mjs and .wasm; browsers refuse a module
script or a streamed wasm compile with the wrong type. This server pins the
types the site needs and can add the cross-origin isolation headers that
threaded WebAssembly would need (GitHub Pages cannot send them, so the site
does not depend on them).
"""
from __future__ import annotations

import functools
import http.server
import json
import socketserver
import threading
from pathlib import Path

TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.md': 'text/markdown; charset=utf-8',
    '.txt': 'text/plain; charset=utf-8',
    '.ts': 'text/plain; charset=utf-8',
    '.xml': 'application/xml',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.f32': 'application/octet-stream',
    '.mjb': 'application/octet-stream',
    '.onnx': 'application/octet-stream',
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **TYPES}
    isolate = False
    quiet = False
    results_sink = None  # a queue.Queue in `tpht-lab test`: tests/index.html?report=1 POSTs here

    def do_POST(self):
        if self.results_sink is None or self.path != '/__results':
            self.send_error(404)
            return
        length = int(self.headers.get('Content-Length', 0))
        try:
            report = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            self.send_error(400)
            return
        self.send_response(204)
        self.end_headers()
        self.results_sink.put(report)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        if self.isolate:
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(directory: Path, host: str = '127.0.0.1', port: int = 8000,
                isolate: bool = False, quiet: bool = False, results_sink=None) -> Server:
    directory = Path(directory).resolve()
    if not (directory / 'index.html').is_file():
        raise SystemExit(f'{directory} has no index.html: point --dir at a built site '
                         '(tools/build_site.py output, or <prefix>/share/tpht-viz-panel)')
    handler = type('SiteHandler', (Handler,), dict(isolate=isolate, quiet=quiet, results_sink=results_sink))
    return Server((host, port), functools.partial(handler, directory=str(directory)))


def serve_in_thread(directory: Path, **kw) -> tuple[Server, str]:
    """Starts a server on a free port (port=0) in a daemon thread; returns it and its base URL."""
    kw.setdefault('port', 0)
    kw.setdefault('quiet', True)
    server = make_server(directory, **kw)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, f'http://{host}:{port}/'
