# Simple HTTP server for annotation
# Run: python start_server.py
# Then open: http://[REDACTED]:8000/simple_annotation.html

import http.server
import socketserver
import os

PORT = 8000

# Set directory to evaluation folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server running at [REDACTED]")
    print(f"Open: [REDACTED]")
    print("Press Ctrl+C to stop")
    httpd.serve_forever()
