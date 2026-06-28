# Test script to verify frames are accessible
import os
import http.server
import socketserver
from urllib.parse import quote

# Change to project directory
os.chdir('D:/Project/Face_project/data/annotations')

PORT = 9000
print(f"Server started at [REDACTED]")
print("Frames directory: ./frames/")
print("")
print("Test URLs:")
print(f"  /v1/images/1bc29f0b-f0e8-499d-af45-8f96aabac0a2?token=32mrO032R0A9bHOi2ERPtbuIsNqJIxFZ")
print(f"  /v1/images/45de8f12-63f7-4f9d-aad7-42fe00889181?token=3WsAXLciyWnyIxn9mWLxhAlAOMe53KMR")
print("")

httpd = socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler)
httpd.serve_forever()
