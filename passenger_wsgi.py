"""
cPanel / Passenger entry point.

cPanel's "Setup Python App" runs WSGI via Phusion Passenger, but this app is
ASGI (FastAPI). a2wsgi bridges the two. In the cPanel Python App screen set:
    Application startup file : passenger_wsgi.py
    Application Entry point  : application

The embedding model loads lazily on the first request (no lifespan needed here).
Set a writable FASTEMBED_CACHE_DIR env var in the cPanel app if the default isn't
writable, and remember the host must allow outbound internet for the model + AI API.
"""
from a2wsgi import ASGIMiddleware

from app.main import app

application = ASGIMiddleware(app)
