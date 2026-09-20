"""Local operator UI; only authenticated device routes are exposed to the LAN."""
from fastapi.responses import JSONResponse
from app import app
from secrets import token_hex
from starlette.middleware.sessions import SessionMiddleware

# Local HTTP session cookies; operator routes are restricted to loopback below.
for middleware in app.user_middleware:
    if middleware.cls is SessionMiddleware:
        middleware.kwargs.update(https_only=False, same_site="lax", secret_key=token_hex(32))


@app.middleware('http')
async def local_operator_only(request, call_next):
    local = request.client and request.client.host in {'127.0.0.1', '::1'}
    if not local and not request.url.path.startswith('/api/devices/'):
        return JSONResponse({'detail': 'Open the dashboard on the host Mac at localhost:7860'}, status_code=403)
    return await call_next(request)
