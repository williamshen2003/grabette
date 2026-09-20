"""Local login must not expose the operator dashboard to the LAN."""
from fastapi.testclient import TestClient
from local_app import app


def test_local_operator_and_lan_device_boundary():
    local = TestClient(app, client=('127.0.0.1', 12345))
    lan = TestClient(app, client=('192.168.1.85', 12345))
    assert local.get('/').status_code == 200
    assert lan.get('/').status_code == 403
    assert lan.get('/oauth/huggingface/login').status_code == 403
    assert lan.get('/api/fleet/devices').status_code == 403
    assert lan.get('/', headers={'X-Forwarded-For': '127.0.0.1'}).status_code == 403
    assert lan.post('/api/devices/heartbeat?device_id=test').status_code == 401
