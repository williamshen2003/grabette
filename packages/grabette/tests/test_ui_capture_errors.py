"""Failed camera/capture commands must surface through the actual UI callbacks."""
from unittest.mock import Mock

import pytest

gr = pytest.importorskip("gradio")


@pytest.mark.parametrize("operation", ["start", "stop", "enable"])
def test_failed_camera_command_shows_error_without_success_update(monkeypatch, caplog, operation):
    from grabette.ui import app

    message = "Depth camera (oakd) failed to initialize: No available devices"
    client = Mock()
    client.get_state.return_value = {"capture": {"is_capturing": operation == "stop"}}
    client.get_oakd_status.return_value = {"supported": True, "enabled": False}
    client.start_capture.return_value = {"error": message}
    client.stop_capture.return_value = {"error": message}
    client.list_tasks.return_value = []
    # Enable returns HTTP 200 with the latched hardware fault in its status.
    client.set_oakd.return_value = {"hardware_error": message}
    monkeypatch.setattr(app, "GrabetteClient", lambda **kwargs: client)
    with app.create_ui() as demo:
        name = "on_toggle_oakd" if operation == "enable" else "on_toggle_capture"
        callback = next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__ == name)
        with pytest.raises(gr.Error, match="No available devices"):
            callback() if operation == "enable" else callback("session")
    assert message in caplog.text
