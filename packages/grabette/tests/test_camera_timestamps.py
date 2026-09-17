"""Encoder startup/drop/drain must not shift the camera's capture timeline."""
from types import SimpleNamespace
from unittest.mock import Mock

from grabette.hardware.camera import VideoCapture
from grabette.hardware.sync import SyncManager


def test_timestamps_follow_saved_frames(tmp_path):
    sync = SyncManager()
    sync._start_time = 100.0
    sync._start_boottime = 100.0
    camera = VideoCapture(sync)
    encoder = SimpleNamespace(firsttimestamp=100_800_000)
    pts = None

    def start_encoder(enc, path, **kwargs):
        nonlocal pts
        # Sensor callbacks continue through an 800ms encoder startup.
        callback = getattr(camera._picam2, 'pre_callback', None)
        if callback:
            for i in range(43):
                callback(SimpleNamespace(get_metadata=lambda i=i: {
                    'SensorTimestamp': 100_000_000_000 + i * 20_000_000,
                }))
        pts = kwargs.get('pts')
        if pts is not None:
            # Only these frames made it to the video; the 20ms frame dropped.
            pts.write('0.000\n40.000\n')

    def stop_encoder():
        if pts is not None:
            pts.write('60.000\n')  # encoder drains a final frame at stop

    camera._encoder = encoder
    camera._picam2 = SimpleNamespace(
        start_encoder=start_encoder, stop_encoder=stop_encoder,
        stop=Mock(), close=Mock(),
    )
    camera._mux_to_mp4 = Mock()
    camera.start_recording(tmp_path / 'raw_video.mp4')
    assert camera.stop() == [800.0, 840.0, 860.0]


def test_mux_mismatch_keeps_raw_video(tmp_path, monkeypatch):
    import pytest

    camera = VideoCapture(SyncManager())
    camera._output_path = tmp_path / 'raw_video.mp4'
    camera._h264_path = tmp_path / 'raw_video.h264'
    camera._h264_path.write_bytes(b'raw recording')
    camera._frame_timestamps = [800.0, 820.0]
    monkeypatch.setattr('grabette.hardware.camera.subprocess.run',
                        lambda *a, **kw: SimpleNamespace(returncode=0))
    camera._count_frames_ffprobe = lambda: 1
    with pytest.raises(RuntimeError, match='video/timestamp mismatch'):
        camera._mux_to_mp4()
    assert camera._h264_path.read_bytes() == b'raw recording'
