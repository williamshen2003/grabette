"""Stalled writes must not stall capture or lose accepted frames/timestamps."""
import io
import json
import sys
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from grabette.hardware.recording_buffer import RecordingBuffer
from grabette.hardware.oakd import OakdCapture


def test_stall_peak_overflow_and_drain():
    entered, release = threading.Event(), threading.Event()
    saved = []

    def write(item):
        entered.set()
        assert release.wait(3)
        saved.append(item)

    buffer = RecordingBuffer('depth', 10, write)
    try:
        assert buffer.submit(('frame1', 100), 4)
        assert entered.wait(1)
        assert buffer.submit(('frame2', 120), 4)  # capture progresses during stall
        assert not buffer.submit(('frame3', 140), 4)
        stats = buffer.stats()
        assert stats['peak_percent'] == 80  # includes frame currently being written
        assert stats['pending_bytes'] == 8
    finally:
        release.set()
        stats = buffer.close()
    assert saved == [('frame1', 100), ('frame2', 120)]
    assert stats['pending_bytes'] == 0 and stats['written_frames'] == 2
    assert stats['rejected_frames'] == 1 and not stats['complete']
    with pytest.raises(RuntimeError, match='closed'):
        buffer.submit('late', 1)


def test_write_and_final_flush_errors_are_visible():
    def fail(_=None):
        raise OSError('disk full')

    buffer = RecordingBuffer('video', 10, fail, finish=fail)
    buffer.submit(b'frame', 5)
    stats = buffer.close()
    assert stats['write_errors'] == 2 and stats['written_frames'] == 0
    assert stats['error'] == 'disk full' and not stats['complete']


def test_oak_stop_drains_before_pack_and_retains_peak(tmp_path, monkeypatch):
    cap = OakdCapture(SimpleNamespace(is_started=True))
    cap._initialized = True
    cap._calib_offline = {'fx': 1}
    cap.start_recording(tmp_path / 'one')
    entered, release = threading.Event(), threading.Event()
    original = cap._recording_buffers['depth']._write

    def slow(payload):
        entered.set()
        assert release.wait(3)
        original(payload)

    cap._recording_buffers['depth']._write = slow
    sample = {'seq': 42, 'host_ms': 123, 'device_us': 456}
    frame = np.full((4, 4), 1234, np.uint16)
    cap._recording_buffers['depth'].submit((frame, sample), frame.nbytes)
    cap._recording_buffers['oak_left'].submit((b'video', sample), 5)
    assert entered.wait(1)
    packed = []

    def pack():
        assert cap._depth_ts == [sample]
        assert (cap._output_dir / 'dcam_depth/00000042.png').exists()
        packed.append(True)

    monkeypatch.setattr(cap, '_pack_depth_video', pack)
    monkeypatch.setattr(cap, '_mux_h264_to_mp4', lambda *args: None)
    result = []
    stop = threading.Thread(target=lambda: result.append(cap.stop_recording()))
    stop.start()
    try:
        stop.join(.03)
        assert stop.is_alive() and not packed
    finally:
        release.set()
        stop.join(3)
    assert not stop.is_alive() and packed
    assert result[0]['buffers']['depth']['peak_bytes'] == frame.nbytes
    assert result[0]['recording_complete']
    assert json.loads((cap._output_dir / 'dcam_depth_timestamps.json').read_text())['samples'] == [sample]
    assert (cap._output_dir / 'dcam_left.h264').read_bytes() == b'video'
    monkeypatch.setattr(cap, '_pack_depth_video', lambda: None)
    cap.start_recording(tmp_path / 'two')
    assert cap.stop_recording()['buffers']['depth']['peak_bytes'] == 0


def test_depth_false_write_marks_incomplete_without_timestamp(tmp_path, monkeypatch):
    cap = OakdCapture(SimpleNamespace(is_started=True))
    cap._initialized = True
    cap._calib_offline = {'fx': 1}
    cap.start_recording(tmp_path)
    monkeypatch.setattr('grabette.hardware.oakd.cv2.imwrite', lambda *args: False)
    cap._recording_buffers['depth'].submit((np.zeros((2, 2), np.uint16), {'seq': 3}), 8)
    stats = cap.stop_recording()
    assert not stats['recording_complete'] and stats['depth_frames'] == 0
    assert stats['buffers']['depth']['write_errors'] == 1


def test_depth_drainer_continues_while_png_write_is_blocked(tmp_path, monkeypatch):
    import cv2
    cap = OakdCapture(SimpleNamespace(is_started=True, monotonic_s_to_ms=lambda t: t * 1000))
    cap._initialized = True
    cap._calib_offline = {'fx': 1}
    cap.start_recording(tmp_path)
    entered, release = threading.Event(), threading.Event()
    saved = []

    def blocked_write(path, frame, params):
        entered.set()
        assert release.wait(3)
        saved.append(frame.copy())
        return True

    monkeypatch.setattr(cv2, 'imwrite', blocked_write)
    sdk_frame = np.ones((2, 2), np.uint16)
    frames = deque(SimpleNamespace(
        getCvFrame=lambda: sdk_frame,
        getTimestamp=lambda: SimpleNamespace(total_seconds=lambda: .2),
        getTimestampDevice=lambda: SimpleNamespace(total_seconds=lambda: .2),
        getSequenceNum=lambda i=i: i,
    ) for i in (1, 2))
    cap._depth_q = SimpleNamespace(has=lambda: bool(frames), tryGet=frames.popleft)
    cap._stop_event.set()  # exit after draining the two fake SDK frames
    reader = threading.Thread(target=cap._writer_loop_depth)
    reader.start()
    try:
        assert entered.wait(1)
        reader.join(1)
        assert not reader.is_alive(), 'capture waited for a disk write'
        assert cap._recording_buffers['depth'].stats()['peak_bytes'] == 16
        sdk_frame[:] = 9  # SDK is free to reuse memory after our drainer returns
        assert cap._depth_ts == []
    finally:
        release.set()
        monkeypatch.setattr(cap, '_pack_depth_video', lambda: None)
        monkeypatch.setattr(cap, '_mux_h264_to_mp4', lambda *a: None)
        cap.stop_recording()
    assert [s['seq'] for s in cap._depth_ts] == [1, 2]
    assert len(saved) == 2 and all(np.all(f == 1) for f in saved)


def test_wrist_output_copies_payload_and_saves_pts_after_write(tmp_path, monkeypatch):
    # Exercise our output adapter against FileOutput's public write/stop contract.
    class FileOutput:
        def __init__(self, path, pts):
            self.fileoutput = open(path, 'wb')
            self.pts = pts
            self.recording = True

        def outputtimestamp(self, timestamp):
            self.pts.write(str(timestamp) + '\n')

        def close(self):
            self.fileoutput.close()

    monkeypatch.setitem(sys.modules, 'picamera2.outputs', SimpleNamespace(FileOutput=FileOutput))
    from grabette.hardware.camera import _buffered_output
    timestamps = io.StringIO()
    output = _buffered_output(tmp_path / 'wrist.h264', timestamps)
    entered, release = threading.Event(), threading.Event()
    original = output.buffer._write

    def slow(payload):
        entered.set()
        assert release.wait(3)
        original(payload)

    output.buffer._write = slow
    data = bytearray(b'first')
    output._write(data, 123)
    try:
        assert entered.wait(1)
        data[:] = b'other'
        output._write(b'second', 456)
        assert timestamps.getvalue() == ''
    finally:
        release.set()
        output.stop()
    assert (tmp_path / 'wrist.h264').read_bytes() == b'firstsecond'
    assert timestamps.getvalue() == '123\n456\n'
    assert output.buffer.stats()['peak_bytes'] == 11
    output.stop()  # idempotent when Picamera2 and capture cleanup both stop


def test_dashboard_summary_and_idle_status_survive_camera_reinit():
    from grabette.backend.rpi import RpiBackend
    from grabette.ui.app import buffer_summary
    backend = RpiBackend(enable_oakd=False)
    backend._last_buffer_episode_id = 'episode1'
    backend._last_buffer_stats = {'depth': {
        'peak_percent': 75.0, 'peak_bytes': 192 * 1024**2, 'capacity_bytes': 256 * 1024**2,
        'complete': False, 'rejected_frames': 2, 'write_errors': 0,
    }}
    cap = backend.get_capture_status().model_dump()
    text = buffer_summary(cap)
    assert 'Depth: 75.0% (192.0 / 256 MiB)' in text
    assert '2 rejected frames' in text and 'episode1' in text
    assert not cap['recording_complete']


def test_backend_saving_keeps_event_loop_live_and_persists_stats(tmp_path, monkeypatch):
    import asyncio
    from grabette.backend.rpi import RpiBackend
    backend = RpiBackend(enable_oakd=False)
    backend._capturing = True
    backend._episode_dir = tmp_path / 'episode1'
    backend._sync = SimpleNamespace(is_started=True, get_timestamp_ms=lambda: 1000, reset=lambda: None)
    stats = {'wrist': {'complete': False, 'peak_percent': 100, 'rejected_frames': 1}}
    entered, release = threading.Event(), threading.Event()
    metadata = []

    def stop_camera():
        entered.set()
        assert release.wait(3)
        return [0, 20]

    backend._camera = SimpleNamespace(stop=stop_camera, frame_count=2, buffer_stats=stats)
    monkeypatch.setattr(backend, '_note_angle_output', lambda _: None)
    monkeypatch.setattr(backend, '_finalize_and_reinit', lambda ep, ft, angles, meta, urdf: metadata.append(meta))

    async def run():
        task = asyncio.create_task(backend.stop_capture())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            assert backend.get_capture_status().is_stopping
            with pytest.raises(RuntimeError, match='already being saved'):
                await backend.stop_capture()
        finally:
            release.set()
        result = await task
        await asyncio.sleep(0)  # deferred metadata write
        assert result.buffer_stats == stats and not result.recording_complete

    asyncio.run(run())
    assert metadata[0]['buffers'] == stats and not metadata[0]['recording_complete']
    backend._camera = None  # camera re-init must not erase the last recording
    assert backend.get_capture_status().buffer_stats == stats
