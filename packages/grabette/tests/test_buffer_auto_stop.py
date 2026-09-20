"""Near-full RAM queues stop locally even when the fleet is unresponsive."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from grabette.backend.rpi import RpiBackend
from grabette.app.main import _stop_on_buffer_pressure


def test_pressure_stop_does_not_wait_for_network_and_registers_once(monkeypatch):
    backend = RpiBackend(enable_oakd=False)
    backend._capturing = True
    stats = {'pending_bytes': 94, 'capacity_bytes': 100, 'rejected_frames': 0}
    backend._camera = SimpleNamespace(_buffered_output=SimpleNamespace(
        buffer=SimpleNamespace(stats=lambda: stats)))
    assert not backend.buffer_pressure()
    stats['pending_bytes'] = 95
    assert 'wrist' in backend.buffer_pressure()
    backend._stopping = True
    assert not backend.buffer_pressure()
    backend._stopping = False
    backend.get_capture_status = lambda: SimpleNamespace(episode_id='episode1')
    manager = SimpleNamespace(register_episode=Mock())

    async def run():
        notified, release = asyncio.Event(), asyncio.Event()

        async def notify(**kw):
            notified.set()
            await release.wait()

        async def stop():
            backend._capturing = False
            return SimpleNamespace(episode_id='episode1')

        monkeypatch.setattr('grabette.fleet_sync.notify_group_stop', notify)
        backend.stop_capture = stop
        task = asyncio.create_task(_stop_on_buffer_pressure(backend, manager))
        try:
            await asyncio.wait_for(notified.wait(), 1)
            assert not backend.is_capturing
            manager.register_episode.assert_called_once_with('episode1')
            assert backend.auto_stop_episode_id == 'episode1'
            assert '95%' in backend.auto_stop_reason
        finally:
            release.set()
            await task
        await _stop_on_buffer_pressure(backend, manager)
        manager.register_episode.assert_called_once()

    asyncio.run(run())
