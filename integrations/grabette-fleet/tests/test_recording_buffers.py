"""Buffer reports cross the authenticated heartbeat and stay scoped in the UI."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app


def test_heartbeat_and_session_buffer_display():
    owner, device_id = 'buffer-test', 'right'
    dev = app.Device(device_id, '<right>', [])
    app.FLEET[owner] = {device_id: dev}
    report = app.RecordingTelemetry(episode_id='episode1', buffers={'depth': {
        'capacity_bytes': 268435456, 'peak_bytes': 134217728, 'peak_percent': 50,
        'rejected_frames': 2, 'write_errors': 0, 'complete': False, 'error': '<failed>',
    }})
    try:
        asyncio.run(app.heartbeat(device_id, telemetry=report, auth=(owner, 'token')))
        assert dev.recording_buffers == report.model_dump()
        asyncio.run(app.heartbeat(device_id, auth=(owner, 'token')))
        assert dev.recording_buffers == report.model_dump()  # legacy heartbeat
        source = Path(app.__file__).read_text()
        render = source.split('function recordingBuffersHtml(s){', 1)[1].split('function renderSessionList(){', 1)[0]
        js = '''const assert=require('node:assert/strict');
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const DEVICES=[{device_id:'right',online:true,recording_buffers:REPORT}];
function recordingBuffersHtml(s){RENDER
const s={members:{right:{device_id:'right',name:'<right>'}},episodes:[{episode_id:'episode1',roles:{right:'right'}}]};
const html=recordingBuffersHtml(s);
assert(html.includes('50.0%')); assert(html.includes('128.0 / 256 MiB'));
assert(html.includes('2 rejected frames')); assert(html.includes('&lt;failed&gt;'));
assert(html.includes('&lt;right&gt;')); assert(!html.includes('<failed>'));
DEVICES[0].online=false; assert(recordingBuffersHtml(s).includes('Offline'));
s.episodes=[]; assert(!recordingBuffersHtml(s).includes('50.0%'));
assert(recordingBuffersHtml(s).includes('No buffer report'));
'''.replace('REPORT', json.dumps(dev.recording_buffers)).replace('RENDER', render)
        subprocess.run(['node', '-e', js], check=True)
    finally:
        app.FLEET.pop(owner, None)
