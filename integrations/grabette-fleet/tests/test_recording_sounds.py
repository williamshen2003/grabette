"""Polling only cues actual transitions, not reloads or lost devices."""
from pathlib import Path
import subprocess


def test_recording_sound_transitions():
    source = (Path(__file__).parents[1] / 'app.py').read_text()
    code = source.split('const recordingSounds=new Map();', 1)[1].split('// A labelled On/Off', 1)[0]
    script = '''const assert=require('node:assert/strict');
const events=[]; const recordingCue=kind=>events.push(kind);
const recordingSounds=new Map();
const SESSIONS=[{id:'s',status:'open',members:{right:{device_id:'r'}},episodes:[{episode_id:'e'}]}];
const report={capture_episode_id:'e',is_capturing:false,is_stopping:false};
const DEVICES=[{device_id:'r',online:true,recording_buffers:report}];
CODE
checkRecordingSounds(); assert.deepEqual(events,[]);
report.is_capturing=true; checkRecordingSounds(); checkRecordingSounds();
assert.deepEqual(events,['start']);
DEVICES[0].online=false; checkRecordingSounds(); assert.deepEqual(events,['start']);
DEVICES[0].online=true; report.is_stopping=true; checkRecordingSounds();
assert.deepEqual(events,['start','stop']);
report.auto_stop_episode_id='e'; report.auto_stop_reason='Buffer nearly full';
checkRecordingSounds(); checkRecordingSounds(); assert.deepEqual(events,['start','stop','full']);
recordingSounds.clear(); checkRecordingSounds(); assert.equal(events.length,3);
'''.replace('CODE', code)
    subprocess.run(['node', '-e', script], check=True)
