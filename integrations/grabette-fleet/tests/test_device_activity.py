"""What the fleet says a device is doing: _device_activity and the recording gate.

The badge in the fleet list and the "busy" gates both read _device_activity, and
it answers from three sources of decreasing authority: work the fleet dispatched
itself, the device's own heartbeat report, then — only for a device that reports
nothing at all — an inference from the session. These tests pin the precedence,
because getting it wrong is invisible in code review and very visible to an
operator: it once painted every member of an open session "● Recording" while all
of them sat idle between takes.

Pure in-memory reads, no HTTP and no devices needed.

Run with:  pip install -r requirements.txt pytest  &&  python -m pytest tests/
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

OWNER = "operator"
LEFT, RIGHT = "dev-left-1", "dev-right-1"


@pytest.fixture(autouse=True)
def clean_state():
    for store in (app.FLEET, app.TASKS, app.SESSIONS, app.GROUPS, app.TASK_SUPPRESSED,
                  app.ORPHANS_PENDING, app.SPLIT_PENDING, app._REPORTED_CACHE,
                  app._REPORTERS_CACHE):
        store.clear()
    yield


def grabette(device_id, name, hand, status=""):
    """A registered device whose heartbeat reported `status` ("" = never reported)."""
    dev = app.Device(device_id, name, ["start_capture"], hand=hand)
    dev.reported_status = status
    return dev


def fleet(*devices):
    app.FLEET[OWNER] = {d.device_id: d for d in devices}
    return {d.device_id: d for d in devices}


def session(*, recording, status="open", members=None):
    s = app.Session(id="sess-1", task_id="task-1",
                    members=members or {"left": LEFT, "right": RIGHT},
                    status=status, recording=recording)
    app.SESSIONS[OWNER] = {s.id: s}
    return s


def activity(dev):
    return app._device_activity(OWNER, dev)


# ── the device's own word wins ───────────────────────────────────────────

def test_a_device_that_reports_idle_is_idle_even_inside_an_open_session():
    # The regression: a session stays open between takes, and its members are
    # idle for most of that time. Inferring "capturing" from membership made the
    # fleet contradict devices that were plainly stopped.
    devs = fleet(grabette(LEFT, "grabette-left", "left", status="idle"),
                 grabette(RIGHT, "grabette-right", "right", status="idle"))
    session(recording=False)
    assert [activity(d) for d in devs.values()] == ["idle", "idle"]


def test_a_device_that_reports_idle_is_idle_even_while_the_session_believes_it_records():
    # The session's flag is optimistic (set on start, cleared by an explicit
    # stop). When the two disagree, the device is the one holding the camera.
    devs = fleet(grabette(LEFT, "grabette-left", "left", status="idle"))
    session(recording=True, members={"left": LEFT})
    assert activity(devs[LEFT]) == "idle"


def test_a_recording_device_is_capturing():
    devs = fleet(grabette(LEFT, "grabette-left", "left", status="capturing"))
    session(recording=True, members={"left": LEFT})
    assert activity(devs[LEFT]) == "capturing"


def test_a_solo_recording_outside_any_session_is_still_capturing():
    # A button press with no session open records locally; the fleet only ever
    # learns about it from the heartbeat, so it must not be swallowed.
    devs = fleet(grabette(LEFT, "grabette-left", "left", status="capturing"))
    assert activity(devs[LEFT]) == "capturing"


def test_local_dashboard_work_is_reported_verbatim():
    devs = fleet(grabette(LEFT, "grabette-left", "left", status="uploading"),
                 grabette(RIGHT, "grabette-right", "right", status="processing"))
    assert [activity(d) for d in devs.values()] == ["uploading", "processing"]


# ── the fallback, for a device that reports nothing ──────────────────────

def test_a_silent_device_is_capturing_only_while_its_session_records():
    devs = fleet(grabette(LEFT, "grabette-left", "left"))  # never reported
    s = session(recording=True, members={"left": LEFT})
    assert activity(devs[LEFT]) == "capturing"
    s.recording = False  # episode over, session still open for the next take
    assert activity(devs[LEFT]) == "idle"


def test_a_silent_device_outside_any_session_is_idle():
    devs = fleet(grabette(LEFT, "grabette-left", "left"))
    session(recording=True, status="closed", members={"left": LEFT})
    assert activity(devs[LEFT]) == "idle"


# ── dispatched work outranks everything ──────────────────────────────────

def test_work_the_fleet_dispatched_outranks_the_device_report():
    # A relay command creates no local job, so the device may well still report
    # "idle" while converting — the queue is the authority here.
    dev = grabette(LEFT, "grabette-left", "left", status="idle")
    dev.queue.append(app.Command(id="c1", type="process_dataset", args={}))
    fleet(dev)
    assert activity(dev) == "processing"


# ── interaction with the recording gate ──────────────────────────────────

def test_an_idle_member_of_an_open_session_is_not_busy_for_recording():
    # _busy_recording_blockers gates launching a session and starting an episode.
    # Members sitting idle in a session must never gate themselves out of it.
    fleet(grabette(LEFT, "grabette-left", "left", status="idle"),
          grabette(RIGHT, "grabette-right", "right", status="idle"))
    session(recording=False)
    assert app._busy_recording_blockers(OWNER, [LEFT, RIGHT]) == []


def test_a_device_converting_a_dataset_is_busy_for_recording():
    fleet(grabette(LEFT, "grabette-left", "left", status="processing"),
          grabette(RIGHT, "grabette-right", "right", status="idle"))
    assert app._busy_recording_blockers(OWNER, [LEFT, RIGHT]) == [LEFT]
