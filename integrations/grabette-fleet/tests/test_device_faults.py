"""Hardware faults: a grabette that refuses to record must SAY so, to the operator.

A device now refuses to record when it cannot produce convertible episodes (no
OAK-D offline calibration, no gripper angle sensors). That refusal is worthless
if it only exists on the device: the operator is looking at the fleet dashboard,
not at a grabette's LED on a bench in another room.

Worse, the fleet dispatches synchronized starts fire-and-forget —
_schedule_episode_start enqueues start_capture to every member and never looks at
a single result. So a member that refused left its peers recording a half-rig
take with nothing anywhere saying so, and the operator found out at dataset build
time, if at all.

Two things must therefore hold, and they are different:
  * a known fault is visible BEFORE the press, and blocks the dispatch;
  * a start that failed anyway is surfaced on the session AFTER the fact,
    because the command result is the only place the fleet can ever learn it.

Pure in-memory reads, no HTTP and no devices needed.

Run with:  pip install -r requirements.txt pytest  &&  python -m pytest tests/
"""
import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

OWNER = "operator"
LEFT, RIGHT = "dev-left-1", "dev-right-1"
CALIB_FAULT = ("the OAK-D reported an unusable calibration (fx=0.0) — this "
               "grabette cannot record convertible episodes.")


@pytest.fixture(autouse=True)
def clean_state():
    for store in (app.FLEET, app.TASKS, app.SESSIONS, app.GROUPS, app.TASK_SUPPRESSED,
                  app.ORPHANS_PENDING, app.SPLIT_PENDING, app._REPORTED_CACHE,
                  app._REPORTERS_CACHE):
        store.clear()
    yield


def grabette(device_id, name, hand, fault=""):
    dev = app.Device(device_id, name, ["start_capture"], hand=hand)
    dev.hardware_error = fault
    return dev


def fleet(*devices):
    app.FLEET[OWNER] = {d.device_id: d for d in devices}
    return app.FLEET[OWNER]


def session(members, **kw):
    s = app.Session(id="sess-1", task_id="task-1", members=members, **kw)
    app.SESSIONS[OWNER] = {s.id: s}
    return s


# --- the fault reaches the fleet, and stops reaching it when fixed ------------

def beat(device_id, **kw):
    """One heartbeat, through the real handler (auth stubbed out)."""
    return asyncio.run(app.heartbeat(device_id, auth=(OWNER, "token"), **kw))


def test_a_fault_reported_on_the_heartbeat_is_stored():
    fleet(grabette(LEFT, "grabette-01", "left"))

    beat(LEFT, error=CALIB_FAULT)

    assert app.FLEET[OWNER][LEFT].hardware_error == CALIB_FAULT


def test_an_empty_report_clears_the_fault():
    # The device sends the field on EVERY beat, empty included — that is what
    # makes "it's fixed now" expressible at all. A field that only appears when
    # broken could never say "fixed".
    fleet(grabette(LEFT, "grabette-01", "left", fault=CALIB_FAULT))

    beat(LEFT, error="")

    assert app.FLEET[OWNER][LEFT].hardware_error == ""


def test_an_older_device_that_sends_nothing_is_not_declared_healthy():
    # `error` absent means "this build doesn't report faults", which is not the
    # same as "no fault" — inventing good news is how a fault gets lost.
    fleet(grabette(LEFT, "grabette-01", "left", fault=CALIB_FAULT))

    beat(LEFT, battery=80.0)

    assert app.FLEET[OWNER][LEFT].hardware_error == CALIB_FAULT


def test_the_device_listing_exposes_the_fault(monkeypatch):
    fleet(grabette(LEFT, "grabette-01", "left", fault=CALIB_FAULT))

    row = next(d for d in _listing(monkeypatch) if d["device_id"] == LEFT)

    assert row["hardware_error"] == CALIB_FAULT
    # Still reported as idle: the fault is a separate axis from what it's doing.
    assert row["activity"] == "idle"


def _listing(monkeypatch):
    monkeypatch.setattr(app, "operator_name", lambda request: OWNER)
    return asyncio.run(app.list_devices(object()))["devices"]


def test_a_fault_is_not_an_activity():
    # Orthogonal on purpose: a faulted device can also be uploading, and folding
    # the fault into the activity enum would hide one behind the other.
    dev = grabette(LEFT, "grabette-01", "left", fault=CALIB_FAULT)
    dev.reported_status = "uploading"
    fleet(dev)

    assert app._device_activity(OWNER, dev) == "uploading"
    assert dev.hardware_error == CALIB_FAULT


# --- a known fault blocks the dispatch ---------------------------------------

def test_a_faulted_member_blocks_the_synchronized_start():
    # THE regression: the dispatch only ever checked that devices were ONLINE, so
    # a faulted grabette let its peer record a take that was unusable the moment
    # it was missing an arm.
    fleet(grabette(LEFT, "grabette-01", "left", fault=CALIB_FAULT),
          grabette(RIGHT, "grabette-02", "right"))
    s = session({"left": LEFT, "right": RIGHT})

    with pytest.raises(HTTPException) as e:
        app._schedule_episode_start(OWNER, "cup grasping", s.members, 1.0, session=s)

    assert e.value.status_code == 409
    faulted = e.value.detail["faulted"]
    assert [f["name"] for f in faulted] == ["grabette-01"]
    assert faulted[0]["error"] == CALIB_FAULT  # names the fix, not just the fact


def test_nothing_is_dispatched_when_a_member_is_faulted():
    # A partial dispatch would be the worst outcome: the healthy peer starts
    # recording alone while the refusal is reported as a failure.
    f = fleet(grabette(LEFT, "grabette-01", "left", fault=CALIB_FAULT),
              grabette(RIGHT, "grabette-02", "right"))
    s = session({"left": LEFT, "right": RIGHT})

    with pytest.raises(HTTPException):
        app._schedule_episode_start(OWNER, "cup grasping", s.members, 1.0, session=s)

    assert f[RIGHT].queue == []


def test_healthy_members_still_start():
    f = fleet(grabette(LEFT, "grabette-01", "left"),
              grabette(RIGHT, "grabette-02", "right"))
    s = session({"left": LEFT, "right": RIGHT})

    app._schedule_episode_start(OWNER, "cup grasping", s.members, 1.0, session=s)

    assert [c.type for c in f[LEFT].queue] == ["start_capture"]
    assert [c.type for c in f[RIGHT].queue] == ["start_capture"]


# --- a start that failed anyway is surfaced ----------------------------------

def _report(dev, cmd_id, result):
    """One command result, through the real /api/devices/result handler."""
    return asyncio.run(app.result(
        app.ResultReq(device_id=dev.device_id, command_id=cmd_id, result=result),
        auth=(OWNER, "token")))


def test_a_failed_start_lands_on_the_session():
    f = fleet(grabette(LEFT, "grabette-01", "left"),
              grabette(RIGHT, "grabette-02", "right"))
    s = session({"left": LEFT, "right": RIGHT})
    app._schedule_episode_start(OWNER, "cup grasping", s.members, 1.0, session=s)

    _report(f[LEFT], f[LEFT].queue[0].id,
            {"status": "error", "message": "the gripper angle sensors produced no samples"})

    assert s.start_errors == {LEFT: "the gripper angle sensors produced no samples"}


def test_a_failed_start_with_no_message_still_says_something():
    # str(TimeoutError()) is the empty string — the failure that says the least
    # is the most common one, so _why_not_ok must fill the blank.
    f = fleet(grabette(LEFT, "grabette-01", "left"))
    s = session({"left": LEFT})
    app._schedule_episode_start(OWNER, "cup grasping", s.members, 1.0, session=s)

    _report(f[LEFT], f[LEFT].queue[0].id, {"status": "error", "message": ""})

    assert s.start_errors[LEFT].strip() != ""


def test_a_successful_start_records_nothing():
    f = fleet(grabette(LEFT, "grabette-01", "left"))
    s = session({"left": LEFT})
    app._schedule_episode_start(OWNER, "cup grasping", s.members, 1.0, session=s)

    _report(f[LEFT], f[LEFT].queue[0].id, {"status": "ok", "episode_id": "20250101_120000"})

    assert s.start_errors == {}


def test_a_new_episode_starts_from_a_clean_slate():
    # Otherwise last take's failure haunts every subsequent one, and the warning
    # becomes noise the operator learns to ignore.
    fleet(grabette(LEFT, "grabette-01", "left"))
    s = session({"left": LEFT})
    s.start_errors[LEFT] = "some earlier failure"

    app._schedule_episode_start(OWNER, "cup grasping", s.members, 1.0, session=s)

    assert s.start_errors == {}


def test_the_session_view_names_the_device_not_just_its_id():
    fleet(grabette(LEFT, "grabette-01", "left"))
    s = session({"left": LEFT})
    s.start_errors[LEFT] = "the gripper angle sensors produced no samples"

    errs = app._session_dict(OWNER, s)["start_errors"]

    assert errs == [{"device_id": LEFT, "name": "grabette-01",
                     "error": "the gripper angle sensors produced no samples"}]


def test_a_result_for_an_unknown_command_is_harmless():
    # The handler used to inspect the for-loop variable after the loop: an id it
    # never matched left that variable unbound (empty queue) or, worse, pointing
    # at an unrelated command — so a stale result could have invented a start
    # failure on a session that was recording perfectly well.
    f = fleet(grabette(LEFT, "grabette-01", "left"))
    s = session({"left": LEFT})
    app._schedule_episode_start(OWNER, "cup grasping", s.members, 1.0, session=s)

    _report(f[LEFT], "no-such-command", {"status": "error", "message": "boom"})

    assert s.start_errors == {}
    assert len(f[LEFT].queue) == 1  # the real command is untouched
