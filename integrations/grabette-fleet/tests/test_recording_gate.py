"""What may start a recording, and — the part that broke — what may not.

A trajectory check is a deliberate pause between takes: the operator is told
"recording paused until it finishes", and every start path has to hold that
promise. There are three of them (fleet launch, fleet record button, and the
device's own physical button via sync/start), and the physical one used to walk
straight past the gate — the fleet scheduled the group episode anyway, the busy
peer refused its own start_capture locally, and the take came out half-rig.

The gate is also WIDER than "is this device uploading right now": a live job
leaves gaps where nobody reports busy — every member that is not the one
converting, and the window between the last upload's result and the processing
command reaching its device. Recording in one of those gaps is precisely the
episode the operator was told they could not take.

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
TASK_ID, TASK_NAME = "task-1", "Pick and place"


@pytest.fixture(autouse=True)
def clean_state():
    for store in (app.FLEET, app.TASKS, app.SESSIONS, app.GROUPS, app.TASK_SUPPRESSED,
                  app.ORPHANS_PENDING, app.SPLIT_PENDING, app._REPORTED_CACHE,
                  app._REPORTERS_CACHE, app.DATASET_JOBS):
        store.clear()
    yield


def setup_session(*, left_status="idle", right_status="idle"):
    left = app.Device(LEFT, "grabette-left", ["start_capture"], hand="left")
    right = app.Device(RIGHT, "grabette-right", ["start_capture"], hand="right")
    left.reported_status, right.reported_status = left_status, right_status
    app.FLEET[OWNER] = {LEFT: left, RIGHT: right}
    app.TASKS[OWNER] = {TASK_ID: app.Task(TASK_ID, TASK_NAME,
                                          device_signature=["left", "right"])}
    s = app.Session(id="sess-1", task_id=TASK_ID, members={"left": LEFT, "right": RIGHT})
    app.SESSIONS[OWNER] = {s.id: s}
    return s


def live_check(*, status="processing", uploaders=(LEFT, RIGHT), processor=RIGHT):
    """A trajectory check in flight, as start_slam_check leaves it."""
    job = app.DatasetJob(id="job-1", task_ids=[TASK_ID], roles=["left", "right"],
                         raw_repo=f"{OWNER}/check-raw", target_repo=f"{OWNER}/check",
                         status=status, check=True, processor=processor,
                         proc_cmd="cmd-proc" if processor else None,
                         upload_cmds={d: f"cmd-{d}" for d in uploaders})
    app.DATASET_JOBS[OWNER] = {job.id: job}
    return job


def press_button(device_id=LEFT):
    """The physical button's only route to a group start."""
    return asyncio.run(app.device_sync_start(device_id, auth=(OWNER, "tok")))


def test_the_physical_button_is_refused_while_a_check_runs():
    setup_session(right_status="processing")
    live_check()
    with pytest.raises(HTTPException) as e:
        press_button()
    assert e.value.status_code == 409
    assert "trajectory check" in e.value.detail["message"]


def test_the_button_is_refused_on_a_device_the_check_left_idle():
    # The pressed device finished its upload; only its peer is converting. Its own
    # local gate reads free, so the fleet is the only thing standing between the
    # press and a half-rig episode.
    setup_session(left_status="idle", right_status="processing")
    live_check()
    with pytest.raises(HTTPException) as e:
        press_button(LEFT)
    assert e.value.status_code == 409


def test_the_button_is_refused_in_the_gap_where_nobody_reports_busy():
    # Uploads done, the processing command not yet picked up: every device reports
    # idle and every queue is empty. The job is what says the session is held.
    setup_session(left_status="idle", right_status="idle")
    live_check(status="raw_ready", processor="")
    with pytest.raises(HTTPException) as e:
        press_button()
    assert e.value.status_code == 409


def test_a_finished_check_releases_the_button():
    setup_session()
    live_check(status="done")
    assert press_button()["status"] == "scheduled"


def test_a_cancelled_check_releases_the_button():
    setup_session()
    live_check().cancelled = True
    assert press_button()["status"] == "scheduled"


def test_the_button_still_starts_a_group_episode_when_nothing_is_running():
    s = setup_session()
    out = press_button()
    assert out["status"] == "scheduled" and out["peers"] == [RIGHT]
    assert s.recording and len(s.episodes) == 1


def test_a_refused_press_records_nothing():
    s = setup_session(right_status="processing")
    live_check()
    with pytest.raises(HTTPException):
        press_button()
    assert not s.recording and not s.episodes
    # And no peer was told to start: a refusal that still dispatched would leave
    # the other hand recording alone.
    assert not app.FLEET[OWNER][RIGHT].queue


def test_the_button_and_the_fleet_button_block_on_the_same_thing():
    s = setup_session(right_status="processing")
    live_check()
    fleet_side = app._recording_blockers(OWNER, s.members.values())
    with pytest.raises(HTTPException) as e:
        press_button()
    assert sorted(e.value.detail["devices"]) == sorted(fleet_side) == [LEFT, RIGHT]


def test_a_device_in_no_session_is_solo_not_refused():
    setup_session()
    app.SESSIONS[OWNER].clear()
    live_check()
    assert press_button()["status"] == "solo"
