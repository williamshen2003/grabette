"""Cross-device episode reconciliation: orphans, split filings, and assignment.

These cover the fleet's only consistency checks between devices, all of which are
pure functions over the in-memory reports — no HTTP, no HF, no devices needed.

Reports are written out by hand here rather than generated from the grabette
package: the wire format IS the contract between the two repos, so spelling it out
keeps this suite independent of the device code (and documents the shape).

Run with:  pip install -r requirements.txt pytest  &&  python -m pytest tests/
"""
import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

OWNER = "operator"
LEFT, RIGHT = "dev-left-1", "dev-right-1"
PAIR = {"left": {"device_id": LEFT, "name": "grabette-left"},
        "right": {"device_id": RIGHT, "name": "grabette-right"}}
MONO_LEFT = {"left": {"device_id": LEFT, "name": "grabette-left"}}
EP = "20260817_140000"


def task_report(name, episode_ids, members=PAIR, signature=("left", "right")):
    """One entry of TaskManager.report_tasks(): episodes grouped by membership."""
    return {"name": name, "description": "", "device_signature": list(signature),
            "groups": [{"members": members, "episode_ids": list(episode_ids)}]}


def grabette(device_id, name, hand, tasks=(), unassigned=None):
    dev = app.Device(device_id, name, ["start_capture"], hand=hand, tasks=list(tasks))
    dev.unassigned = unassigned or {}
    return dev


@pytest.fixture(autouse=True)
def clean_state():
    """Every owner-keyed global, including the memo caches — a stale signature
    leaking between tests would make them pass or fail for the wrong reason."""
    for store in (app.FLEET, app.TASKS, app.SESSIONS, app.GROUPS, app.TASK_SUPPRESSED,
                  app.ORPHANS_PENDING, app.SPLIT_PENDING, app._REPORTED_CACHE,
                  app._REPORTERS_CACHE):
        store.clear()
    yield


def fleet(*devices):
    app.FLEET[OWNER] = {d.device_id: d for d in devices}


def reconcile():
    return app._reconcile_episodes(OWNER)


# ── nothing to report ────────────────────────────────────────────────────

def test_healthy_pair_is_quiet():
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Pick and place", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP])]))
    assert reconcile() == {"orphans": [], "split": []}


def test_a_solo_take_cannot_disagree_with_itself():
    # One reporter, so there is no second opinion to conflict with.
    fleet(grabette(LEFT, "grabette-left", "left",
                   [task_report("Solo work", [EP], MONO_LEFT, ["left"])]),
          grabette(RIGHT, "grabette-right", "right"))
    assert reconcile() == {"orphans": [], "split": []}


def test_an_offline_peer_is_never_used_as_evidence():
    # We cannot know what an offline device still holds, so it neither orphans nor
    # splits anything.
    off = grabette(RIGHT, "grabette-right", "right", [task_report("Other name", [EP])])
    off.last_seen = 0.0
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Pick and place", [EP])]), off)
    assert reconcile() == {"orphans": [], "split": []}


# ── split filings ────────────────────────────────────────────────────────

def test_a_local_refile_on_one_device_is_detected():
    # The reported symptom: "Move to task" on the left grabette's own UI. Orphan
    # detection is blind to it — the left device still reports the episode, just
    # under another name — which is why this check exists.
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP])]))
    snap = reconcile()
    assert snap["orphans"] == []
    assert len(snap["split"]) == 1
    s = snap["split"][0]
    assert s["episode_id"] == EP
    assert s["tasks"] == ["Pick and place", "Sorting"]
    assert {(f["name"], f["task"]) for f in s["filings"]} == {
        ("grabette-left", "Sorting"), ("grabette-right", "Pick and place")}


def test_a_peer_back_with_a_stale_task_name_is_detected():
    # A rename only reaches devices that are online (_devices_reporting_task skips
    # the rest), so a peer that was off during the rename comes back carrying the
    # old name — the same divergence, with nobody having touched a dropdown.
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Pick and place v2", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP])]))
    assert [s["tasks"] for s in reconcile()["split"]] == [["Pick and place", "Pick and place v2"]]


def test_an_episode_being_recorded_is_not_reconciled():
    # Mid-flight episodes are skipped: members register and re-report at different
    # moments, so a recording in progress must produce no noise at all.
    s = app.Session(id="s1", task_id="t1", members={"left": LEFT, "right": RIGHT})
    s.episodes = [{"episode_id": EP}]
    app.SESSIONS[OWNER] = {"s1": s}
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP])]))
    assert reconcile() == {"orphans": [], "split": []}


# ── orphans, and the suppression trap ────────────────────────────────────

def test_a_real_orphan_is_still_detected():
    # The left device genuinely deleted its copy: the episode is absent from its
    # whole report, not merely filed elsewhere.
    fleet(grabette(LEFT, "grabette-left", "left", []),
          grabette(RIGHT, "grabette-right", "right", [task_report("Doomed", [EP])]))
    snap = reconcile()
    assert snap["split"] == []
    assert len(snap["orphans"]) == 1
    o = snap["orphans"][0]
    assert [h["name"] for h in o["holders"]] == ["grabette-right"]
    assert [d["name"] for d in o["deleted_by"]] == ["grabette-left"]


def test_a_normal_delete_stays_quiet_while_the_devices_catch_up():
    # Both still report the task the fleet just deleted; they simply haven't
    # applied the command yet. Flagging that would be the "transient orphans
    # mid-teardown" the name suppression exists to prevent.
    app.TASK_SUPPRESSED[OWNER] = {"Being deleted": time.time() + 20}
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Being deleted", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Being deleted", [EP])]))
    assert reconcile() == {"orphans": [], "split": []}


def test_suppressing_a_diverging_name_does_not_fake_an_orphan():
    # THE regression. Acting on a split — renaming or deleting the odd task —
    # suppresses that name for 20s. If suppression also hid the episode's
    # existence, the left device (whose only copy sat there) would read as having
    # deleted it, and the banner would offer to delete the RIGHT device's copy:
    # the good one. Suppression must hide the NAME, never the episode.
    app.TASK_SUPPRESSED[OWNER] = {"Sorting": time.time() + 20}
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP])]))
    assert reconcile() == {"orphans": [], "split": []}


def test_the_cleanup_button_has_nothing_to_delete_in_that_window(monkeypatch):
    # Same state, one level up: whatever the cached snapshot says, the cleanup
    # endpoint recomputes before acting, so it must dispatch no deletion at all.
    app.TASK_SUPPRESSED[OWNER] = {"Sorting": time.time() + 20}
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP])]))
    sent = []
    monkeypatch.setattr(app, "_enqueue", lambda dev, cmd: sent.append((dev.device_id, cmd.type)))
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    result = asyncio.run(app.cleanup_orphans(app.OrphanCleanupReq(episode_ids=[EP]), None))
    assert result["dispatched"] == 0
    assert sent == []


# ── the banner payload ───────────────────────────────────────────────────

async def _as_owner(_request):
    return OWNER


def test_the_banner_groups_splits_by_the_disagreeing_names(monkeypatch):
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP, "ep2"])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP, "ep2"])]))
    snap = reconcile()
    app.SPLIT_PENDING[OWNER] = snap["split"]
    payload = asyncio.run(app.list_orphans(None))
    # A batch refile shares one divergence, so it collapses into a single card.
    assert len(payload["split"]) == 1
    g = payload["split"][0]
    assert g["count"] == 2 and g["episode_ids"] == [EP, "ep2"]
    # Names for the operator to read, ids for the repair to dispatch to.
    assert g["by_task"]["Sorting"] == [{"device_id": LEFT, "name": "grabette-left"}]
    assert g["by_task"]["Pick and place"] == [{"device_id": RIGHT, "name": "grabette-right"}]


# ── assignment ───────────────────────────────────────────────────────────

def _assign(**kw):
    return asyncio.run(app.assign_episodes(app.AssignReq(**kw), None))


def test_assign_dispatches_one_command_per_device(monkeypatch):
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    sent = []
    monkeypatch.setattr(app, "_enqueue", lambda dev, cmd: sent.append((dev.device_id, cmd.type, cmd.args)))
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP])]))

    _assign(task_name="Pick and place", device_ids=[LEFT, RIGHT], episode_ids=[EP])

    assert [(d, t) for d, t, _ in sent] == [(LEFT, "assign_episodes"), (RIGHT, "assign_episodes")]
    assert sent[0][2] == {"task_name": "Pick and place", "episode_ids": [EP]}


def test_assign_refuses_a_mono_take_into_a_bimanual_task(monkeypatch):
    # Filing a one-handed recording into a two-handed task would leave it counted
    # in the task yet permanently incomplete, and silently dropped from datasets.
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    inbox = {"total": 1, "episodes": [{"episode_id": "20260817_150000", "members": MONO_LEFT,
                                       "duration_seconds": 9.0, "has_video": True}]}
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Bimanual", [EP])], inbox),
          grabette(RIGHT, "grabette-right", "right", [task_report("Bimanual", [EP])]))
    app._reconcile_tasks(OWNER)  # the task exists with signature [left, right]

    with pytest.raises(app.HTTPException) as e:
        _assign(task_name="Bimanual", device_ids=[LEFT], episode_ids=["20260817_150000"])
    assert e.value.status_code == 409
    assert e.value.detail["required"] == ["left", "right"]
    assert e.value.detail["got"] == ["left"]


def test_assign_allows_a_mono_take_into_a_task_with_no_signature(monkeypatch):
    # Nothing is claimed about such a task yet, so filing seeds it — that is what
    # makes a solo recording usable for a dataset at all.
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    sent = []
    monkeypatch.setattr(app, "_enqueue", lambda dev, cmd: sent.append(dev.device_id))
    inbox = {"total": 1, "episodes": [{"episode_id": "20260817_150000", "members": MONO_LEFT,
                                       "duration_seconds": 9.0, "has_video": True}]}
    fleet(grabette(LEFT, "grabette-left", "left", [], inbox))
    app.TASKS[OWNER] = {}
    tid = "t1"
    app.TASKS[OWNER][tid] = app.Task(id=tid, name="Fresh", device_signature=[])

    _assign(task_name="Fresh", device_ids=[LEFT], episode_ids=["20260817_150000"])

    assert sent == [LEFT]


def test_assign_refuses_an_offline_device(monkeypatch):
    # A queued command would be delivered much later, against a state the operator
    # can no longer see.
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    off = grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])])
    off.last_seen = 0.0
    fleet(off)
    with pytest.raises(app.HTTPException) as e:
        _assign(task_name="Sorting", device_ids=[LEFT], episode_ids=[EP])
    assert e.value.status_code == 409


# ── discarding takes ─────────────────────────────────────────────────────

def _del(**kw):
    return asyncio.run(app.delete_episodes(app.EpisodeDeleteReq(**kw), None))


def test_delete_sends_one_batched_command_per_device(monkeypatch):
    # One command carrying every id, not one per episode: the relay worker is
    # serial, so a fifty-take selection would otherwise queue fifty commands.
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    sent = []
    monkeypatch.setattr(app, "_enqueue", lambda d, c: sent.append((d.device_id, c.type, c.args)))
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP, "ep2"])]))

    _del(device_ids=[LEFT], episode_ids=[EP, "ep2"])

    assert sent == [(LEFT, "delete_episode", {"episode_ids": [EP, "ep2"]})]


def test_delete_refuses_an_offline_device(monkeypatch):
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    off = grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])])
    off.last_seen = 0.0
    fleet(off)
    with pytest.raises(app.HTTPException) as e:
        _del(device_ids=[LEFT], episode_ids=[EP])
    assert e.value.status_code == 409


def test_delete_clears_the_banners_for_those_episodes(monkeypatch):
    # Discarding a copy settles both kinds of inconsistency it could be part of, so
    # neither banner should keep pointing at it until the next report.
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    monkeypatch.setattr(app, "_enqueue", lambda d, c: None)
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP])]))
    snap = reconcile()
    app.SPLIT_PENDING[OWNER] = snap["split"]
    app.ORPHANS_PENDING[OWNER] = [{"episode_id": EP, "task": "Sorting", "started_at": 0.0,
                                   "holders": [], "deleted_by": []}]
    assert app.SPLIT_PENDING[OWNER]

    _del(device_ids=[LEFT, RIGHT], episode_ids=[EP])

    assert app.SPLIT_PENDING[OWNER] == []
    assert app.ORPHANS_PENDING[OWNER] == []


# ── dataset plan: no episode goes in half, and nothing is dropped in silence ──

def _bimanual_with_a_half_rig():
    """A task holding one proper pair plus one take the left grabette recorded
    alone — the shape a solo button press or a local refile leaves behind."""
    full = {"members": PAIR, "episode_ids": [EP]}
    half = {"members": MONO_LEFT, "episode_ids": ["20260817_150000"]}
    entry = {"name": "Bimanual", "description": "", "device_signature": ["left", "right"],
             "groups": [full, half]}
    fleet(grabette(LEFT, "grabette-left", "left", [entry]),
          grabette(RIGHT, "grabette-right", "right", [entry]))
    app._reconcile_tasks(OWNER)
    return next(t.id for t in app.TASKS[OWNER].values() if t.name == "Bimanual")


def test_dataset_excludes_a_half_rig_episode_and_says_so():
    tid = _bimanual_with_a_half_rig()

    roles, plan, report = app._resolve_dataset_plan(OWNER, [tid])

    assert roles == ["left", "right"]
    # Only the complete pair is uploaded — the half-rig take would have produced an
    # episode with a left stream and no right one.
    assert sorted(plan[LEFT]["episode_ids"]) == [EP]
    assert sorted(plan[RIGHT]["episode_ids"]) == [EP]
    assert report == {"included": 1, "incomplete": ["20260817_150000"], "unavailable": []}


def test_dataset_refuses_when_every_episode_is_a_half_rig():
    entry = {"name": "Mono only", "description": "", "device_signature": ["left", "right"],
             "groups": [{"members": MONO_LEFT, "episode_ids": [EP]}]}
    fleet(grabette(LEFT, "grabette-left", "left", [entry]),
          grabette(RIGHT, "grabette-right", "right", [entry]))
    app._reconcile_tasks(OWNER)
    tid = next(t.id for t in app.TASKS[OWNER].values() if t.name == "Mono only")

    with pytest.raises(app.HTTPException) as e:
        app._resolve_dataset_plan(OWNER, [tid])
    # Not "no recorded episodes": they exist, they just don't have every role.
    assert e.value.status_code == 409
    assert "was recorded with all of" in e.value.detail["message"]
    assert e.value.detail["incomplete"] == [EP]


def test_dataset_use_only_accepts_the_half_rig_for_a_left_only_build():
    # "Use only [left]" asks for one role, which the half-rig take does have — so
    # narrowing the request is the legitimate way to include it.
    tid = _bimanual_with_a_half_rig()

    roles, plan, report = app._resolve_dataset_plan(OWNER, [tid], roles_override=["left"])

    assert roles == ["left"]
    assert sorted(plan[LEFT]["episode_ids"]) == [EP, "20260817_150000"]
    assert RIGHT not in plan
    assert report["included"] == 2 and report["incomplete"] == []


def test_dataset_only_available_counts_what_it_skipped():
    tid = _bimanual_with_a_half_rig()
    app.FLEET[OWNER][RIGHT].last_seen = 0.0  # right goes offline

    with pytest.raises(app.HTTPException) as e:
        app._resolve_dataset_plan(OWNER, [tid], only_available=True)
    assert e.value.detail["message"] == "no episodes have all their devices online"
    # The pair is unusable because a device is down; the half-rig for a different
    # reason. Both are reported, and not conflated.
    assert e.value.detail["unavailable"] == [EP]
    assert e.value.detail["incomplete"] == ["20260817_150000"]


def test_assign_clears_the_split_from_the_banner_snapshot(monkeypatch):
    # So the card disappears on the next 3s poll instead of lingering until the
    # devices have re-registered.
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    monkeypatch.setattr(app, "_enqueue", lambda dev, cmd: None)
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]),
          grabette(RIGHT, "grabette-right", "right", [task_report("Pick and place", [EP])]))
    app.SPLIT_PENDING[OWNER] = reconcile()["split"]
    assert app.SPLIT_PENDING[OWNER]

    _assign(task_name="Pick and place", device_ids=[LEFT, RIGHT], episode_ids=[EP])

    assert app.SPLIT_PENDING[OWNER] == []


# ── unsafe episode ids ───────────────────────────────────────────────────
# A device turns an id into a path (episodes/<id>) and its deletions rmtree the
# result, so a blank id resolves to the episodes ROOT: dispatched from here it
# lands a phantom entry in a task's registry, and the next "delete task" on that
# device takes every episode with it. Stopped here, at the source of the ids.

def test_assign_drops_ids_that_are_not_episode_ids(monkeypatch):
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    sent = []
    monkeypatch.setattr(app, "_enqueue", lambda d, c: sent.append(c.args))
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]))

    out = _assign(task_name="Sorting", device_ids=[LEFT],
                  episode_ids=["", "  ", ".", "..", "a/b", EP, EP])

    assert sent == [{"task_name": "Sorting", "episode_ids": [EP]}]
    assert out["episodes"] == 1  # what was really dispatched, not what was asked


def test_assign_refuses_a_request_left_with_no_usable_id(monkeypatch):
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]))
    with pytest.raises(app.HTTPException) as e:
        _assign(task_name="Sorting", device_ids=[LEFT], episode_ids=["", " "])
    assert e.value.status_code == 400


def test_delete_drops_ids_that_are_not_episode_ids(monkeypatch):
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    sent = []
    monkeypatch.setattr(app, "_enqueue", lambda d, c: sent.append(c.args))
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]))

    _del(device_ids=[LEFT], episode_ids=["", "../..", EP])

    assert sent == [{"episode_ids": [EP]}]


def test_delete_refuses_a_request_left_with_no_usable_id(monkeypatch):
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    fleet(grabette(LEFT, "grabette-left", "left", [task_report("Sorting", [EP])]))
    with pytest.raises(app.HTTPException) as e:
        _del(device_ids=[LEFT], episode_ids=[""])
    assert e.value.status_code == 400


# ── dataset plan: the operator's per-task episode picking ─────────────────
# "Advanced selection" hands the build an allow-list of episode ids. What it
# leaves out is a CHOICE, so it must not land in the skipped accounting the job
# message reports — that column is for what the fleet dropped on its own.

def _pair_task(name, episode_ids):
    entry = {"name": name, "description": "", "device_signature": ["left", "right"],
             "groups": [{"members": PAIR, "episode_ids": list(episode_ids)}]}
    fleet(grabette(LEFT, "grabette-left", "left", [entry]),
          grabette(RIGHT, "grabette-right", "right", [entry]))
    app._reconcile_tasks(OWNER)
    return next(t.id for t in app.TASKS[OWNER].values() if t.name == name)


def test_dataset_uploads_only_the_picked_episodes():
    tid = _pair_task("Sorting", [EP, "20260817_150000", "20260817_160000"])

    roles, plan, report = app._resolve_dataset_plan(
        OWNER, [tid], only_episodes={EP, "20260817_160000"})

    assert sorted(plan[LEFT]["episode_ids"]) == [EP, "20260817_160000"]
    assert sorted(plan[RIGHT]["episode_ids"]) == [EP, "20260817_160000"]
    # The unpicked take is not "incomplete" or "unavailable" — it was not asked for.
    assert report == {"included": 2, "incomplete": [], "unavailable": []}


def test_dataset_without_a_pick_takes_the_whole_task():
    tid = _pair_task("Sorting", [EP, "20260817_150000"])

    _roles, plan, report = app._resolve_dataset_plan(OWNER, [tid], only_episodes=None)

    assert sorted(plan[LEFT]["episode_ids"]) == [EP, "20260817_150000"]
    assert report["included"] == 2


def test_dataset_refuses_a_pick_that_matches_nothing():
    # A stale id (the episode was deleted since the page rendered) must not build
    # an empty dataset silently.
    tid = _pair_task("Sorting", [EP])
    with pytest.raises(app.HTTPException) as e:
        app._resolve_dataset_plan(OWNER, [tid], only_episodes={"20990101_000000"})
    assert e.value.status_code == 409


def test_dataset_request_carries_the_pick(monkeypatch):
    # End of the wire: the endpoint's episode_ids reach the plan, and an empty
    # list still means "every episode" (what an untouched selection sends).
    monkeypatch.setattr(app, "_operator_loaded", _as_owner)
    seen = {}

    def _spy(owner, task_ids, only_available=False, roles_override=None, only_episodes=None):
        seen["only_episodes"] = only_episodes
        raise app.HTTPException(409, "stop here")  # nothing to dispatch in a unit test

    monkeypatch.setattr(app, "_resolve_dataset_plan", _spy)
    tid = _pair_task("Sorting", [EP])

    for sent, expected in (([EP, "", "a/b"], {EP}), ([], None)):
        with pytest.raises(app.HTTPException):
            asyncio.run(app.create_lerobot_dataset(
                app.DatasetReq(task_ids=[tid], name="ds", episode_ids=sent), None))
        assert seen["only_episodes"] == expected
