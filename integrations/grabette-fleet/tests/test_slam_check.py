"""The SLAM check: which episodes it sends, and how the Space's report is read.

Two things can go quietly wrong here, and both would be worse than no check at all:

  * sending the WRONG episodes — the in-progress take (not muxed yet) or one whose
    devices no longer hold all its roles — makes the check flag a take the operator
    never recorded that way;
  * reading the report loosely — merging a pre-check warning and a trajectory
    verdict into one confident "GOOD", or letting a MISSING report look like a
    clean one — tells the operator to keep recording when nobody knows if SLAM
    held.

Everything below is in-memory: no HTTP, no HF, no devices. The Space's report
shape is written out by hand because it IS the contract between the two repos.

Run with:  pip install -r requirements.txt pytest  &&  python -m pytest tests/
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

OWNER = "operator"
LEFT, RIGHT = "dev-left-1", "dev-right-1"
TASK_ID, TASK_NAME = "task-1", "Pick and place"
PAIR = {"left": {"device_id": LEFT, "name": "grabette-left"},
        "right": {"device_id": RIGHT, "name": "grabette-right"}}


@pytest.fixture(autouse=True)
def clean_state():
    for store in (app.FLEET, app.TASKS, app.SESSIONS, app.GROUPS, app.TASK_SUPPRESSED,
                  app.ORPHANS_PENDING, app.SPLIT_PENDING, app._REPORTED_CACHE,
                  app._REPORTERS_CACHE, app.DATASET_JOBS):
        store.clear()
    yield


def task_report(episode_ids, members=PAIR, signature=("left", "right")):
    """One entry of the device's report_tasks(): episodes grouped by membership."""
    return {"name": TASK_NAME, "description": "", "device_signature": list(signature),
            "groups": [{"members": members, "episode_ids": list(episode_ids)}]}


def setup_fleet(left_eps, right_eps):
    """A left/right pair, each reporting the episodes it actually holds."""
    left = app.Device(LEFT, "grabette-left", ["start_capture"], hand="left",
                      tasks=[task_report(left_eps)] if left_eps else [])
    right = app.Device(RIGHT, "grabette-right", ["start_capture"], hand="right",
                       tasks=[task_report(right_eps)] if right_eps else [])
    app.FLEET[OWNER] = {LEFT: left, RIGHT: right}
    app.TASKS[OWNER] = {TASK_ID: app.Task(TASK_ID, TASK_NAME,
                                          device_signature=["left", "right"])}
    return left, right


def session(episode_ids, *, recording=False):
    s = app.Session(id="sess-1", task_id=TASK_ID, members={"left": LEFT, "right": RIGHT})
    s.recording = recording
    s.episodes = [{"episode_id": e, "roles": dict(s.members)} for e in episode_ids]
    app.SESSIONS[OWNER] = {s.id: s}
    return s


def plan(s, count):
    return app._slam_check_plan(OWNER, s, count)


# ── which episodes go up ─────────────────────────────────────────────────

EPS = ["20260826_100000", "20260826_100500", "20260826_101000", "20260826_101500"]


def test_takes_the_last_n_in_recording_order():
    setup_fleet(EPS, EPS)
    roles, dev_plan, episode_ids, _skipped = plan(session(EPS), 3)
    # Newest three, but reported oldest-first: the report is read top-down while
    # the operator thinks in "the takes I just did".
    assert episode_ids == EPS[1:]
    assert roles == ["left", "right"]
    assert dev_plan[LEFT] == {"role": "left", "episode_ids": set(EPS[1:])}
    assert dev_plan[RIGHT] == {"role": "right", "episode_ids": set(EPS[1:])}


def test_asking_for_more_than_exists_takes_what_there_is():
    setup_fleet(EPS[:2], EPS[:2])
    _roles, _plan, episode_ids, _skipped = plan(session(EPS[:2]), 5)
    assert episode_ids == EPS[:2]


def test_the_in_progress_take_is_never_checked():
    # While recording, the last manifest entry is the take being recorded right
    # now: its data isn't muxed on the device, so SLAM would judge a fragment.
    setup_fleet(EPS, EPS)
    _roles, _plan, episode_ids, _skipped = plan(session(EPS, recording=True), 2)
    assert episode_ids == EPS[1:3]


def test_a_take_the_device_has_not_reported_yet_is_still_checked():
    # The regression that made a check untrustworthy: the fleet learns a device's
    # episode list only when it re-registers, a heartbeat after the mux. Right
    # after a stop the manifest has the take and the report does not — and reading
    # that absence as "the device hasn't got it" checked the three takes recorded
    # BEFORE the one the operator had just finished.
    setup_fleet(EPS[:-1], EPS[:-1])                      # reports one take short
    _roles, _plan, episode_ids, _skipped = plan(session(EPS), 3)   # manifest has all four
    assert episode_ids == EPS[1:]                        # newest one included


def test_a_deleted_episode_is_skipped_not_half_checked():
    # Here the absence IS meaningful: the right grabette reports a NEWER take
    # without this one, so it really is gone (deleted, or refiled). Checking it
    # left-only would flag a bimanual take nobody recorded that way.
    setup_fleet(EPS, [EPS[0], EPS[1], EPS[3]])
    _roles, _plan, episode_ids, _skipped = plan(session(EPS), 3)
    assert episode_ids == [EPS[0], EPS[1], EPS[3]]


def test_a_take_recorded_without_every_role_is_skipped():
    setup_fleet(EPS, EPS)
    s = session(EPS)
    s.episodes[-1]["roles"] = {"left": LEFT}  # the right grabette wasn't in that take
    _roles, _plan, episode_ids, _skipped = plan(s, 2)
    assert episode_ids == EPS[1:3]


def test_an_offline_device_takes_its_episodes_out_of_reach():
    # Reports only count from online devices, so an offline peer's episodes are
    # not "complete" — the check falls back to nothing rather than a half-set.
    left, right = setup_fleet(EPS, EPS)
    right.last_seen = 0.0
    with pytest.raises(app.HTTPException) as e:
        plan(session(EPS), 3)
    assert e.value.status_code == 409


def test_a_take_the_report_contradicts_is_named_in_skipped():
    # Positively gone (the right grabette reports a NEWER take without it), so the
    # plan passes it over — and says which one, because an omission the operator
    # can't see is what makes a check untrustworthy.
    setup_fleet(EPS, [EPS[0], EPS[1], EPS[3]])
    _roles, _plan, episode_ids, skipped = plan(session(EPS), 3)
    assert episode_ids == [EPS[0], EPS[1], EPS[3]]
    assert skipped == [EPS[2]]


def test_the_device_has_the_last_word_on_a_missing_take():
    # A manifest entry is appended when a start is DISPATCHED, so a start that never
    # produced a recording leaves one behind — indistinguishable, from the reports
    # alone, from a take not reported yet. So the plan includes it (that tolerance is
    # what stopped checks running on stale takes) and the DEVICE settles it at upload
    # time by answering "missing", which the check then names.
    job = app.DatasetJob(id="j", task_ids=[TASK_ID], roles=["left"], raw_repo="r",
                         target_repo="o/t", check=True, episode_ids=EPS[:3])
    assert app._slam_check_note(job, 3, 3) == ""          # nothing to explain
    job.missing_episodes = [EPS[2]]
    note = app._slam_check_note(job, 3, 2)
    assert "1 take(s) had no data on their device" in note and EPS[2] in note
    # Without that answer, a count that doesn't add up is still worth saying.
    job.missing_episodes = []
    assert "(2 of 3 came back.)" in app._slam_check_note(job, 3, 2)


def test_nothing_recorded_is_refused_not_an_empty_run():
    setup_fleet([], [])
    with pytest.raises(app.HTTPException) as e:
        plan(session([]), 3)
    assert e.value.status_code == 409


def test_the_repo_name_carries_the_check_flag_to_the_space():
    # The marker is not cosmetic: it is HOW the Space is told to run a check rather
    # than a build (the device rebuilds the payload from fixed keys).
    repo = app._slam_check_repo(OWNER, "Pick & place / v2", "20260826_153012")
    assert repo == f"{OWNER}/pick-place-v2_trajectorycheck_20260826_153012"
    assert app.SLAM_CHECK_MARKER in repo
    # A name made only of separators still yields a usable repo id.
    assert app._slam_check_repo(OWNER, "///", "20260826_153012") == \
        f"{OWNER}/task_trajectorycheck_20260826_153012"
    # A long task name is trimmed, and the marker survives it.
    long_repo = app._slam_check_repo(OWNER, "a" * 60, "20260826_153012")
    assert long_repo == f"{OWNER}/{'a' * 32}_trajectorycheck_20260826_153012"


# ── reading the Space's report ───────────────────────────────────────────

def traj(name, verdict, tracking, lost, frames, jumps=0, warnings=()):
    return {"name": name, "kind": "trajectory", "verdict": verdict,
            "tracking_pct": tracking, "n_lost": lost, "n_frames": frames,
            "n_tracked": frames - lost, "n_jumps": jumps,
            "errors": [], "warnings": list(warnings), "excluded": False}


def test_one_row_per_episode_judged_by_its_trajectory():
    # The Space reports per CHECK, not per episode: a pre-check warning and a
    # trajectory verdict for the same take are two entries. The verdict shown is
    # the SLAM one — a recording warning is a note, and must not turn a take that
    # tracked perfectly into a WARN (which is what made every row read WARN).
    rows = app._summarize_slam_quality([
        {"name": "ep-1", "kind": "pre_check", "verdict": "WARN", "errors": [],
         "warnings": ["angle: distal joint appears static"], "excluded": False},
        traj("ep-1", "GOOD", 99.8, 2, 900),
    ])
    assert len(rows) == 1
    assert rows[0]["verdict"] == "GOOD"          # the trajectory decides
    assert rows[0]["tracking_pct"] == 99.8       # figures survive the merge
    assert rows[0]["n_lost"] == 2 and rows[0]["n_frames"] == 900


def test_a_pre_check_error_is_the_outcome():
    # No trajectory at all: the take never reached SLAM, so the pre-check error
    # IS the result — that one is not a note.
    rows = app._summarize_slam_quality([
        {"name": "ep-1", "kind": "pre_check", "verdict": "ERROR",
         "errors": ["no IMU data"], "warnings": [], "excluded": True}])
    assert rows[0]["verdict"] == "ERROR"
    assert rows[0]["tracking_pct"] is None  # it never reached SLAM, so no metrics


def test_a_take_not_judged_yet_is_neither_good_nor_warned():
    # Mid-run: the report streams, so a take can be pre-checked before it is
    # SLAM'd. An empty verdict is the UI's "…" — inventing GOOD here would be a
    # clean answer about work that hasn't happened.
    rows = app._summarize_slam_quality([
        {"name": "ep-1", "kind": "pre_check", "verdict": "WARN", "errors": [],
         "warnings": ["angle: proximal joint appears static"], "excluded": False}])
    assert rows[0]["verdict"] == ""


def test_rows_come_back_in_recording_order():
    rows = app._summarize_slam_quality([traj("20260826_101000", "GOOD", 100.0, 0, 10),
                                        traj("20260826_100000", "BAD", 40.0, 60, 100)])
    assert [r["episode"] for r in rows] == ["20260826_100000", "20260826_101000"]


def test_an_unknown_verdict_is_not_reported_as_clean():
    rows = app._summarize_slam_quality([traj("ep-1", "SOMETHING_NEW", 50.0, 1, 2)])
    assert rows[0]["verdict"] == "SOMETHING_NEW"  # surfaced as-is…
    # …and it outranks GOOD, so a verdict this fleet doesn't know can never be
    # swallowed by a GOOD entry for the same episode.
    rows = app._summarize_slam_quality([traj("ep-1", "GOOD", 100.0, 0, 2),
                                        traj("ep-1", "SOMETHING_NEW", 50.0, 1, 2)])
    assert rows[0]["verdict"] == "SOMETHING_NEW"


def test_a_row_is_the_metrics_and_nothing_else():
    # No prose per episode: the figures are what say what happened, and every
    # message stays in the Space's own report.
    rows = app._summarize_slam_quality([
        traj("ep-1", "BAD", 61.0, 351, 900, jumps=4, warnings=["lost tracking for 3.5s"])])
    assert rows[0] == {"episode": "ep-1", "verdict": "BAD", "tracking_pct": 61.0,
                       "n_lost": 351, "n_frames": 900, "n_jumps": 4}


def test_a_junk_entry_does_not_take_the_report_down():
    rows = app._summarize_slam_quality(["nonsense", None, traj("ep-1", "GOOD", 100.0, 0, 5)])
    assert [r["episode"] for r in rows] == ["ep-1"]


# ── the verdict the operator actually reads ──────────────────────────────

def check_job(n_episodes=3):
    return app.DatasetJob(id="job1", task_ids=[TASK_ID], roles=["left", "right"],
                          raw_repo="o/slamcheck-x-raw", target_repo="o/slamcheck-x",
                          check=True, episode_ids=EPS[:n_episodes])


def apply_report(monkeypatch, job, report, final=True):
    async def fake(_repo):
        return report
    monkeypatch.setattr(app, "_slam_quality", fake)
    asyncio.run(app._apply_slam_quality(job, final=final))
    return job


def test_a_clean_check_says_recording_can_continue(monkeypatch):
    job = apply_report(monkeypatch, check_job(), {
        "status": "done", "flagged": [], "result": None, "visualizer": None,
        "quality": [traj(e, "GOOD", 100.0, 0, 900) for e in EPS[:3]]})
    assert job.flagged == []
    assert "No tracking loss" in job.message
    # A clean check pushes nothing, so there is nothing to link to.
    assert job.result_url is None and job.visualizer_url is None


def test_a_flagged_check_points_at_the_visualizer(monkeypatch):
    job = apply_report(monkeypatch, check_job(), {
        "status": "done", "flagged": [EPS[1]],
        "result": "https://huggingface.co/datasets/o/slamcheck-x",
        "visualizer": "https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=o/slamcheck-x&episode=0",
        "quality": [traj(EPS[0], "GOOD", 100.0, 0, 900),
                    traj(EPS[1], "BAD", 61.0, 351, 900),
                    traj(EPS[2], "GOOD", 99.9, 1, 900)]})
    assert job.flagged == [EPS[1]]
    assert "1 of 3 episode(s) flagged" in job.message
    assert "visualizer" in job.message
    assert job.visualizer_url and job.result_url


def test_a_flagged_check_that_could_not_push_says_so(monkeypatch):
    # Every take failed the pre-check → no trajectory, so nothing could be built.
    job = apply_report(monkeypatch, check_job(1), {
        "status": "done", "flagged": [EPS[0]], "result": None, "visualizer": None,
        "quality": [{"name": EPS[0], "kind": "pre_check", "verdict": "ERROR",
                     "errors": ["no IMU data"], "warnings": [], "excluded": True}]})
    assert "Nothing could be pushed" in job.message
    assert job.quality[0]["verdict"] == "ERROR"


def test_a_recording_warning_stays_out_of_a_clean_check(monkeypatch):
    # Tracking held on every take; one carries a pre-check warning that says
    # nothing about SLAM. It pushes nothing, it is not the verdict, and it is not
    # shown — three GOOD rows is the whole answer.
    job = apply_report(monkeypatch, check_job(), {
        "status": "done", "flagged": [], "result": None, "visualizer": None,
        "quality": [{"name": EPS[0], "kind": "pre_check", "verdict": "WARN", "errors": [],
                     "warnings": ["angle: distal joint appears static"], "excluded": False},
                    traj(EPS[0], "GOOD", 100.0, 0, 900),
                    traj(EPS[1], "GOOD", 100.0, 0, 900),
                    traj(EPS[2], "GOOD", 100.0, 0, 900)]})
    assert job.flagged == []
    assert [r["verdict"] for r in job.quality] == ["GOOD", "GOOD", "GOOD"]
    assert "No tracking loss" in job.message
    assert job.result_url is None


def test_the_count_is_episodes_not_findings(monkeypatch):
    # The Space's `flagged` is a list of ENTRIES: four takes each carrying a
    # recording warning plus one bad trajectory made it five items long, and a
    # 4-episode check announced "5 of 4 episode(s) flagged".
    warn = [{"name": e, "kind": "pre_check", "verdict": "WARN", "errors": [],
             "warnings": ["angle: distal joint appears static"], "excluded": False}
            for e in EPS]
    job = apply_report(monkeypatch, check_job(4), {
        "status": "done",
        "flagged": [e for e in EPS] + [EPS[2]],          # 5 findings, 4 episodes
        "result": "https://huggingface.co/datasets/o/slamcheck-x", "visualizer": "viz",
        "quality": warn + [traj(EPS[0], "GOOD", 100.0, 0, 900),
                           traj(EPS[1], "GOOD", 100.0, 0, 900),
                           traj(EPS[2], "BAD", 61.0, 351, 900),
                           traj(EPS[3], "GOOD", 100.0, 0, 900)]})
    assert job.flagged == [EPS[2]]
    assert "1 of 4 episode(s) flagged" in job.message


def test_an_episode_that_never_came_back_is_not_covered_by_all_clear(monkeypatch):
    job = apply_report(monkeypatch, check_job(3), {
        "status": "done", "result": None, "visualizer": None,
        "quality": [traj(EPS[0], "GOOD", 100.0, 0, 900),
                    traj(EPS[1], "GOOD", 100.0, 0, 900)]})
    assert job.flagged == []
    assert "No tracking loss on the 2 episode(s)" in job.message
    assert "(2 of 3 came back.)" in job.message


def test_a_dataset_pushed_over_a_clean_result_is_explained(monkeypatch):
    # The deployed Space still pushes on a recording warning. The link would
    # otherwise sit under an "all clear" line with nothing to explain it.
    job = apply_report(monkeypatch, check_job(1), {
        "status": "done", "result": "https://huggingface.co/datasets/o/slamcheck-x",
        "visualizer": "viz", "quality": [traj(EPS[0], "GOOD", 100.0, 0, 900)]})
    assert job.flagged == []
    assert "No tracking loss" in job.message
    assert "The tested set was pushed" in job.message


def test_a_missing_report_is_not_a_pass(monkeypatch):
    # The Space keeps its job list in memory: a restart loses the report. The
    # check must say that, not imply the episodes came out clean.
    job = apply_report(monkeypatch, check_job(), None)
    assert job.flagged is None
    assert "could not be read back" in job.message
    job = apply_report(monkeypatch, check_job(), {"status": "not_found"})
    assert job.flagged is None
    assert "could not be read back" in job.message


def test_a_report_with_no_verdicts_is_not_a_pass(monkeypatch):
    job = apply_report(monkeypatch, check_job(), {"status": "done", "flagged": None,
                                                 "quality": [], "result": None})
    assert job.flagged is None
    assert "no verdict" in job.message


def test_a_live_tick_shows_what_the_space_is_doing(monkeypatch):
    job = apply_report(monkeypatch, check_job(), {
        "status": "running", "flagged": None, "quality": [],
        "log_tail": "Running SLAM on 20260826_100500…"}, final=False)
    assert job.message.startswith("Checking SLAM on 3 episode(s)…")
    assert "Running SLAM" in job.message
    assert job.flagged is None  # no verdict yet, and none invented


def test_verdicts_are_read_even_if_the_space_names_no_summary(monkeypatch):
    # An older Space reports per-episode verdicts but no `flagged` list. Reporting
    # "no answer" over a table full of answers would be the wrong reading.
    job = apply_report(monkeypatch, check_job(2), {
        "status": "done", "result": None, "visualizer": None,
        "quality": [traj(EPS[0], "GOOD", 100.0, 0, 900),
                    traj(EPS[1], "FAIL", 12.0, 790, 900)]})
    assert job.flagged == [EPS[1]]
    assert "1 of 2 episode(s) flagged" in job.message


# ── two grabettes: one take is two rows ──────────────────────────────────

def bimanual(eps, bad_arm=None):
    """The Space runs SLAM per ARM, so a bimanual raw yields "{episode}/left" and
    "{episode}/right" — two entries for one take."""
    out = []
    for e in eps:
        for role in ("left", "right"):
            v = "BAD" if bad_arm == f"{e}/{role}" else "GOOD"
            out.append(traj(f"{e}/{role}", v, 61.0 if v == "BAD" else 100.0,
                            351 if v == "BAD" else 0, 900))
    return out


def test_a_bimanual_take_is_one_episode_in_the_counts(monkeypatch):
    job = apply_report(monkeypatch, check_job(3), {
        "status": "done", "result": "https://huggingface.co/datasets/o/x",
        "visualizer": "viz", "quality": bimanual(EPS[:3], bad_arm=f"{EPS[1]}/left")})
    assert len(job.quality) == 6            # the table keeps both arms…
    assert job.flagged == [EPS[1]]          # …the count is takes, not arms
    assert "1 of 3 episode(s) flagged" in job.message
    assert f"{EPS[1]}/left" not in job.message  # the take is named, not the arm


def test_both_arms_of_one_take_flag_it_once(monkeypatch):
    rows = bimanual(EPS[:3])
    for r in rows:
        if r["name"].startswith(EPS[2]):
            r["verdict"], r["tracking_pct"], r["n_lost"] = "BAD", 40.0, 540
    job = apply_report(monkeypatch, check_job(3), {
        "status": "done", "result": None, "visualizer": None, "quality": rows})
    assert job.flagged == [EPS[2]]
    assert "1 of 3 episode(s) flagged" in job.message


def test_a_clean_bimanual_check_does_not_count_six_of_three(monkeypatch):
    job = apply_report(monkeypatch, check_job(3), {
        "status": "done", "result": None, "visualizer": None,
        "quality": bimanual(EPS[:3])})
    assert job.flagged == []
    assert "No tracking loss on the 3 episode(s) checked" in job.message
    assert "came back" not in job.message
