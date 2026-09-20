"""Episodes left out of a dataset build must be counted and named.

Reported from use: a task where oakd_calib_offline.json was missing on a FEW
episodes produced a build that said "Dataset ready." and nothing else. No count,
no names — the operator had no way to know their dataset was assembled from 17 of
20 takes, let alone which 3 were gone or why.

The information existed at every step and was discarded at the fleet boundary:
  * the device screens out unconvertible episodes before uploading and returns
    them in `incomplete` — the fleet read the result's status and dropped the rest;
  * the conversion Space returns per-episode quality with the actual reasons —
    the device now forwards them, and the fleet used to overwrite the message with
    "Dataset ready." regardless.

So: one ledger on the job, fed by every stage that can drop an episode, deduped
because the stages cascade, and surfaced in the result line.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

OWNER = "operator"
LEFT, RIGHT = "dev-left-1", "dev-right-1"


@pytest.fixture(autouse=True)
def clean_state():
    for store in (app.FLEET, app.TASKS, app.SESSIONS, app.GROUPS, app.DATASET_JOBS,
                  app.TASK_SUPPRESSED, app.ORPHANS_PENDING, app.SPLIT_PENDING,
                  app._REPORTED_CACHE, app._REPORTERS_CACHE):
        store.clear()
    yield


def job(**kw):
    return app.DatasetJob(id="j1", task_ids=["t1"], roles=["left", "right"],
                          raw_repo="me/ds-raw", target_repo="me/ds", **kw)


# --- the ledger ---------------------------------------------------------------

def test_an_empty_ledger_says_nothing():
    assert app._excluded_summary(job()) == ""


def test_the_summary_leads_with_the_count():
    # The count is the number that was missing entirely; it comes first.
    j = job()
    app._note_excluded(j, [
        {"episode_id": "ep1", "reason": "missing oakd_calib_offline.json"},
        {"episode_id": "ep2", "reason": "missing oakd_calib_offline.json"},
        {"episode_id": "ep3", "reason": "missing oakd_calib_offline.json"},
    ])

    assert app._excluded_summary(j) == (
        "3 episode(s) left out: missing oakd_calib_offline.json (3)")


def test_reasons_are_counted_not_listed():
    j = job()
    app._note_excluded(j, [{"episode_id": f"ep{i}", "reason": "no trajectory"}
                           for i in range(20)])

    summary = app._excluded_summary(j)

    assert summary.startswith("20 episode(s) left out: no trajectory (20)")
    assert summary.count("no trajectory") == 1


def test_many_distinct_reasons_are_capped_but_accounted_for():
    # Truncating silently would be the same class of bug all over again.
    j = job()
    app._note_excluded(j, [{"episode_id": f"ep{i}", "reason": f"reason {i}"}
                           for i in range(6)])

    summary = app._excluded_summary(j)

    assert "+3 other reason(s)" in summary
    assert summary.startswith("6 episode(s) left out")


def test_the_same_episode_is_never_counted_twice():
    # The stages cascade: an arm screened out at upload makes the Space drop the
    # whole recording for a missing arm. Counting both would misreport the damage.
    j = job()
    app._note_excluded(j, [{"episode_id": "ep1", "role": "left",
                            "reason": "missing oakd_calib_offline.json"}])
    app._note_excluded(j, [{"episode_id": "ep1", "role": "left",
                            "reason": "excluded by the conversion"}])

    assert len(j.excluded) == 1
    # And the FIRST reason survives — the upload knows which file is missing, the
    # Space only knows the arm never arrived.
    assert j.excluded[0]["reason"] == "missing oakd_calib_offline.json"
    assert app._excluded_summary(j).startswith("1 episode(s)")


def test_the_two_arms_of_one_recording_are_distinct_entries():
    j = job()
    app._note_excluded(j, [
        {"episode_id": "ep1", "role": "left", "reason": "missing angle_data.json"},
        {"episode_id": "ep1", "role": "right", "reason": "missing angle_data.json"},
    ])

    assert len(j.excluded) == 2
    # ...but it is still ONE lost recording, and that is what the count reports.
    assert app._excluded_summary(j).startswith("1 episode(s)")


def test_entries_without_an_episode_id_are_dropped():
    j = job()
    app._note_excluded(j, [{"reason": "orphan"}, {"episode_id": "", "reason": "x"}])

    assert j.excluded == []


def test_a_missing_reason_still_produces_an_entry():
    j = job()
    app._note_excluded(j, [{"episode_id": "ep1"}])

    assert j.excluded[0]["reason"] == "excluded"


# --- the build reports it -----------------------------------------------------

def _run(job_obj, *, upload_results, proc_result):
    """Drive _run_dataset_job with canned device results, no waiting."""
    fleet = {}
    for dev_id, res in upload_results.items():
        dev = app.Device(dev_id, dev_id, ["upload_episodes", "process_dataset"])
        cmd = app.Command(id=f"cmd-{dev_id}", type="upload_episodes", args={})
        cmd.status, cmd.result = "done", res
        dev.history.insert(0, cmd)
        job_obj.upload_cmds[dev_id] = cmd.id
        fleet[dev_id] = dev
    app.FLEET[OWNER] = fleet

    processor = next(iter(upload_results))
    real_enqueue = app._enqueue

    def _instant_enqueue(dev, cmd):
        # The processing command completes the moment it is dispatched.
        real_enqueue(dev, cmd)
        cmd.status, cmd.result = "done", proc_result
        dev.queue.remove(cmd)
        dev.history.insert(0, cmd)

    # The runner polls on 2s/3s sleeps; every result here is already in place,
    # so waiting them out would only make the suite slow.
    real_sleep = asyncio.sleep

    async def _no_sleep(_delay, *a, **kw):
        return await real_sleep(0)

    app._enqueue = _instant_enqueue
    asyncio.sleep = _no_sleep
    try:
        asyncio.run(app._run_dataset_job(OWNER, job_obj, False, processor))
    finally:
        app._enqueue = real_enqueue
        asyncio.sleep = real_sleep
    return job_obj


def test_a_partial_build_records_what_was_left_out():
    # THE regression: an operator whose task had a few bad episodes had no way to
    # know their dataset was assembled from a subset. The ledger is what carries
    # it — the result line stays terse and the panel below reports it, so the
    # count is not read twice on the way to the link.
    j = job()

    _run(j,
         upload_results={LEFT: {"status": "ok", "uploaded": ["ep1", "ep2"],
                                "incomplete": [{"episode_id": "ep3",
                                                "missing": ["oakd_calib_offline.json"]}]}},
         proc_result={"status": "ok", "result_url": "https://hf.co/datasets/me/ds"})

    assert j.status == "done"
    assert j.message == "Dataset ready."  # not repeated here
    assert app._excluded_summary(j).startswith("1 episode(s) left out")
    assert j.excluded[0]["episode_id"] == "ep3"
    assert j.excluded[0]["reason"] == "missing oakd_calib_offline.json"
    assert j.excluded[0]["stage"] == "upload"
    assert j.excluded[0]["device"] == LEFT  # which grabette to go and look at


def test_a_complete_build_stays_terse():
    # No exclusions, no noise: the report must not cry wolf on a clean build.
    j = job()

    _run(j, upload_results={LEFT: {"status": "ok", "uploaded": ["ep1"]}},
         proc_result={"status": "ok", "result_url": "https://hf.co/datasets/me/ds"})

    assert j.message == "Dataset ready."
    assert j.excluded == []


def test_exclusions_from_the_conversion_are_kept_too():
    j = job()

    _run(j, upload_results={LEFT: {"status": "ok", "uploaded": ["ep1", "ep2"]}},
         proc_result={"status": "ok", "result_url": "https://hf.co/datasets/me/ds",
                      "excluded": [{"episode_id": "ep2", "role": "left",
                                    "reason": "SLAM failed: no trajectory produced"}]})

    assert app._excluded_summary(j).startswith("1 episode(s) left out")
    assert j.excluded[0]["stage"] == "conversion"


def test_a_failed_upload_still_names_what_it_screened_out():
    # The error alone doesn't say which takes are unusable, and that is usually
    # the actionable half.
    j = job()

    _run(j,
         upload_results={LEFT: {"status": "error", "message": "network unreachable",
                                "incomplete": [{"episode_id": "ep9",
                                                "missing": ["angle_data.json"]}]}},
         proc_result={"status": "ok"})

    assert j.status == "error"
    assert j.message.startswith("Failed:")
    assert j.excluded[0]["episode_id"] == "ep9"
    assert "angle_data.json" in j.excluded[0]["reason"]


def _status(j):
    app.DATASET_JOBS[OWNER] = {j.id: j}
    orig = app._operator_loaded
    app._operator_loaded = lambda request: _async(OWNER)
    try:
        return asyncio.run(app.lerobot_dataset_status(j.id, object()))
    finally:
        app._operator_loaded = orig


def test_the_status_endpoint_exposes_the_ledger_when_the_build_is_over():
    j = job(status="done")
    app._note_excluded(j, [{"episode_id": "ep3", "role": "left",
                            "reason": "missing oakd_calib_offline.json"}])

    out = _status(j)

    assert out["excluded_summary"].startswith("1 episode(s) left out")
    assert out["excluded"][0]["episode_id"] == "ep3"


def test_a_running_build_sends_the_summary_but_not_the_whole_list():
    # This endpoint is polled every 2s for the entire build. The list does not
    # change between polls and the UI only renders it once the build is terminal,
    # so shipping it early was several kilobytes a second down the wire for a
    # fold nobody could open yet.
    j = job(status="uploading")
    app._note_excluded(j, [{"episode_id": f"ep{i}", "role": "left",
                            "reason": "missing oakd_calib_offline.json"}
                           for i in range(40)])

    out = _status(j)

    assert out["excluded"] == []
    assert out["excluded_summary"].startswith("40 episode(s) left out")


async def _async(v):
    return v


# --- how many episodes actually ended up in the dataset -----------------------
# Derived from what the plan asked for minus what was lost after planning: the
# Space knows the real number and does not expose it, so the arithmetic has to be
# exact or the result line would lie about the delivery.

def test_the_count_is_unknown_without_a_plan_figure():
    # A job from before this was tracked must say nothing rather than say zero.
    assert app._episodes_in_dataset(job()) is None


def test_a_clean_build_holds_everything_it_asked_for():
    assert app._episodes_in_dataset(job(episodes_requested=12)) == 12


def test_losses_after_planning_are_subtracted():
    j = job(episodes_requested=12)
    app._note_excluded(j, [{"episode_id": "ep1", "stage": "upload",
                            "reason": "missing oakd_calib_offline.json"},
                           {"episode_id": "ep2", "stage": "conversion",
                            "reason": "SLAM failed"}])

    assert app._episodes_in_dataset(j) == 10


def test_plan_stage_drops_are_not_subtracted_twice():
    # THE trap: episodes_requested already excludes them. Subtracting again would
    # under-report the delivery — the opposite failure from the one we just fixed,
    # and just as misleading.
    j = job(episodes_requested=12)
    app._note_excluded(j, [{"episode_id": f"ep{i}", "stage": "plan",
                            "reason": "recorded without all of left+right"}
                           for i in range(3)])

    assert app._episodes_in_dataset(j) == 12


def test_a_recording_that_lost_both_arms_counts_once():
    j = job(episodes_requested=12)
    app._note_excluded(j, [
        {"episode_id": "ep1", "role": "left", "stage": "upload", "reason": "x"},
        {"episode_id": "ep1", "role": "right", "stage": "upload", "reason": "x"},
    ])

    assert app._episodes_in_dataset(j) == 11


def test_the_cascade_does_not_subtract_the_same_recording_twice():
    # An arm screened out at upload makes the Space drop the whole recording. The
    # device now reports its role, so the two reports dedupe onto one entry — and
    # even if they did not, the count is over distinct episode ids.
    j = job(episodes_requested=12)
    app._note_excluded(j, [{"episode_id": "ep1", "role": "left", "stage": "upload",
                            "reason": "missing oakd_calib_offline.json"}])
    app._note_excluded(j, [{"episode_id": "ep1", "role": "left", "stage": "conversion",
                            "reason": "excluded by the conversion"}])

    assert len(j.excluded) == 1
    assert app._episodes_in_dataset(j) == 11


def test_the_count_never_goes_negative():
    j = job(episodes_requested=2)
    app._note_excluded(j, [{"episode_id": f"ep{i}", "stage": "upload", "reason": "x"}
                           for i in range(5)])

    assert app._episodes_in_dataset(j) == 0


def test_the_status_endpoint_carries_the_count():
    j = job(episodes_requested=12, status="done")
    app._note_excluded(j, [{"episode_id": "ep1", "stage": "upload", "reason": "x"}])

    assert _status(j)["episodes"] == 11


def test_a_build_records_the_plan_figure():
    # Wired at creation, not recomputed later: the plan is the only moment the
    # fleet knows how many episodes it set out to include.
    j = job(episodes_requested=7)

    _run(j, upload_results={LEFT: {"status": "ok", "uploaded": ["ep1"]}},
         proc_result={"status": "ok", "result_url": "https://hf.co/datasets/me/ds"})

    assert app._episodes_in_dataset(j) == 7
