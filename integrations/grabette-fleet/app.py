"""Grabette fleet — operator dashboard + command broker (Docker HF Space).

A free-tier-friendly Docker Space. It is self-contained (no `grabette` import) so
it deploys standalone to HF. Responsibilities:

  * Operator UI + login via HF Spaces native OAuth (`hf_oauth: true`).
  * Command broker: an in-memory, per-owner device registry + command queue.
  * Device auth: devices call with `Authorization: Bearer <hf_token>`; we resolve
    the owner via `whoami` (cached) and group devices by HF identity.

Transport is short-polling (devices GET /api/devices/poll every couple seconds).
That polling traffic doubles as the keep-alive that stops a free Space sleeping
(sleep is timed from the last request). State is in-memory, so a restart drops
it — devices simply re-register on their next poll; durable data lives in the
device's own HF datasets, not here.

Designed to be duplicated per user: one Space = one owner = one fleet.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

logger = logging.getLogger("grabette-fleet")
from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth, whoami
from pydantic import BaseModel, Field
# Only used to read the SLAM Space's per-episode report (an unauthenticated GET,
# see _fetch_slam_quality_blocking). Declared in requirements.txt rather than
# leaned on as huggingface_hub's transitive dependency.
import requests

# A device is "online" if the fleet heard from it within this window. Liveness
# no longer rides on the command poll (long-polling holds that open ~LONG_POLL_S,
# and muxing runs off the poll loop) but on a dedicated lightweight HEARTBEAT the
# device sends every DEVICE_HEARTBEAT_S. So this only needs to cover a missed
# heartbeat + jitter (~3× the interval) — small, for fast disconnect detection.
DEVICE_HEARTBEAT_S = 5.0   # must match the device's relay heartbeat cadence
ONLINE_WINDOW = 15.0       # ~3 missed heartbeats → offline within ~15s
# Only discard a device's queued commands once it's been gone THIS long — far
# beyond ONLINE_WINDOW and any transient blip. Clearing on the mere online
# window (15s) would nuke an in-flight start/stop the instant a device flapped,
# breaking group sync; a device that's briefly "offline" but still holding a
# long-poll receives its queued command the moment it's enqueued regardless.
STALE_QUEUE_S = 120.0
# Command delivery is at-MOST-once by construction: poll() hands a command to the
# device inside an HTTP response and flips it to "sent" there and then. If that
# response never lands (dropped long-poll, proxy cutting a held connection, device
# restarting its poll loop) NOTHING re-sends it — the command is simply lost.
# For a stop_capture that is unacceptable: the pressed grabette stops locally
# while its peer, having never received the stop, records on forever — the pair
# desynchronises exactly when it matters. So the commands listed here are upgraded
# to at-LEAST-once: while unacknowledged (no /result yet) they are re-armed and
# re-delivered on the device's next poll. Only IDEMPOTENT commands may go in here
# — a second stop_capture on an already-stopped device is a no-op, whereas a
# re-delivered start_capture would schedule a phantom episode.
RETRYABLE_CMDS = {"stop_capture"}
# How long to wait for a command's result before assuming its delivery was lost.
# Must comfortably exceed a device's stop+mux+report round-trip, so a device that
# is simply busy tearing down isn't sent a redundant duplicate.
CMD_ACK_TIMEOUT_S = 8.0
# Give up re-arming after this many deliveries — bounds the retry when a device
# receives the command but never reports a result at all. The give-up is not
# silent: the session then names the device as not having confirmed its stop
# (see Session.pending_stop_cmds / _session_dict "stop_unconfirmed").
CMD_MAX_ATTEMPTS = 5
WHOAMI_TTL = 300.0
# Long-polling: the poll endpoint holds the connection open up to this many
# seconds when the device's queue is empty, returning the instant a command is
# enqueued (see _enqueue). This cuts command-delivery latency from ~1 poll
# interval to a network round-trip — notably a fanned-out group STOP reaches a
# peer in ~ms, so grouped episodes end within ~ms of each other instead of ~1s.
# MUST stay below BOTH the HF Space reverse-proxy idle timeout AND ONLINE_WINDOW
# (a held poll only refreshes last_seen when it returns, so a device mid-hold
# must not age out as offline). Set to 0 to DISABLE → classic short-polling: the
# poll returns immediately and the client throttles to its poll_interval. The
# relay client auto-detects which mode the server is in, so flipping this needs
# no device change — the escape hatch if the Space's proxy/sleep misbehaves.
LONG_POLL_S = 25.0
# LeRobot dataset generation: the fleet gathers the selected tasks' episodes,
# has each device push its OWN streams to a shared raw dataset (by role), then
# triggers a processing Space that converts raw → LeRobot (mono or bimanual,
# per the device set). Overridable so the test Space can be targeted.
LEROBOT_SPACE_URL = os.environ.get(
    "GRABETTE_LEROBOT_SPACE_URL", "https://pollen-robotics-grabette-slam.hf.space"
).rstrip("/")
# The intermediate raw dataset is currently NEVER deleted — not after a successful
# conversion, not on cancel. Keeping it means a build can be re-run or inspected
# without re-uploading from the devices, and a cancel leaves nothing half-cleaned.
# The fleet passes this as keep_raw in the process_dataset command; deleting the
# raw is the processing Space's action, so the Space must honour the flag for this
# to hold. Flip to False (here or via the env var) to restore the delete-after-
# conversion behaviour — that one switch is the whole opt-in.
KEEP_RAW_DATASET = os.environ.get("GRABETTE_KEEP_RAW_DATASET", "1") not in ("0", "false", "False")
# SLAM check ("Test SLAM" in an open session): the same upload → Space pipeline as a
# dataset build, run on the session's last few episodes to answer one question —
# is SLAM still tracking on this task, or is it producing is_lost frames? The Space
# reports per-episode quality and pushes a dataset ONLY when an episode is flagged,
# so a clean check leaves nothing on the Hub and a bad one leaves something to look
# at in the LeRobot visualizer.
#
# How the Space is told it's a check: by the TARGET REPO NAME. The relay client on
# the device rebuilds the Space payload from a FIXED set of keys, so a new command
# argument would only arrive once every Pi is redeployed, whereas the repo name the
# fleet chooses travels today. Must stay in sync with the grabette-slam Space,
# which reads it — and note the ORDER when changing this: the Space has to accept a
# marker before the fleet starts sending it, or a check lands there as an ordinary
# build and pushes a dataset unconditionally.
#
# Kept delimited by underscores rather than matched as a bare substring, so an
# ordinary dataset that happens to mention a trajectory check in its name isn't
# mistaken for one.
SLAM_CHECK_MARKER = "_trajectorycheck_"
SLAM_CHECK_DEFAULT_N = 3
# Upper bound on the episodes one check covers: it has to fit between two takes,
# and SLAM is the slow step. Mirrored in the UI's SLAM_MAX_N.
SLAM_CHECK_MAX_N = 5
# One report fetch. Short: it is polled repeatedly during a run, and a slow answer
# is only ever a missing table row, never a failed check.
SLAM_QUALITY_TIMEOUT_S = 10.0
# Lead time before a group's synchronized start actually fires. The device
# warms its hardware DURING this lead, then waits out the shared T0 on its own
# NTP-disciplined clock. The lead must cover (poll delivery + hardware warmup)
# so warmup finishes BEFORE T0 — otherwise the variable warmup leaks into the
# start and the devices drift apart. Warmup is only long when the OAK-D is cold
# (multi-second cold boot); once it has recorded recently it stays warm (device
# keepalive), so we use two leads:
# Must cover (poll delivery to the peer ~2.5s + OAK-D cold boot). Measured: a
# peer overran a 12s lead by ~0.7s (cold boot ≈10.2s, but it only had ~9.5s
# after poll delivery), so 15s gives ~2.3s margin over the observed cold boot.
GROUP_START_LEAD_COLD_S = 15.0   # OAK-D likely asleep → cover poll delivery + cold boot
# Even when warm, the lead must exceed the WORST-CASE command delivery to a
# peer: a device polls only every ~2.5s AND its relay is blocked while it muxes
# the previous episode's stop (several seconds), so a too-short warm lead makes
# the peer receive its start_capture after T0 (→ started late, best-effort — it
# no longer drops the episode, but it's desynced). 3s covers the common case;
# the real cure for back-to-back on-time starts is non-blocking relay muxing.
GROUP_START_LEAD_WARM_S = 1.0
# Fleet can't see the OAK-D power state directly; it infers "warm" from the time
# since the session's last recording stop. This window MUST stay safely below
# the device's OAK-D keepalive (GRABETTE oakd_keepalive_s, default 30s) so that
# whenever fleet says "warm" the OAK-D is DEFINITELY still powered — a false
# "warm" would desync the start, a false "cold" only costs an unnecessarily long
# lead. Fleet's stop timestamp is the dispatch time, earlier than the device's
# actual keepalive countdown start, which makes the estimate extra conservative.
OAK_WARM_WINDOW_S = 25.0
# Lead for a lead-based SYNCHRONIZED stop (shared future T_stop fanned to every
# member so they end together). CURRENTLY UNUSED: stops are dispatched
# immediately (see _dispatch_episode_stop) so a button press feels instant and
# peers trail only by the ~1s poll delivery. Kept here (with the device-side
# CaptureScheduler.schedule_stop path) so lead-based sync can be re-enabled
# without re-plumbing — e.g. if delivery latency ever grew. To switch back:
# have _dispatch_episode_stop send args={"stop_at_utc": now + GROUP_STOP_LEAD_S}.
GROUP_STOP_LEAD_S = 3.0


# --- device identity (Bearer token -> HF username, cached) -------------------
_whoami_cache: dict[str, tuple[str, float]] = {}
# Derived namespace list per owner (username + orgs) for the dataset owner
# dropdown — NOT the raw token. Device tokens list orgs reliably (unlike the
# short-lived operator OAuth token), so we compute the list from a device token
# when one passes through, cache the RESULT, and discard the token. In a shared
# Space this avoids holding every user's write-capable token in memory.
# owner -> (expiry_epoch, [namespaces]).
_namespaces_cache: dict[str, tuple[float, list[str]]] = {}
_NAMESPACES_TTL = 600.0


def _cached_user(token: str) -> Optional[str]:
    """Fast path: no I/O. Returns the cached username if still fresh."""
    hit = _whoami_cache.get(token)
    if hit and hit[1] > time.time():
        return hit[0]
    return None


def _whoami_blocking(token: str) -> str:
    """Slow path: hits the HF API. Synchronous — the huggingface_hub client
    has no async variant, so this must always be run off the event loop
    thread (see verify_user) or a single cache-miss stalls every other
    request the whole broker is serving (all owners' polling, dispatch, and
    the synchronized-start scheduling), since this process runs a single
    worker with all state in memory.
    """
    try:
        info = whoami(token=token)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid HF token: {e}") from e
    name = (info or {}).get("name", "")
    if not name:
        raise ValueError("Could not resolve HF username")
    _whoami_cache[token] = (name, time.time() + WHOAMI_TTL)
    return name


async def verify_user(token: str) -> str:
    cached = _cached_user(token)
    if cached is not None:
        return cached
    return await asyncio.to_thread(_whoami_blocking, token)


async def device_auth(authorization: Optional[str] = Header(None)) -> tuple[str, str]:
    """FastAPI dep: resolve (owner, token) from a device's Bearer HF token.

    Returns the raw token too — handlers use it transiently within the request
    (e.g. dispatching a device's own Space call). It is deliberately NOT retained
    anywhere on the fleet: in a shared Space that would concentrate every user's
    write-capable HF token in one process. Owner identity is resolved via a cached
    whoami; the operator's namespace list is derived + cached separately (see
    _touch_namespaces) so no raw token needs to be kept.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing 'Authorization: Bearer <hf_token>'")
    token = authorization[len("Bearer ") :].strip()
    try:
        owner = await verify_user(token)
    except ValueError as e:
        raise HTTPException(401, str(e)) from e
    return owner, token


async def device_owner(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dep: resolve just the device's owner from its Bearer HF token."""
    owner, _token = await device_auth(authorization)
    return owner


def _cache_namespaces_blocking(owner: str, token: str) -> None:
    """Blocking: resolve the owner's pushable namespaces (username + orgs) from a
    device token and cache the RESULT — never the token itself."""
    names = [owner]
    try:
        info = whoami(token=token)
        names += [o["name"] for o in (info.get("orgs") or []) if o.get("name")]
    except Exception:
        logger.debug("namespaces whoami failed for %s", owner, exc_info=True)
    _namespaces_cache[owner] = (time.time() + _NAMESPACES_TTL, list(dict.fromkeys(names)))


def _touch_namespaces(owner: str, token: Optional[str]) -> None:
    """TTL-gated, fire-and-forget: refresh the owner's namespace list from a
    passing device token, so the dataset-owner dropdown works WITHOUT the fleet
    keeping the token. Marks the cache fresh up front to avoid a refresh stampede."""
    if not token:
        return
    ent = _namespaces_cache.get(owner)
    if ent and ent[0] > time.time():
        return  # still fresh
    _namespaces_cache[owner] = (time.time() + _NAMESPACES_TTL, ent[1] if ent else [owner])
    t = asyncio.create_task(asyncio.to_thread(_cache_namespaces_blocking, owner, token))
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)


def operator_auth(request: Request) -> tuple[str, Optional[str]]:
    """Resolve (owner, oauth_access_token) for the logged-in operator, or 401.

    The OAuth token identifies the operator (their namespace). The fleet keeps
    no durable state of its own — the task/episode view is aggregated live from
    connected devices' reports (see _reported_tasks)."""
    info = parse_huggingface_oauth(request)
    if info is None:
        raise HTTPException(401, "Not logged in")
    owner = info.user_info.preferred_username or info.user_info.name
    return owner, getattr(info, "access_token", None)


def operator_name(request: Request) -> str:
    """Resolve the logged-in operator's HF username, or 401."""
    return operator_auth(request)[0]


# --- in-memory fleet state ---------------------------------------------------
@dataclass
class Command:
    id: str
    type: str
    args: dict[str, Any]
    status: str = "pending"  # pending | sent | done
    result: Optional[dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    done_at: Optional[float] = None
    # Delivery bookkeeping for the at-least-once retry (see RETRYABLE_CMDS):
    # when this delivery went out, and how many times it has been handed over.
    # "sent" only means "written into a poll response" — never that the device
    # got it; only a /result does. attempts>1 ⇒ this is a re-delivery.
    sent_at: Optional[float] = None
    attempts: int = 0


@dataclass
class Device:
    device_id: str
    name: str
    capabilities: list[str]
    hand: str = ""  # "left" or "right", reported by the device on register
    ip: str = ""  # device LAN IPv4, reported by the device on register
    # WiFi SSID the device is on, reported on register ("" on a wired/hotspot
    # device, or one running a build that predates the field). The IP alone
    # doesn't say which LAN a device sits on when several are in play.
    network: str = ""
    battery: Optional[float] = None  # last-reported battery %, sent via heartbeat
    # Device self-reported activity, sent on the heartbeat:
    # "" (device not updated / unknown) | idle | capturing | uploading | processing.
    # Empty falls back to fleet-side inference (see _device_activity).
    reported_status: str = ""
    # Why the device REFUSES to record ("" = healthy), self-reported on the
    # heartbeat: a hardware fault that would make every episode unconvertible
    # (no OAK-D offline calibration, no gripper angle sensors). Deliberately NOT
    # folded into reported_status: the two are orthogonal — a faulted device can
    # also be uploading — and an operator needs the fault even when the device is
    # otherwise busy. A device on a build that predates this always reports ""
    # and simply looks healthy, exactly as before.
    hardware_error: str = ""
    recording_buffers: dict = field(default_factory=dict)
    # This device's recorded tasks, reported on register (see TaskManager.report_tasks).
    # The device is the durable source of truth; the fleet aggregates these across
    # connected devices (phase 2). Stored but not yet consumed in phase 1.
    tasks: list[dict] = field(default_factory=list)
    # Loose episodes this device holds outside any task, as reported: {"total",
    # "episodes"}. Deliberately NOT folded into `tasks`: tasks are merged across
    # devices by name, while a loose episode was recorded alone and belongs to
    # this device only. Keeping it apart is also what stops these episodes from
    # ever reaching task selection or dataset generation.
    unassigned: dict = field(default_factory=dict)
    # Bumped each register (when tasks are (re)reported). Feeds the aggregation
    # cache signature so it recomputes only when a device's report changed.
    report_rev: int = 0
    last_seen: float = field(default_factory=time.time)
    queue: list[Command] = field(default_factory=list)
    history: list[Command] = field(default_factory=list)
    pending_delete: bool = False
    # Set whenever a command is enqueued (see _enqueue) to release a long-poll
    # holding on this device. Excluded from repr/eq — it's live runtime state,
    # never serialized (devices aren't persisted). asyncio.Event() binds to a
    # loop lazily (on first await), so constructing it off-loop here is fine.
    wakeup: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)

    @property
    def online(self) -> bool:
        return (time.time() - self.last_seen) < ONLINE_WINDOW


FLEET: dict[str, dict[str, Device]] = {}  # owner -> device_id -> Device


def _fleet_of(owner: str) -> dict[str, Device]:
    return FLEET.setdefault(owner, {})


def kind_of(dev: Device) -> str:
    """Mirrors the frontend's kindOf(): classify a device as grabette/gripette/casquette."""
    s = f"{dev.name} {dev.device_id} {' '.join(dev.capabilities)}".lower()
    if "gripette" in s:
        return "gripette"
    if "casquette" in s:
        return "casquette"
    return "grabette"


def _device_slot(dev: Device) -> Optional[str]:
    """The task-signature role this device fills: 'left'/'right' for a handed
    grabette, 'casquette' for a casquette, else None (not recordable)."""
    k = kind_of(dev)
    if k == "casquette":
        return "casquette"
    if k == "grabette" and dev.hand in ("left", "right"):
        return dev.hand
    return None


# --- tasks (registry, source of truth for task names sent to devices) --------
# A task is a *type of action*. It carries only the device *roles* it expects
# (its "signature"), never concrete device ids — those live on the group. The
# task NAME is the stable join key devices resolve locally via
# get_or_create_task, so it must come from this single registry.
VALID_SLOTS = ("left", "right", "casquette")


@dataclass
class Task:
    id: str
    name: str
    description: str = ""
    # Expected device roles, a subset of VALID_SLOTS (e.g. ["left","right"] for
    # a bimanual task, ["right"] for a single right-hand one). Empty = no
    # constraint. Used to validate the group assigned to the task and to tell
    # the dataset builder which roles produce data.
    device_signature: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


TASKS: dict[str, dict[str, Task]] = {}  # owner -> task_id -> Task


def _tasks_of(owner: str) -> dict[str, Task]:
    return TASKS.setdefault(owner, {})


# --- task/episode aggregation from device reports ---------------------------
# The devices are the durable source of truth for tasks and for who recorded each
# episode. On connect each device reports its tasks (see TaskManager.report_tasks);
# the fleet merges the reports from currently-connected devices to reconstruct the
# task/episode view — so ANY operator sees the existing tasks and can generate a
# dataset from them, regardless of which HF account did the original acquisition.

# Task names to hide briefly after an edit(rename)/delete, until the devices have
# processed the command and re-reported without them. Bridges the window where a
# stale report would otherwise resurrect a just-deleted/renamed task. owner ->
# {name: expiry_epoch}.
TASK_SUPPRESSED: dict[str, dict[str, float]] = {}
_SUPPRESS_S = 20.0
# Grace after a session stops before its episodes can be considered "orphaned":
# covers the window where one member has registered + re-reported the episode but
# the peer hasn't yet, so an unfinished recording isn't mistaken for a lost pair.
ORPHAN_GRACE_S = 30.0
# Episode reconciliation is EVENT-DRIVEN: recomputed when a device registers
# (connects / re-reports its tasks) — the only moments these can change — and
# cached here, so the operator UI just reads the cache instead of forcing a
# recompute on every poll. Both come from ONE pass (see _reconcile_episodes); the
# per-poll cost of surfacing them is therefore zero, which matters because the
# Space is a single event loop shared by every operator and every device poll.
# owner -> list of orphan-episode dicts / of split-filing dicts.
ORPHANS_PENDING: dict[str, list[dict]] = {}
SPLIT_PENDING: dict[str, list[dict]] = {}


def _suppress_task_name(owner: str, name: str) -> None:
    TASK_SUPPRESSED.setdefault(owner, {})[name] = time.time() + _SUPPRESS_S


def _suppressed_names(owner: str) -> set[str]:
    now = time.time()
    supp = TASK_SUPPRESSED.get(owner)
    if not supp:
        return set()
    # Prune expired so a name can come back if it's ever legitimately re-created.
    for n in [n for n, exp in supp.items() if exp <= now]:
        del supp[n]
    return set(supp)


def _devices_reporting_task(owner: str, name: str) -> list["Device"]:
    """Online devices whose latest report includes a task with this name."""
    out = []
    for dev in _fleet_of(owner).values():
        if dev.online and any((t.get("name") == name) for t in (dev.tasks or [])):
            out.append(dev)
    return out

def _task_episode_items(t: dict):
    """Yield (episode_id, members) for a reported task, supporting the compact
    'groups' format (episodes grouped by shared membership) and the legacy
    per-episode 'episodes' format (older device firmware)."""
    if "groups" in t:
        for grp in t.get("groups", []):
            members = grp.get("members") or {}
            for eid in grp.get("episode_ids", []):
                if eid:
                    yield eid, members
    else:
        for ep in t.get("episodes", []):
            eid = ep.get("episode_id")
            if eid:
                yield eid, (ep.get("members") or {})


# Aggregation is memoized per owner: recomputing it (O(all reported episodes))
# on every dashboard poll doesn't scale. A cheap signature over the owner's
# devices (online status + a per-device report revision + suppressed names)
# tells us when anything that feeds the aggregation actually changed; otherwise
# we return the cached result. owner -> (signature, aggregation).
_REPORTED_CACHE: dict[str, tuple] = {}


def _reported_sig(owner: str):
    devs = _fleet_of(owner)
    return (tuple(sorted((d.device_id, d.online, d.report_rev) for d in devs.values())),
            tuple(sorted(_suppressed_names(owner))))


def _compute_reported_tasks(owner: str) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    suppressed = _suppressed_names(owner)  # just-edited/deleted names, hidden briefly
    for dev in _fleet_of(owner).values():
        if not dev.online:
            continue
        for t in dev.tasks or []:
            name = t.get("name")
            if not name or name in suppressed:
                continue
            entry = agg.setdefault(name, {"description": "", "device_signature": [], "episodes": {}})
            if t.get("description") and not entry["description"]:
                entry["description"] = t["description"]
            if t.get("device_signature") and not entry["device_signature"]:
                entry["device_signature"] = list(t["device_signature"])
            for eid, ep_members in _task_episode_items(t):
                members = entry["episodes"].setdefault(eid, {})
                for role, who in ep_members.items():
                    members.setdefault(role, who)
    return agg


def _reported_tasks(owner: str) -> dict[str, dict]:
    """Merge online devices' reported tasks, keyed by task name (memoized). The
    SAME episode is reported by every member device (same episode_id + members),
    so we dedup by episode_id and union members. Returns {name: {"description",
    "device_signature", "episodes": {episode_id: {role: {device_id, name}}}}}.
    The result is cached and must be treated as READ-ONLY by callers."""
    sig = _reported_sig(owner)
    cached = _REPORTED_CACHE.get(owner)
    if cached is not None and cached[0] == sig:
        return cached[1]
    agg = _compute_reported_tasks(owner)
    _REPORTED_CACHE[owner] = (sig, agg)
    return agg


def _episode_started_at(episode_id: str) -> float:
    """Best-effort epoch seconds from an episode id (UTC 'YYYYmmdd_HHMMSS')."""
    try:
        return datetime.strptime(episode_id, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def _episodes_from_entry(entry: Optional[dict]) -> list[dict]:
    """Normalise a _reported_tasks entry's episodes into a sorted list carrying
    both members (role → {device_id, name}) and a plain roles map (role → id)."""
    if not entry:
        return []
    out = []
    for eid, members in entry["episodes"].items():
        out.append({
            "episode_id": eid,
            "members": members,
            "roles": {role: who.get("device_id") for role, who in members.items()},
            "started_at": _episode_started_at(eid),
        })
    out.sort(key=lambda e: e["episode_id"])
    return out


_REPORTERS_CACHE: dict[str, tuple] = {}


def _online_episode_reporters(owner: str) -> dict[str, dict]:
    """From CONNECTED devices, map episode_id → {role: {device_id, name}} built
    from each reporting device's slot. Lets the fleet reconstruct who recorded an
    episode saved before the device persisted its members — the reporting device
    IS a member, and its role is its slot. One pass over the reports.

    Memoized on the same signature as _reported_tasks, for the same reason and it
    has the same inputs (online devices + their report revisions; a name or hand
    change only happens on a register, which bumps the revision). Without this it
    was the one O(all reported episodes) pass left on a READ path: list_tasks calls
    it on every /api/fleet/tasks request, i.e. every 3s per open dashboard tab, all
    on the Space's single event loop. Treat the result as READ-ONLY.
    """
    sig = _reported_sig(owner)
    cached = _REPORTERS_CACHE.get(owner)
    if cached is not None and cached[0] == sig:
        return cached[1]
    out: dict[str, dict] = {}
    for dev in _fleet_of(owner).values():
        if not dev.online:
            continue
        role = _device_slot(dev)
        if not role:
            continue
        who = {"device_id": dev.device_id, "name": dev.name}
        for t in dev.tasks or []:
            for eid, _members in _task_episode_items(t):
                if eid:
                    out.setdefault(eid, {}).setdefault(role, who)
    _REPORTERS_CACHE[owner] = (sig, out)
    return out


def _reconcile_tasks(owner: str) -> None:
    """Ensure a Task exists locally for every task reported by connected devices,
    so a freshly-connected operator's registry is populated from the devices (the
    source of truth). Purely in-memory: the fleet persists nothing, so this is
    rebuilt from the reports on every restart."""
    tasks = _tasks_of(owner)
    by_name = {t.name: t for t in tasks.values()}
    for name, entry in _reported_tasks(owner).items():
        t = by_name.get(name)
        if t is None:
            tid = uuid.uuid4().hex[:8]
            tasks[tid] = Task(id=tid, name=name, description=entry["description"],
                              device_signature=list(entry["device_signature"]))
        else:
            # Backfill from the authoritative device report where we're missing it.
            if not t.device_signature and entry["device_signature"]:
                t.device_signature = list(entry["device_signature"])
            if not t.description and entry["description"]:
                t.description = entry["description"]


def _reconcile_episodes(owner: str) -> dict[str, list[dict]]:
    """Both cross-device episode inconsistencies, from ONE pass over the reports.

    Returns {"orphans": [...], "split": [...]}.

    * orphans — an episode LOST A PAIR: one of its member devices is ONLINE but no
      longer reports it (it deleted the episode / task) while another online member
      still holds it. This is what surfaces after a multi-device task was deleted
      with a peer offline: on reconnect the peer's copies are seen as orphaned and
      cleanup is proposed.

    * split — an episode FILED UNDER DIFFERENT TASKS: two online devices both hold
      it, but disagree on which task it belongs to. Orphan detection is blind to
      this by construction, because it only asks whether each device still reports
      the episode AT ALL, never under which name — so a device that merely refiles
      an episode locally ("Move to task" on its own UI) still counts as a holder.
      Two ways in: that local refile, and a fleet rename that reached only the
      online devices (_devices_reporting_task skips offline ones) so a peer comes
      back carrying the old name past the suppression window. Left undetected it is
      worse than cosmetic — the fleet merges reports BY NAME, so the episode shows
      up under both tasks looking complete, and a dataset built from either asks
      every member to upload it (uploads go by episode_id and never consult the
      task), putting the same take in two datasets under two different labels.

    Both are computed from the same {episode → which devices report it, and under
    what name} mapping, so they share one traversal: this runs on every device
    register, and the reports are O(all episodes ever recorded).

    Episodes whose missing member is OFFLINE are never flagged — we can't tell what
    that device still has. Suppressed (just renamed/deleted) task names stop a stale
    name from being attributed, but never make an episode look absent — see the
    inner loop, where that distinction is what keeps the cleanup button safe."""
    fleet = _fleet_of(owner)
    suppressed = _suppressed_names(owner)
    online = {d.device_id: d for d in fleet.values() if d.online}
    # Episodes still "in flux" — being recorded now, or just stopped — must NOT be
    # flagged: one member registers + re-reports before the other, so for a few
    # seconds it exists on one device and not (yet) the peer, which looks orphaned
    # but is just an unfinished recording. Skip any episode in an open session, or
    # in a session stopped within the grace window (covers the re-report lag).
    now = time.time()
    in_flux: set[str] = set()
    for s in _sessions_of(owner).values():
        if s.status == "open" or (s.last_stop_at is not None and now - s.last_stop_at < ORPHAN_GRACE_S):
            for ep in s.episodes:
                eid = ep.get("episode_id")
                if eid:
                    in_flux.add(eid)
    dev_eps: dict[str, set[str]] = {}          # device_id -> episode ids it reports
    ep_info: dict[str, dict] = {}              # episode_id -> {members, task}
    # episode_id -> {device_id: the task name THAT device files it under}. This is
    # the one addition the split check needs: ep_info keeps only the first
    # reporter's name (setdefault), which is precisely the information that made
    # divergent filing invisible.
    ep_filings: dict[str, dict[str, str]] = {}
    for dev in online.values():
        eids: set[str] = set()
        for t in dev.tasks or []:
            name = t.get("name")
            if not name:
                continue
            # Two DIFFERENT questions, and suppression answers only the second:
            #   "does this device still hold the episode?"  → the whole report,
            #    suppressed names included;
            #   "under which task does it file it?"          → suppressed names
            #    excluded, so a just-renamed/deleted name can't reappear.
            # Conflating them is what made the cleanup button dangerous: a device
            # whose only copy sat in the suppressed task reported nothing about
            # that episode, so it read as having DELETED it, and the peer holding
            # the correctly-filed copy was flagged an orphan and offered for
            # deletion. The old code only got away with it while suppression hid
            # the name on every member device symmetrically — which is exactly
            # what stops being true once one device files the episode elsewhere.
            for eid, ep_members in _task_episode_items(t):
                eids.add(eid)
                if name in suppressed:
                    continue
                ep_info.setdefault(eid, {"members": ep_members, "task": name})
                ep_filings.setdefault(eid, {})[dev.device_id] = name
        dev_eps[dev.device_id] = eids

    def _name(mid: str, fallback: str) -> str:
        return online[mid].name if mid in online else (fallback or mid)

    orphans: list[dict] = []
    split: list[dict] = []
    for eid, info in ep_info.items():
        if eid in in_flux:
            continue  # being recorded / just stopped — not a real inconsistency
        member_ids = {w.get("device_id"): w.get("name") for w in info["members"].values() if w.get("device_id")}
        deleters = [(m, _name(m, n)) for m, n in member_ids.items()
                    if m in online and eid not in dev_eps.get(m, set())]
        holders = [(m, _name(m, n)) for m, n in member_ids.items()
                   if m in online and eid in dev_eps.get(m, set())]
        if deleters and holders:  # a peer deleted it, but someone online still has it
            orphans.append({
                "episode_id": eid, "task": info["task"], "started_at": _episode_started_at(eid),
                "holders": [{"device_id": m, "name": n} for m, n in holders],
                "deleted_by": [{"device_id": m, "name": n} for m, n in deleters],
            })
        # Keyed on who REPORTS the episode, not on its recorded members: the
        # reporters are the devices actually holding a copy filed somewhere, which
        # is what has to agree. An episode only one device reports can't disagree.
        filings = ep_filings.get(eid, {})
        names = set(filings.values())
        if len(filings) > 1 and len(names) > 1:
            split.append({
                "episode_id": eid, "started_at": _episode_started_at(eid),
                "tasks": sorted(names),
                "filings": sorted(
                    ({"device_id": m, "name": _name(m, member_ids.get(m, "")), "task": tn}
                     for m, tn in filings.items()),
                    key=lambda f: (f["task"], f["name"]),
                ),
            })
    return {"orphans": orphans, "split": split}


def _task_name(owner: str, task_id: str) -> str:
    t = _tasks_of(owner).get(task_id)
    return t.name if t else ""


@dataclass
class Group:
    id: str
    name: str
    left: str = ""  # device_id of the left-hand grabette in this group
    right: str = ""  # device_id of the right-hand grabette in this group
    casquette: str = ""  # device_id of the casquette in this group
    created_at: float = field(default_factory=time.time)
    # (No task here — the task is chosen when a session is launched, not stored
    # on the group. See Session.)


GROUPS: dict[str, dict[str, Group]] = {}  # owner -> group_id -> Group


def _groups_of(owner: str) -> dict[str, Group]:
    return GROUPS.setdefault(owner, {})


# --- sessions (fleet-only: the recording-run + upload manifest) --------------
# A session is a run of one or more episodes recorded together for ONE task.
# It carries its own role→device membership (a "group of one" is just a
# single-entry map — no Group object is materialised) and the per-episode
# manifest: which physical device held which role for each episode. That
# manifest is what lets the dataset builder regroup an episode's data even
# when the device filling a role changes between episodes.
@dataclass
class Session:
    id: str
    task_id: str
    members: dict[str, str]  # role -> device_id, captured at launch
    status: str = "open"     # open | closed
    # Optimistic recording indicator: True between an episode start and its
    # stop. Maintained on every start/stop path (operator + physical button).
    # Not a hardware truth (real per-device state is reconciled in phase 3) —
    # enough to drive the "● recording" indicator in the UI.
    recording: bool = False
    # Epoch time of the last episode stop dispatched for this session. Drives
    # the warm/cold lead decision: within OAK_WARM_WINDOW_S the OAK-D is still
    # powered (device keepalive), so the next episode can use the short lead.
    # None until the first episode stops → first episode uses the cold lead.
    last_stop_at: Optional[float] = None
    # "stopping" phase, reconciled against reality: set True when an episode stop
    # is dispatched, and cleared when every dispatched stop_capture command has
    # reported its result (i.e. the devices actually finished tearing down + mux)
    # — so the UI's "stopping" indicator ends when the devices really stop, not
    # after a fixed guess. pending_stop_cmds maps each awaited stop_capture command
    # id → the device it was sent to, so a device that never confirms can be NAMED
    # (it may well still be recording) instead of the phase silently timing out.
    stopping: bool = False
    pending_stop_cmds: dict[str, str] = field(default_factory=dict)
    # device_id -> epoch when it FIRST reported itself idle while this session
    # still believed it was recording. Feeds the lost-stop safety net
    # (_reconcile_lost_stop); cleared as soon as the device reports otherwise.
    idle_since: dict[str, float] = field(default_factory=dict)
    # device_id -> why its start_capture failed, for the CURRENT episode.
    # _schedule_episode_start fires the starts and never looks at the results, so
    # a device that refused (hardware fault, busy, hardware still down) used to
    # leave its peers recording a half-rig take with nothing anywhere saying so —
    # the operator found out at dataset build time, if at all. Recorded on the
    # /result path and cleared at each new episode start.
    start_errors: dict[str, str] = field(default_factory=dict)
    # Each entry: {"episode_id": str, "roles": {role: device_id}, "started_at": float}
    episodes: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


SESSIONS: dict[str, dict[str, Session]] = {}  # owner -> session_id -> Session


@dataclass
class DatasetJob:
    """A LeRobot dataset build. Transient runtime state (not persisted): the
    fleet dispatches per-device uploads, waits for them, then (step 3) triggers
    the processing Space and tracks it to completion."""
    id: str
    task_ids: list[str]
    roles: list[str]              # the shared device signature of the selected tasks
    raw_repo: str                 # {owner}/<name>-raw — kept, see KEEP_RAW_DATASET
    target_repo: str              # {owner}/<name> — the resulting LeRobot dataset
    status: str = "uploading"     # uploading | raw_ready | processing | done | error | cancelled
    message: str = ""
    # Set by a cancel request. The background runner checks it on every tick and
    # bows out without touching the status (the cancel path owns it), so a cancel
    # never lands as a spurious "error". Terminal: a cancelled job is never resumed.
    cancelled: bool = False
    # 0..1 fraction for the determinate upload phase; None once processing starts
    # (the Space conversion runs opaquely behind one blocking device call, so the
    # UI shows an indeterminate bar there). 1.0 when done.
    progress: Optional[float] = 0.0
    result_url: Optional[str] = None
    error: Optional[str] = None
    # True once EVERY device finished its upload, i.e. the raw dataset is complete
    # on HF. A failure after that point is a conversion-only failure: the data is
    # safe and the operator can re-run the conversion straight from the Space
    # instead of re-uploading from the devices — the UI says so, so it needs to
    # tell the two failure kinds apart.
    raw_uploaded: bool = False
    # Every episode left OUT of this build, and why. Three separate stages can
    # drop one, and each used to report to nobody:
    #   • the plan      — recorded with fewer devices than the dataset needs, or
    #                     with a device that is offline;
    #   • each upload   — present on the device but missing a required file, so
    #                     the device screened it out before pushing gigabytes
    #                     (the reason is named per file, e.g. a missing
    #                     oakd_calib_offline.json);
    #   • the Space     — rejected after SLAM (bad recording check, no usable
    #                     trajectory, an arm missing from the recording).
    # The device and the Space both reported theirs faithfully and the fleet threw
    # the results away, so an operator whose task had a few bad episodes saw a
    # green "Dataset ready." and no number. One ledger, because the question is
    # one question: which of my takes are not in there, and why.
    # Entries: {"episode_id", "role", "reason", "stage", "device"}.
    excluded: list[dict[str, Any]] = field(default_factory=list)
    # Episodes the PLAN selected, i.e. what this build set out to include. The
    # plan's own drops (recorded with fewer devices, offline device) are already
    # subtracted here — which is why _episodes_in_dataset must not subtract them
    # a second time. 0 = unknown.
    episodes_requested: int = 0
    upload_cmds: dict[str, str] = field(default_factory=dict)  # device_id -> command id
    # The one device asked to run the raw → LeRobot conversion, and the command
    # doing it (set when the processing phase starts). Kept on the job so a cancel
    # can reach the processing device too, not just the uploaders.
    processor: str = ""
    proc_cmd: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    # --- trajectory check (see SLAM_CHECK_MARKER). It runs the exact same pipeline,
    # so it rides on this same job: progress, the per-device Cancel and the
    # "don't record on a busy device" gate all apply unchanged. What differs is
    # what the operator is shown at the end — a per-episode tracking report
    # instead of a dataset link — and that a clean one pushes nothing at all.
    check: bool = False
    episode_ids: list[str] = field(default_factory=list)  # exactly what was checked
    # The real task name, for the Space's `task` field: without it a check dataset
    # would be described by its own throwaway repo name.
    task_desc: str = ""
    # Per-episode rows from the Space (see _summarize_slam_quality). Filled DURING
    # the run — the Space publishes its report live — so the table builds up
    # episode by episode instead of appearing all at once at the end.
    quality: list[dict[str, Any]] = field(default_factory=list)
    # Episode names the Space did not consider clean. None = no report (yet, or the
    # report could not be read back), [] = everything tracked. The three states are
    # distinct on purpose: "no answer" must never render as "all good".
    flagged: Optional[list[str]] = None
    visualizer_url: Optional[str] = None
    # Takes a device answered "I don't have that" to at upload time. The most
    # reliable signal there is about a missing episode — better than any inference
    # from the reports — and the one that explains why a check covered fewer takes
    # than were asked for.
    missing_episodes: list[str] = field(default_factory=list)

    def device_cmds(self) -> dict[str, str]:
        """Every device this job put to work → the command doing that work. This is
        the set a cancel must cover: all uploaders, plus the processing device."""
        out = dict(self.upload_cmds)
        if self.processor and self.proc_cmd:
            out[self.processor] = self.proc_cmd
        return out


def _note_excluded(job: "DatasetJob", entries) -> None:
    """Add exclusions to a job's ledger, keeping the FIRST reason per episode+role.

    Deduped because the stages cascade: an arm screened out at upload for a
    missing calibration makes the Space drop the whole recording for a missing
    arm, and counting that as two lost episodes would misreport the damage. The
    first reason is also the most actionable one — the upload knows the file that
    is missing, the Space only knows the arm never arrived.
    """
    seen = {(e["episode_id"], e.get("role") or "") for e in job.excluded}
    for e in entries or []:
        eid = str(e.get("episode_id") or "").strip()
        if not eid:
            continue
        key = (eid, e.get("role") or "")
        if key in seen:
            continue
        seen.add(key)
        job.excluded.append({"episode_id": eid, "role": e.get("role") or "",
                             "reason": e.get("reason") or "excluded",
                             "stage": e.get("stage") or "", "device": e.get("device") or ""})


def _episodes_in_dataset(job: "DatasetJob") -> Optional[int]:
    """How many episodes the finished dataset holds, or None when unknowable.

    Derived, not reported: the authoritative count lives in the conversion Space
    (push_lerobot returns it) and its /api/status does not expose it. So we take
    what the plan asked for and subtract the recordings lost AFTER planning.

    Two things make this exact rather than a guess:
      • plan-stage exclusions are skipped — they were never in
        episodes_requested, and subtracting them would count them twice;
      • the count is over DISTINCT episode ids, so a bimanual recording that lost
        both arms is one lost recording, not two.

    Every way the Space drops a recording leaves a quality entry behind (a failed
    pre-check, a failed SLAM), and the fleet does not enable the trajectory-verdict
    filters — so nothing disappears silently between here and the dataset.
    """
    if not job.episodes_requested:
        return None
    lost = {e["episode_id"] for e in job.excluded if e.get("stage") != "plan"}
    return max(0, job.episodes_requested - len(lost))


def _excluded_summary(job: "DatasetJob") -> str:
    """"3 episode(s) left out: missing oakd_calib_offline.json (3)" — or "".

    Reasons are counted rather than listed: twenty episodes failing the same way
    is one fact. The COUNT is what was missing entirely before, so it leads."""
    if not job.excluded:
        return ""
    episodes = {e["episode_id"] for e in job.excluded}
    counts: dict[str, int] = {}
    for e in job.excluded:
        counts[e["reason"]] = counts.get(e["reason"], 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    detail = ", ".join(f"{reason} ({n})" for reason, n in ranked)
    more = "" if len(counts) <= 3 else f", +{len(counts) - 3} other reason(s)"
    return f"{len(episodes)} episode(s) left out: {detail}{more}"


DATASET_JOBS: dict[str, dict[str, DatasetJob]] = {}  # owner -> job_id -> job
# A job is cancellable while it still has devices working for it.
DATASET_LIVE = ("uploading", "raw_ready", "processing")


def _dataset_jobs_of(owner: str) -> dict[str, DatasetJob]:
    return DATASET_JOBS.setdefault(owner, {})


def _live_dataset_job_by_device(owner: str) -> dict[str, str]:
    """device_id → id of the live dataset job it's working for. Built once per
    request (not per device) so exposing it on the fleet listing stays O(jobs)."""
    out: dict[str, str] = {}
    for job in _dataset_jobs_of(owner).values():
        if job.status in DATASET_LIVE and not job.cancelled:
            for dev_id in job.device_cmds():
                out[dev_id] = job.id
    return out


def _sessions_of(owner: str) -> dict[str, Session]:
    return SESSIONS.setdefault(owner, {})


def _open_session_for_device(owner: str, device_id: str) -> Optional[Session]:
    for s in _sessions_of(owner).values():
        if s.status == "open" and device_id in s.members.values():
            return s
    return None


# --- device activity: what a device is doing, for the operator UI + the recording
# gates. Prefer what the fleet can INFER on its own from the work it dispatched (an
# in-flight upload/process command in the device's queue); then the device's own
# self-reported status (which also covers dashboard-initiated work the fleet never
# sees), which WINS once the device answers at all; and only for a device that
# reports nothing, capture inferred from the session's own state. Pure in-memory
# reads, no extra I/O, so it stays cheap at fleet scale.
_RECORDING_BLOCKERS = ("uploading", "processing")  # dataset work that must not overlap a recording


def _device_activity(owner: str, dev: Device) -> str:
    """One of: idle | capturing | uploading | processing."""
    # Dataset work the fleet ITSELF dispatched — authoritative, and the device may
    # not self-report it (relay commands don't create a local job). A command is
    # removed from the queue on result, so anything left here is live.
    types = {c.type for c in dev.queue}
    if "process_dataset" in types:
        return "processing"
    if "upload_episodes" in types:
        return "uploading"
    # Device self-reported activity — covers local dashboard work (SLAM push /
    # episode upload) and live capture that the fleet can't see on its own.
    st = (dev.reported_status or "").strip().lower()
    if st in ("capturing", "uploading", "processing"):
        return st
    if st:
        # The device answered, and it answered "idle". It is the only party that
        # knows whether its camera is rolling, so that answer wins. Falling through
        # to the inference below is what used to paint EVERY member of an open
        # session "recording" while they all sat idle between takes — the operator
        # then saw devices the fleet called busy and the devices called free.
        return "idle"
    # Nothing reported at all (a device predating the status heartbeat): fall back
    # to the session's own view, and only while it believes an episode is running.
    # "A session is open" is NOT "an episode is in progress" — a session stays open
    # between takes on purpose.
    s = _open_session_for_device(owner, dev.device_id)
    return "capturing" if (s is not None and s.recording) else "idle"


def _busy_recording_blockers(owner: str, device_ids) -> list[str]:
    """Subset of device_ids currently tied up by dataset work (upload/convert),
    which must be free before they can (re)start a recording."""
    fleet = _fleet_of(owner)
    return [did for did in device_ids
            if (d := fleet.get(did)) is not None
            and _device_activity(owner, d) in _RECORDING_BLOCKERS]


def _busy_what(owner: str, device_ids) -> str:
    """What these devices are busy WITH, for the message an operator gets when a
    recording is refused: a SLAM check is something they started themselves a
    minute ago, and calling it "processing a dataset" sends them looking for a
    build nobody launched."""
    checks = {j.id for j in _dataset_jobs_of(owner).values() if j.check}
    live = _live_dataset_job_by_device(owner)
    return ("running the trajectory check" if any(live.get(d) in checks for d in device_ids)
            else "processing a dataset")


def _dataset_job_blockers(owner: str, device_ids) -> list[str]:
    """Subset of device_ids enrolled in a LIVE dataset build or trajectory check.

    Wider than _busy_recording_blockers, and deliberately so: a job stays live
    across the gaps where no device reports uploading/processing — between an
    upload's result and the processing command reaching its device, and for every
    member that is not the one converting. Those gaps are exactly when a session
    the operator was told is paused would accept a new episode.
    """
    live = _live_dataset_job_by_device(owner)
    return [did for did in device_ids if did in live]


def _recording_blockers(owner: str, device_ids) -> list[str]:
    """Every reason a device in this set can't (re)start a recording: fleet work
    it is enrolled in, plus work it reports doing on its own (dashboard SLAM push,
    local upload) that no fleet job covers. THE gate — every start path calls it,
    so the fleet can never refuse a start the device would take, or the reverse."""
    blocked = dict.fromkeys(_dataset_job_blockers(owner, device_ids))
    blocked.update(dict.fromkeys(_busy_recording_blockers(owner, device_ids)))
    return list(blocked)


def _raise_if_recording_blocked(owner: str, device_ids) -> None:
    busy = _recording_blockers(owner, device_ids)
    if not busy:
        return
    fleet = _fleet_of(owner)
    names = ", ".join(fleet[d].name for d in busy if d in fleet) or "a device"
    raise HTTPException(409, {"message": f"{names} is busy {_busy_what(owner, busy)} — "
                                         f"wait for it to finish",
                              "devices": busy})


def _remove_from_groups(owner: str, device_id: str) -> None:
    for g in _groups_of(owner).values():
        if g.left == device_id:
            g.left = ""
        if g.right == device_id:
            g.right = ""
        if g.casquette == device_id:
            g.casquette = ""


def _validate_task_signature(owner: str, task_id: str, slots: set[str]) -> None:
    """The devices launched for a task must fill exactly the task's required
    roles — exact match (not just superset) so the dataset builder knows
    precisely which roles produce data. No-op when the task has no signature."""
    t = _tasks_of(owner).get(task_id)
    if t is None:
        raise HTTPException(404, f"Task {task_id} not found")
    if t.device_signature and set(t.device_signature) != slots:
        raise HTTPException(400, {
            "message": "selected devices don't match the task's required devices",
            "task": t.name,
            "required": sorted(t.device_signature),
            "got": sorted(slots),
        })


# --- persistence: none -------------------------------------------------------
# The fleet keeps NO durable state of its own. Devices are the source of truth:
# each re-registers on connect and reports its tasks + per-episode membership
# (see TaskManager.report_tasks), which the fleet aggregates in memory (see
# _reported_tasks / _reconcile_tasks). This means ANY operator sees the existing
# tasks and can build a dataset from them, regardless of which HF account did the
# acquisition — and nothing is ever written to an HF namespace on the fleet's
# behalf. Groups and open sessions are runtime-only and reset on a Space restart.

# Strong refs to fire-and-forget background tasks (e.g. dataset jobs). asyncio
# only weakly references tasks, so without this the GC can cancel one mid-run.
_bg_tasks: set[asyncio.Task] = set()


def _enqueue(dev: "Device", cmd: Command) -> None:
    """Queue a command for a device AND wake any long-poll holding on it, so the
    command is delivered on the next network round-trip instead of the next poll
    interval. Centralized on purpose: a bare dev.queue.append() would silently
    strand the command for up to LONG_POLL_S until the hold times out."""
    dev.queue.append(cmd)
    dev.wakeup.set()


def _rearm_unacked(dev: "Device") -> list[Command]:
    """Put back to "pending" every retryable command that was handed to the device
    but never acknowledged with a /result within CMD_ACK_TIMEOUT_S — its delivery
    was almost certainly lost with the response that carried it (see
    RETRYABLE_CMDS). Returns the re-armed commands.

    This is the watchdog that makes a group stop reliable: without it a single
    dropped poll response leaves one grabette recording forever while its peer has
    stopped. Called on every poll AND every heartbeat, so a device holding a
    long-poll gets the re-delivery pushed to it (the heartbeat sets dev.wakeup)
    instead of waiting out the hold."""
    now = time.time()
    rearmed = []
    for c in dev.queue:
        if (c.status == "sent" and c.type in RETRYABLE_CMDS
                and c.attempts < CMD_MAX_ATTEMPTS
                and c.sent_at is not None and (now - c.sent_at) >= CMD_ACK_TIMEOUT_S):
            c.status = "pending"
            rearmed.append(c)
    return rearmed


async def _operator_loaded(request: Request) -> str:
    """Resolve the authenticated operator (namespace). Kept as the single
    dependency every operator-facing handler uses; there is no snapshot to load
    anymore — the task/episode view is aggregated live from device reports."""
    owner, _token = operator_auth(request)
    return owner


# --- app + OAuth -------------------------------------------------------------
app = FastAPI(title="Grabette fleet")
attach_huggingface_oauth(app)  # adds /oauth/huggingface/{login,logout,callback}


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Public, auth-free liveness ping. A device hits this to WAKE the (free-tier,
    sleep-when-idle) Space and confirm it's up before starting OAuth — so the
    OAuth callback, which HF routes through this Space, lands on a running relay
    even when no operator has a fleet dashboard tab open to keep it warm."""
    return {"status": "ok"}


# === device-facing API (Bearer auth) ========================================
class RegisterReq(BaseModel):
    device_id: str
    name: str = ""
    capabilities: list[str] = []
    hand: str = ""  # "left" or "right" (from GRABETTE_HAND on the device)
    ip: str = ""  # device LAN IPv4, best-effort
    # WiFi SSID, best-effort. None (key absent) = a device build that doesn't know
    # the field yet → keep what we had; "" = the device looked and found none
    # (wired, hotspot, no nmcli) → clear it, don't show a stale network.
    network: Optional[str] = None
    tasks: list[dict] = []  # this device's recorded tasks (source of truth for aggregation)
    # This device's loose episodes, recorded outside any task: {"total": int,
    # "episodes": [...]} (see TaskManager.report_unassigned). Absent on a device
    # build that predates the field — an empty inbox, which is the safe reading.
    unassigned: dict = {}


class ResultReq(BaseModel):
    device_id: str
    command_id: str
    result: dict[str, Any] = {}


@app.post("/api/devices/register")
async def register(req: RegisterReq, auth: tuple[str, str] = Depends(device_auth)) -> dict[str, Any]:
    owner, token = auth
    _touch_namespaces(owner, token)  # refresh the owner's org list (token not kept)
    fleet = _fleet_of(owner)
    dev = fleet.get(req.device_id)
    if dev is None:
        dev = Device(req.device_id, req.name or req.device_id, req.capabilities,
                     hand=req.hand, ip=req.ip, network=req.network or "", tasks=req.tasks,
                     unassigned=req.unassigned or {})
        fleet[req.device_id] = dev
    else:
        dev.name = req.name or dev.name
        dev.capabilities = req.capabilities or dev.capabilities
        dev.hand = req.hand or dev.hand
        dev.ip = req.ip or dev.ip
        if req.network is not None:  # see RegisterReq.network: "" is meaningful here
            dev.network = req.network
        # Reported every register: the device is authoritative, so replace wholesale.
        dev.tasks = req.tasks
        dev.unassigned = req.unassigned or {}
    dev.report_rev += 1  # invalidates the aggregation cache for this owner
    dev.last_seen = time.time()
    # A device just (re)reported its tasks — the moment orphaned or divergently
    # filed episodes can appear/change. Recompute the reconciliation snapshot now
    # (event-driven, one pass for both) rather than on every operator poll.
    snapshot = _reconcile_episodes(owner)
    ORPHANS_PENDING[owner] = snapshot["orphans"]
    SPLIT_PENDING[owner] = snapshot["split"]
    return {"status": "ok", "pending": len(dev.queue)}


class BufferStats(BaseModel):
    capacity_bytes: int = Field(gt=0)
    peak_bytes: int = Field(ge=0)
    peak_percent: float = Field(ge=0, le=100)
    rejected_frames: int = Field(ge=0)
    write_errors: int = Field(ge=0)
    complete: bool
    error: str = Field(default="", max_length=2000)


class RecordingTelemetry(BaseModel):
    episode_id: Optional[str] = Field(default=None, max_length=200)
    buffers: dict[str, BufferStats] = Field(default_factory=dict, max_length=8)
    auto_stop_reason: str = Field(default="", max_length=2000)
    auto_stop_episode_id: Optional[str] = Field(default=None, max_length=200)
    is_capturing: bool = False
    is_stopping: bool = False
    capture_episode_id: Optional[str] = Field(default=None, max_length=200)


@app.post("/api/devices/heartbeat")
async def heartbeat(device_id: str, battery: Optional[float] = None,
                    status: Optional[str] = None,
                    error: Optional[str] = None,
                    telemetry: Optional[RecordingTelemetry] = None,
                    auth: tuple[str, str] = Depends(device_auth)) -> dict[str, str]:
    """Lightweight liveness ping, sent every DEVICE_HEARTBEAT_S independently of
    the (long-held) command poll — this is what keeps last_seen fresh and lets
    the fleet detect a disconnect within ONLINE_WINDOW. Deliberately does no HF
    I/O so it stays cheap at a few-second cadence. Also carries the device's
    battery % and its self-reported activity, so the fleet can show device state
    without polling."""
    owner, _token = auth
    dev = _fleet_of(owner).get(device_id)
    if dev is None:
        raise HTTPException(404, "Device not registered")
    dev.last_seen = time.time()
    if telemetry is not None:
        dev.recording_buffers = telemetry.model_dump()
    if battery is not None:
        dev.battery = battery
    if error is not None:
        # Sent on every beat, empty included — that is what lets a cleared fault
        # be seen as cleared. `None` means an older device that never sends the
        # field at all, which must not be read as "healthy now": leave whatever
        # we last knew rather than inventing good news.
        dev.hardware_error = error.strip()
    if status is not None:
        dev.reported_status = status
        # The status just refreshed, so this is the moment to check it against
        # what the session believes — catches a stop whose fan-out was lost
        # before it ever reached us (see _reconcile_lost_stop). Guarded: this
        # endpoint IS the device's liveness, and a 500 here would flap it offline
        # and drop its queued commands — far worse than a missed reconciliation.
        try:
            _reconcile_lost_stop(owner, dev)
        except Exception:  # noqa: BLE001
            logger.exception("lost-stop reconciliation failed for %s", device_id)
    # Watchdog tick: the heartbeat is the fleet's only regular per-device beat
    # (every DEVICE_HEARTBEAT_S, independent of the long-held poll), so it's where
    # a lost stop_capture is noticed. Waking the device's poll pushes the
    # re-delivery immediately rather than waiting for the hold to time out.
    if _rearm_unacked(dev):
        dev.wakeup.set()
    return {"status": "ok"}


@app.get("/api/devices/poll")
async def poll(device_id: str, auth: tuple[str, str] = Depends(device_auth)) -> dict[str, Any]:
    owner, token = auth
    dev = _fleet_of(owner).get(device_id)
    if dev is None:
        raise HTTPException(404, "Device not registered")
    _touch_namespaces(owner, token)  # keeps the org list fresh (TTL-gated; token not kept)
    dev.last_seen = time.time()
    # A fresh poll means the previous response is water under the bridge: any
    # retryable command still unacknowledged from it is re-armed BEFORE the
    # long-poll gate below, so it goes out in THIS response instead of waiting.
    _rearm_unacked(dev)
    # Long-poll: hold the connection open until a command is enqueued (which
    # sets dev.wakeup) or LONG_POLL_S elapses, so a command is delivered on the
    # next round-trip instead of the next poll interval. Clear the event FIRST,
    # then re-check the queue: an enqueue landing in that gap sets the event, so
    # the wait() returns immediately rather than stranding the command. When
    # LONG_POLL_S<=0 this whole block is skipped → short-poll (return at once).
    if LONG_POLL_S > 0:
        dev.wakeup.clear()
        if not any(c.status == "pending" for c in dev.queue):
            try:
                await asyncio.wait_for(dev.wakeup.wait(), timeout=LONG_POLL_S)
            except asyncio.TimeoutError:
                pass
            # NB: do NOT refresh last_seen here — a held poll that times out on a
            # DEAD device would otherwise look like fresh contact and delay
            # offline detection. last_seen reflects the poll's ARRIVAL (above);
            # liveness is kept fresh by the separate lightweight heartbeat.
    pending = [c for c in dev.queue if c.status == "pending"]
    now = time.time()
    for c in pending:
        c.status = "sent"
        c.sent_at = now      # starts the acknowledgement clock (see _rearm_unacked)
        c.attempts += 1
    return {"commands": [{"id": c.id, "type": c.type, "args": c.args} for c in pending]}


@app.post("/api/devices/result")
async def result(req: ResultReq, auth: tuple[str, str] = Depends(device_auth)) -> dict[str, str]:
    owner, _token = auth
    dev = _fleet_of(owner).get(req.device_id)
    if dev is None:
        raise HTTPException(404, "Device not registered")
    matched: Optional[Command] = None  # never the loop variable: an unmatched
    # id would otherwise leave `c` unbound (empty queue) or, worse, pointing at
    # an unrelated command, and the checks below would act on it.
    for c in dev.queue:
        if c.id == req.command_id:
            matched = c
            c.status = "done"
            c.result = req.result
            c.done_at = time.time()
            dev.queue.remove(c)
            dev.history.insert(0, c)
            # A cancel that came back closes out the work it was cancelling: the
            # device has stopped, so its upload/conversion command must not linger
            # in the queue — that queue entry is what makes the device read as
            # "uploading"/"processing", which would block recording forever if the
            # device only ever reports on the cancel itself.
            if c.type == "cancel_dataset":
                for target in (c.args or {}).get("command_ids") or []:
                    for q in [x for x in dev.queue if x.id == target]:
                        q.status, q.result, q.done_at = "done", {"status": "cancelled"}, time.time()
                        dev.queue.remove(q)
                        dev.history.insert(1, q)  # keep the cancel itself first
            del dev.history[20:]
            if c.type == "logout" and dev.pending_delete:
                fleet = _fleet_of(owner)
                fleet.pop(dev.device_id, None)
                _remove_from_groups(owner, dev.device_id)
            break
    # A start_capture that came back NOT ok means this device is not recording,
    # while its peers are. The fleet dispatches starts fire-and-forget (see
    # _schedule_episode_start), so this result is the only moment it can ever
    # learn that — attach it to the device's open session, which is what the
    # operator is looking at.
    # "scheduled" is a SUCCESS ack, not a failure: the fleet only ever dispatches
    # synchronized starts (every start_capture carries a shared start_at_utc), and a
    # device answers those with "scheduled" after accepting T0 — "ok" comes back only
    # from the immediate, target-less path the fleet never uses. Treating it as an
    # error reported every good group start as a missing arm.
    if (matched is not None and matched.type == "start_capture"
            and (matched.result or {}).get("status") not in ("ok", "scheduled")):
        sess = _open_session_for_device(owner, req.device_id)
        if sess is not None:
            sess.start_errors[req.device_id] = _why_not_ok(matched.result or {}, "start failed")
            logger.warning("start_capture failed on %s (session %s): %s",
                           req.device_id, sess.id, sess.start_errors[req.device_id])

    # A stop_capture result means that device actually finished tearing down its
    # capture. End the session's "stopping" phase once EVERY dispatched stop has
    # reported — so the UI leaves "stopping" exactly when the devices really stop.
    for s in _sessions_of(owner).values():
        if req.command_id in s.pending_stop_cmds:
            s.pending_stop_cmds.pop(req.command_id, None)
            if not s.pending_stop_cmds:
                s.stopping = False
            break
    return {"status": "ok"}


# === operator-facing API (session OAuth) =====================================
class DispatchReq(BaseModel):
    device_id: str
    type: str
    args: dict[str, Any] = {}


@app.get("/api/fleet/me")
async def me(request: Request) -> dict[str, Any]:
    info = parse_huggingface_oauth(request)
    if info is None:
        return {"logged_in": False}
    return {"logged_in": True, "username": info.user_info.preferred_username or info.user_info.name}


@app.get("/api/fleet/namespaces")
async def namespaces(request: Request) -> dict[str, Any]:
    """The namespaces the operator can push a dataset to: their own username +
    the orgs they belong to. Used to populate the dataset owner dropdown.

    Served from the cache built when a device token passed through (device tokens
    list orgs reliably; see _touch_namespaces) — so the fleet keeps no raw token.
    Cold start (no device seen yet for this owner): best-effort via the operator's
    OAuth token, which lists orgs poorly, falling back to just the username."""
    owner, oauth_token = operator_auth(request)
    ent = _namespaces_cache.get(owner)
    if ent and ent[1]:
        return {"namespaces": ent[1], "default": owner}
    names = [owner]
    if oauth_token:
        try:
            info = await asyncio.to_thread(whoami, oauth_token)
            names += [o["name"] for o in (info.get("orgs") or []) if o.get("name")]
        except Exception:
            logger.warning("whoami for namespaces failed for %s", owner, exc_info=True)
    seen: set[str] = set()
    uniq = [n for n in names if not (n in seen or seen.add(n))]
    return {"namespaces": uniq, "default": owner}


@app.get("/api/fleet/devices")
async def list_devices(request: Request) -> dict[str, Any]:
    owner = operator_name(request)
    devices = list(_fleet_of(owner).values())
    # Which live dataset build each device is working for, so the listing can offer
    # a Cancel even to an operator who never saw (or has left) the build page.
    ds_jobs = _live_dataset_job_by_device(owner)
    # Which of those are SLAM checks: "busy uploading" reads very differently to an
    # operator mid-session depending on whether it's a build or the check they just
    # started themselves.
    check_ids = {j.id for j in _dataset_jobs_of(owner).values() if j.check}
    now = time.time()
    for d in devices:
        # Only drop stale commands for a LONG-gone device — never on the mere
        # 15s online window (would nuke an in-flight group start/stop). Retryable
        # commands SURVIVE the purge: dropping a queued stop_capture is exactly how
        # a grabette that was recording when it dropped off comes back, never gets
        # told to stop, and keeps filling its card.
        if now - d.last_seen > STALE_QUEUE_S:
            d.queue[:] = [c for c in d.queue
                          if c.type in RETRYABLE_CMDS and c.attempts < CMD_MAX_ATTEMPTS]
    return {
        "owner": owner,
        "devices": [
            {
                "device_id": d.device_id,
                "name": d.name,
                "online": d.online,
                "capabilities": d.capabilities,
                "hand": d.hand,
                "ip": d.ip,
                "network": d.network,
                "battery": d.battery,
                "activity": _device_activity(owner, d),
                # Why this device refuses to record ("" = healthy). Separate from
                # activity: a device can be both faulted and busy.
                "hardware_error": d.hardware_error,
                "recording_buffers": d.recording_buffers,
                # id of the dataset build this device is working for ("" = none):
                # what the fleet list's Cancel button acts on.
                "dataset_job": ds_jobs.get(d.device_id, ""),
                "dataset_check": ds_jobs.get(d.device_id, "") in check_ids,
                "pending": len(d.queue),
                "history": [
                    {"type": c.type, "args": c.args, "result": c.result, "ts": c.done_at or c.created_at}
                    for c in d.history[:5]
                ],
            }
            for d in devices
        ],
    }


@app.delete("/api/fleet/devices/{device_id}")
async def remove_device(device_id: str, request: Request) -> dict[str, str]:
    owner = await _operator_loaded(request)
    fleet = _fleet_of(owner)
    if device_id not in fleet:
        raise HTTPException(404, "Device not found")
    del fleet[device_id]
    _remove_from_groups(owner, device_id)
    return {"status": "ok"}


@app.post("/api/fleet/devices/{device_id}/remove")
async def remove_device_graceful(device_id: str, request: Request) -> dict[str, str]:
    """Remove a device: dispatch logout if online (then delete on result), delete immediately if offline."""
    owner = await _operator_loaded(request)
    fleet = _fleet_of(owner)
    dev = fleet.get(device_id)
    if dev is None:
        raise HTTPException(404, "Device not found")
    if dev.online:
        dev.pending_delete = True
        cmd = Command(id=uuid.uuid4().hex[:12], type="logout", args={})
        _enqueue(dev, cmd)
    else:
        fleet.pop(device_id, None)
        _remove_from_groups(owner, device_id)
    return {"status": "ok"}


# === tasks (operator-facing, session OAuth) ==================================
class TaskReq(BaseModel):
    name: str = ""
    description: str = ""
    device_signature: list[str] = []


def _task_dict(t: Task) -> dict[str, Any]:
    return {"id": t.id, "name": t.name, "description": t.description,
            "device_signature": t.device_signature}


@app.get("/api/fleet/tasks")
async def list_tasks(request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    _reconcile_tasks(owner)  # surface tasks reported by connected devices
    agg = _reported_tasks(owner)
    reporters = _online_episode_reporters(owner)  # for repairing incomplete episodes
    out = []
    for t in _tasks_of(owner).values():
        d = _task_dict(t)
        eps = _episodes_from_entry(agg.get(t.name))
        sig = set(t.device_signature or [])
        for ep in eps:
            # Roles we actually know a device for. Incomplete = the task's expected
            # roles aren't all covered (or none is), e.g. episodes recorded before
            # the device stored its members. Fillable = a connected device that
            # reports this episode could supply a role we're missing.
            present = {r for r, w in ep["members"].items() if w.get("device_id")}
            incomplete = (bool(sig) and not sig.issubset(present)) or not present
            recon = reporters.get(ep["episode_id"], {})
            ep["incomplete"] = incomplete
            ep["fillable"] = incomplete and any(r not in present for r in recon)
        d["episodes"] = eps
        d["episode_count"] = len(eps)
        out.append(d)
    return {"tasks": out}


@app.post("/api/fleet/tasks/{task_id}/fill-devices")
async def fill_task_devices(task_id: str, request: Request) -> dict[str, Any]:
    """Repair ALL of a task's incomplete episodes at once (recorded before the
    device persisted their members): reconstruct role → device from the CONNECTED
    devices that still report each episode, then persist onto them so the fleet's
    view — and the devices themselves — become the durable source of truth.
    One batched command per device (not per episode). Partial by design: fills
    only the roles whose devices are online now, so a re-run completes the rest."""
    owner = await _operator_loaded(request)
    task = _tasks_of(owner).get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    eps = _episodes_from_entry(_reported_tasks(owner).get(task.name))
    sig = set(task.device_signature or [])
    # One pass: episode_id → online devices that still report it.
    ep_devs: dict[str, list[Device]] = {}
    for dev in _fleet_of(owner).values():
        if not dev.online:
            continue
        seen: set[str] = set()
        for t in dev.tasks or []:
            for eid, _m in _task_episode_items(t):
                if eid and eid not in seen:
                    seen.add(eid)
                    ep_devs.setdefault(eid, []).append(dev)
    # Build one batch of (episode_id, reconstructed members) per target device.
    batches: dict[str, dict] = {}  # device_id → {"dev": Device, "entries": [...]}
    filled: set[str] = set()
    for ep in eps:
        eid = ep["episode_id"]
        present = {r for r, w in ep["members"].items() if w.get("device_id")}
        incomplete = (bool(sig) and not sig.issubset(present)) or not present
        if not incomplete:
            continue
        devs = ep_devs.get(eid, [])
        recon: dict[str, dict] = {}
        for dev in devs:
            role = _device_slot(dev)
            if role:
                recon.setdefault(role, {"device_id": dev.device_id, "name": dev.name})
        if not any(r not in present for r in recon):  # nothing new we can add now
            continue
        filled.add(eid)
        for dev in devs:
            batches.setdefault(dev.device_id, {"dev": dev, "entries": []})["entries"].append(
                {"episode_id": eid, "members": recon})
    if not batches:
        raise HTTPException(409, {"message": "no incomplete episode can be repaired right now — "
                                             "connect the grabettes that recorded them, then retry"})
    for b in batches.values():
        cmd = Command(id=uuid.uuid4().hex[:12], type="set_episode_members",
                      args={"task_name": task.name, "episodes": b["entries"],
                            "device_signature": list(task.device_signature)})
        _enqueue(b["dev"], cmd)
    return {"status": "ok", "episodes": len(filled),
            "devices": [b["dev"].device_id for b in batches.values()]}


def _clean_signature(sig: list[str]) -> list[str]:
    bad = [s for s in sig if s not in VALID_SLOTS]
    if bad:
        raise HTTPException(400, f"Invalid device_signature entries: {bad}; allowed: {list(VALID_SLOTS)}")
    # de-dup, canonical order
    cleaned = [s for s in VALID_SLOTS if s in sig]
    if not cleaned:
        raise HTTPException(400, "A task must require at least one device")
    return cleaned


@app.post("/api/fleet/tasks")
async def create_task(req: TaskReq, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    if not req.name.strip():
        raise HTTPException(400, "Task needs a name")
    tasks = _tasks_of(owner)
    if any(t.name == req.name.strip() for t in tasks.values()):
        raise HTTPException(400, "A task with this name already exists")
    tid = uuid.uuid4().hex[:8]
    tasks[tid] = Task(id=tid, name=req.name.strip(), description=req.description.strip(),
                      device_signature=_clean_signature(req.device_signature))
    return {"status": "ok", "id": tid, "task": _task_dict(tasks[tid])}


@app.put("/api/fleet/tasks/{task_id}")
async def update_task(task_id: str, req: TaskReq, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    _reconcile_tasks(owner)
    t = _tasks_of(owner).get(task_id)
    if t is None:
        raise HTTPException(404, "Task not found")
    old_name = t.name
    new_name = req.name.strip() or old_name
    new_desc = req.description.strip()
    new_sig = _clean_signature(req.device_signature)

    # The device signature is locked ONCE it's known and episodes exist — the
    # recordings were made with that device set, so changing it would misdescribe
    # them. But an old task with episodes and NO recorded signature can be set
    # once (to backfill it); after that it's known → locked.
    has_episodes = bool(_episodes_from_entry(_reported_tasks(owner).get(old_name)))
    sig_locked = has_episodes and bool(t.device_signature)
    if sig_locked and sorted(new_sig) != sorted(t.device_signature):
        raise HTTPException(409, {"message": "cannot change the required devices of a task "
                                             "that already has recorded episodes"})
    if new_name != old_name and any(o.name == new_name for o in _tasks_of(owner).values()):
        raise HTTPException(400, "A task with this name already exists")

    sig_changed = (not sig_locked) and sorted(new_sig) != sorted(t.device_signature)

    # Apply on the fleet, then propagate to every device that recorded this task
    # (the source of truth) so the change survives the next report reconcile.
    t.name = new_name
    t.description = new_desc
    if not sig_locked:
        t.device_signature = new_sig
    for dev in _devices_reporting_task(owner, old_name):
        args = {"name": old_name,
                "new_name": new_name if new_name != old_name else None,
                "description": new_desc}
        if sig_changed:  # backfilling an old task's signature → devices must store it
            args["device_signature"] = new_sig
        _enqueue(dev, Command(id=uuid.uuid4().hex[:12], type="edit_task", args=args))
    if new_name != old_name:
        _suppress_task_name(owner, old_name)  # hide stale old-name reports until re-report
    return {"status": "ok", "task": _task_dict(t)}


@app.delete("/api/fleet/tasks/{task_id}")
async def delete_task(task_id: str, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    _reconcile_tasks(owner)
    tasks = _tasks_of(owner)
    t = tasks.get(task_id)
    if t is None:
        raise HTTPException(404, "Task not found")
    if any(s.status == "open" and s.task_id == task_id for s in _sessions_of(owner).values()):
        raise HTTPException(409, "Cannot delete a task with an open session")
    name = t.name
    # Delete on every ONLINE device that has this task — the task AND its recorded
    # episodes go (the device is the source of truth; removing here alone would let
    # the next report resurrect it). Offline members keep their copy for now; they
    # are reconciled when they reconnect (see orphan detection, phase 3).
    dispatched = []
    for dev in _devices_reporting_task(owner, name):
        _enqueue(dev, Command(id=uuid.uuid4().hex[:12], type="delete_task", args={"name": name}))
        dispatched.append(dev.device_id)
    del tasks[task_id]
    _suppress_task_name(owner, name)  # hide it until the devices re-report without it
    return {"status": "ok", "dispatched": dispatched}


class OrphanCleanupReq(BaseModel):
    episode_ids: list[str] = []


@app.get("/api/fleet/orphans")
async def list_orphans(request: Request) -> dict[str, Any]:
    """Episode reconciliation issues for the banner, from the cached snapshot
    computed on the last device register — never a per-poll recompute.

    Two kinds, served together on purpose: they come from one pass and belong in
    one banner, so surfacing the second costs the dashboard no extra request on
    its 3s poll (every one of those lands on the Space's single event loop).

      * groups — orphaned episodes (a peer deleted them, a connected device still
        holds them), grouped by task.
      * split  — episodes two online devices file under DIFFERENT tasks, grouped
        by the set of task names in disagreement, since a batch refile produces
        many episodes sharing the same divergence.
    """
    owner = await _operator_loaded(request)
    reported = _reported_tasks(owner)
    groups: dict[str, dict] = {}
    for o in ORPHANS_PENDING.get(owner, []):
        g = groups.setdefault(o["task"], {"task": o["task"], "episode_ids": [],
                                          "holders": {}, "deleted_by": {}})
        g["episode_ids"].append(o["episode_id"])
        for h in o["holders"]:
            g["holders"][h["device_id"]] = h["name"]
        for d in o["deleted_by"]:
            g["deleted_by"][d["device_id"]] = d["name"]
    out = []
    for g in groups.values():
        # If EVERY remaining episode of the task is orphaned, the whole task was
        # deleted on the peer(s) — say so, rather than just "episodes lost pairs".
        entry = reported.get(g["task"])
        total = len(entry["episodes"]) if entry else len(g["episode_ids"])
        out.append({"task": g["task"], "episode_ids": g["episode_ids"],
                    "count": len(g["episode_ids"]),
                    "whole_task": len(g["episode_ids"]) >= total,
                    "holders": list(g["holders"].values()),
                    "deleted_by": list(g["deleted_by"].values())})
    # Split filings, grouped by the disagreeing set of task names.
    sgroups: dict[tuple, dict] = {}
    for s in SPLIT_PENDING.get(owner, []):
        key = tuple(s["tasks"])
        g = sgroups.setdefault(key, {"tasks": s["tasks"], "episode_ids": [], "by_task": {}})
        g["episode_ids"].append(s["episode_id"])
        for f in s["filings"]:
            # Who files these under which name — the operator needs to know which
            # device to correct, and a batch shares the same answer. Both id and
            # name: the name is for reading, the id is what a repair dispatches to.
            g["by_task"].setdefault(f["task"], {})[f["device_id"]] = f["name"]
    split = [{"tasks": g["tasks"], "episode_ids": sorted(g["episode_ids"]),
              "count": len(g["episode_ids"]),
              "by_task": {t: [{"device_id": i, "name": n} for i, n in sorted(d.items())]
                          for t, d in g["by_task"].items()}}
             for g in sgroups.values()]
    return {"groups": out, "split": split}


@app.get("/api/fleet/unassigned")
async def list_unassigned(request: Request) -> dict[str, Any]:
    """Loose episodes per device — recordings made outside any task (a button
    press with no session open, or with the fleet unreachable).

    Read straight off the last report of each ONLINE device, the same rule the
    task aggregation follows: an offline device's inbox is a stale snapshot it may
    already have triaged locally, and nothing here should propose acting on it.

    Grouped by device, never merged: a loose episode was recorded alone, so it
    belongs to exactly one device — unlike tasks, which are merged by name across
    the fleet. `total` vs the length of `episodes` tells the caller when a device
    is reporting only its most recent takes (see UNASSIGNED_REPORT_LIMIT there),
    so the UI can say so instead of implying it listed everything.
    """
    owner = await _operator_loaded(request)
    out = []
    for dev in _fleet_of(owner).values():
        if not dev.online:
            continue
        inbox = dev.unassigned or {}
        episodes = inbox.get("episodes") or []
        if not episodes:
            continue
        out.append({
            "device_id": dev.device_id,
            "device": dev.name,
            "role": _device_slot(dev),
            "total": inbox.get("total", len(episodes)),
            # started_at is derived here, from the episode id, so the dashboard
            # renders these takes with the same timestamp helper it uses for task
            # episodes instead of parsing the id format in the browser.
            #
            # members are filled in from the reporting device's own slot when it
            # sent none — the same inference _online_episode_reporters makes for
            # task episodes, and true for the same reason: the device holding the
            # files contributed that role. Up-to-date devices now stamp this
            # themselves, so this only covers one on an older build; without it the
            # operator was told "device not recorded" about a take sitting on a
            # grabette whose role the fleet knew all along.
            "episodes": [dict(ep, started_at=_episode_started_at(ep.get("episode_id", "")),
                              members=(ep.get("members")
                                       or ({role: who} if (role := _device_slot(dev))
                                           and (who := {"device_id": dev.device_id,
                                                        "name": dev.name}) else {})))
                         for ep in episodes],
        })
    out.sort(key=lambda g: g["device"])
    return {"devices": out}


def _clean_episode_ids(ids: list[str]) -> list[str]:
    """Episode ids fit to dispatch: blanks — and anything that isn't a bare
    directory name — dropped, order kept, duplicates collapsed.

    A device turns an id into a path (episodes/<id>) and its deletions rmtree the
    result, so a blank id resolves to the episodes ROOT: one dispatched from here
    landed a phantom entry in a task's registry, and the next "delete task" on
    that device would have taken every episode with it. The device guards its own
    deletions too (see TaskManager._deletable_episode_dir), but the fleet is
    where these ids come from, so it is where they should stop.
    """
    out = []
    for eid in ids:
        eid = (eid or "").strip()
        if eid and eid not in (".", "..") and "/" not in eid and "\\" not in eid:
            out.append(eid)
    return list(dict.fromkeys(out))


class AssignReq(BaseModel):
    task_name: str
    device_ids: list[str] = []
    episode_ids: list[str] = []


def _known_episode_roles(owner: str, device_id: str, episode_id: str) -> dict[str, str]:
    """role → device_id for an episode as THIS device last reported it, whether it
    sits in a task or in the unassigned inbox (the two report channels)."""
    dev = _fleet_of(owner).get(device_id)
    if dev is None:
        return {}
    for ep in (dev.unassigned or {}).get("episodes") or []:
        if ep.get("episode_id") == episode_id:
            return {r: w.get("device_id") for r, w in (ep.get("members") or {}).items()}
    for t in dev.tasks or []:
        for eid, members in _task_episode_items(t):
            if eid == episode_id:
                return {r: w.get("device_id") for r, w in (members or {}).items()}
    return {}


@app.post("/api/fleet/episodes/assign")
async def assign_episodes(req: AssignReq, request: Request) -> dict[str, Any]:
    """File episodes under a task, by name, on the given devices.

    One endpoint for both callers: triaging the unassigned inbox, and repairing a
    split (dispatch the SAME episode ids to every disagreeing device so they all
    land on one name).

    Refuses when the target task has a known signature the episodes cannot satisfy
    — filing a mono take into a bimanual task creates an episode that is incomplete
    for ever, counted in the task yet unusable. The check lives here because the
    fleet is the only side that knows the target's signature; the devices stay
    mechanical, and their own local "Move to task" warns instead of refusing (see
    TaskManager.move_episodes) since reorganising both sides is legitimate.
    """
    owner = await _operator_loaded(request)
    name = req.task_name.strip()
    if not name:
        raise HTTPException(400, "task_name is required")
    episode_ids = _clean_episode_ids(req.episode_ids)
    if not episode_ids or not req.device_ids:
        raise HTTPException(400, "device_ids and episode_ids are required")
    fleet = _fleet_of(owner)
    offline = [d for d in req.device_ids if d not in fleet or not fleet[d].online]
    if offline:
        raise HTTPException(409, {"message": "these devices are offline", "devices": offline})

    _reconcile_tasks(owner)
    target = next((t for t in _tasks_of(owner).values() if t.name == name), None)
    sig = set(target.device_signature) if target and target.device_signature else set()
    if sig:
        for device_id in req.device_ids:
            for eid in episode_ids:
                roles = {r for r, d in _known_episode_roles(owner, device_id, eid).items() if d}
                if roles and not sig.issubset(roles):
                    raise HTTPException(409, {
                        "message": f"“{name}” requires {sorted(sig)}, but episode {eid} "
                                   f"was recorded with {sorted(roles)} — filing it there "
                                   "would leave it permanently incomplete",
                        "episode_id": eid, "required": sorted(sig), "got": sorted(roles)})

    dispatched = []
    for device_id in req.device_ids:
        _enqueue(fleet[device_id], Command(
            id=uuid.uuid4().hex[:12], type="assign_episodes",
            args={"task_name": name, "episode_ids": episode_ids}))
        dispatched.append(device_id)
    # Clear the affected episodes from the cached snapshots so the banners settle at
    # once; the devices' re-register confirms the real state within a beat.
    if owner in SPLIT_PENDING:
        SPLIT_PENDING[owner] = [s for s in SPLIT_PENDING[owner]
                                if s["episode_id"] not in set(episode_ids)]
    return {"status": "ok", "task": name, "devices": dispatched,
            "episodes": len(episode_ids)}


class EpisodeDeleteReq(BaseModel):
    device_ids: list[str] = []
    episode_ids: list[str] = []


@app.post("/api/fleet/episodes/delete")
async def delete_episodes(req: EpisodeDeleteReq, request: Request) -> dict[str, Any]:
    """Discard episodes on the given devices — files and registry entry.

    The triage counterpart of assigning: a 0.8s misfire has no task to belong to,
    and leaving it in the inbox for ever is not a resolution. Deliberately separate
    from the orphan cleanup, which computes its own target list from a detected
    inconsistency; here the operator names exactly what to drop.

    One batched command per device (the device handler takes episode_ids), so a
    fifty-take selection doesn't queue fifty commands through a serial worker.
    """
    owner = await _operator_loaded(request)
    episode_ids = _clean_episode_ids(req.episode_ids)
    if not episode_ids or not req.device_ids:
        raise HTTPException(400, "device_ids and episode_ids are required")
    fleet = _fleet_of(owner)
    offline = [d for d in req.device_ids if d not in fleet or not fleet[d].online]
    if offline:
        raise HTTPException(409, {"message": "these devices are offline", "devices": offline})
    for device_id in req.device_ids:
        _enqueue(fleet[device_id], Command(
            id=uuid.uuid4().hex[:12], type="delete_episode",
            args={"episode_ids": episode_ids}))
    # Settle the banners at once; the devices' re-register confirms within a beat.
    dropped = set(episode_ids)
    for store in (ORPHANS_PENDING, SPLIT_PENDING):
        if owner in store:
            store[owner] = [x for x in store[owner] if x["episode_id"] not in dropped]
    return {"status": "ok", "devices": list(req.device_ids), "episodes": len(dropped)}


@app.post("/api/fleet/orphans/cleanup")
async def cleanup_orphans(req: OrphanCleanupReq, request: Request) -> dict[str, Any]:
    """Delete the given orphaned episodes on the devices still holding them.
    Recomputes orphans so only genuinely-orphaned episodes are touched."""
    owner = await _operator_loaded(request)
    # Recompute live here (accuracy matters for a destructive action).
    orphans = {o["episode_id"]: o for o in _reconcile_episodes(owner)["orphans"]}
    fleet = _fleet_of(owner)
    dispatched = 0
    cleaned = set()
    for eid in req.episode_ids:
        o = orphans.get(eid)
        if not o:
            continue
        for h in o["holders"]:
            dev = fleet.get(h["device_id"])
            if dev is not None and dev.online:
                _enqueue(dev, Command(id=uuid.uuid4().hex[:12], type="delete_episode",
                                      args={"episode_id": eid}))
                dispatched += 1
                cleaned.add(eid)
    # Drop the cleaned episodes from the cached snapshot so the banner clears at
    # once; the holders' re-register will confirm on the next report. Split
    # filings are pruned on the same episode ids: deleting a copy also settles any
    # disagreement about where that copy belonged.
    if cleaned:
        if owner in ORPHANS_PENDING:
            ORPHANS_PENDING[owner] = [o for o in ORPHANS_PENDING[owner]
                                      if o["episode_id"] not in cleaned]
        if owner in SPLIT_PENDING:
            SPLIT_PENDING[owner] = [s for s in SPLIT_PENDING[owner]
                                    if s["episode_id"] not in cleaned]
    return {"status": "ok", "dispatched": dispatched}


# === device groups (operator-facing, session OAuth) ==========================
class GroupReq(BaseModel):
    name: str = ""
    left: str = ""
    right: str = ""
    casquette: str = ""


def _validate_group_members(owner: str, req: GroupReq, group_id: Optional[str] = None) -> None:
    fleet = _fleet_of(owner)
    groups = _groups_of(owner)
    slots = {"left": req.left, "right": req.right, "casquette": req.casquette}
    if not any(slots.values()):
        raise HTTPException(400, "Group needs at least one device")
    for slot, device_id in slots.items():
        if not device_id:
            continue
        dev = fleet.get(device_id)
        if dev is None:
            raise HTTPException(404, f"Device {device_id} not found")
        if slot in ("left", "right"):
            if kind_of(dev) != "grabette" or dev.hand != slot:
                raise HTTPException(400, f"Device {device_id} is not a {slot}-hand grabette")
        elif kind_of(dev) != "casquette":
            raise HTTPException(400, f"Device {device_id} is not a casquette")
        for gid, g in groups.items():
            if gid == group_id:
                continue
            if device_id in (g.left, g.right, g.casquette):
                raise HTTPException(400, f"Device {device_id} is already in another group")


@app.get("/api/fleet/groups")
async def list_groups(request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    groups = list(_groups_of(owner).values())
    return {
        "groups": [
            {"id": g.id, "name": g.name, "left": g.left, "right": g.right, "casquette": g.casquette}
            for g in groups
        ]
    }


@app.post("/api/fleet/groups")
async def create_group(req: GroupReq, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    _validate_group_members(owner, req)
    groups = _groups_of(owner)
    gid = uuid.uuid4().hex[:8]
    groups[gid] = Group(
        id=gid, name=req.name.strip() or f"Group {len(groups) + 1}", left=req.left, right=req.right, casquette=req.casquette,
    )
    return {"status": "ok", "id": gid}


@app.put("/api/fleet/groups/{group_id}")
async def update_group(group_id: str, req: GroupReq, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    groups = _groups_of(owner)
    g = groups.get(group_id)
    if g is None:
        raise HTTPException(404, "Group not found")
    _validate_group_members(owner, req, group_id=group_id)
    g.name = req.name.strip() or g.name
    g.left, g.right, g.casquette = req.left, req.right, req.casquette
    return {"status": "ok"}


@app.delete("/api/fleet/groups/{group_id}")
async def delete_group(group_id: str, request: Request) -> dict[str, str]:
    owner = await _operator_loaded(request)
    groups = _groups_of(owner)
    if group_id not in groups:
        raise HTTPException(404, "Group not found")
    del groups[group_id]
    return {"status": "ok"}


def _group_members(g: Group) -> list[str]:
    return [d for d in (g.left, g.right, g.casquette) if d]


def _lead_for(session: Session) -> float:
    """Pick the start lead for the next episode: short if the OAK-D is still
    warm (a recent episode stopped within OAK_WARM_WINDOW_S), long otherwise
    (first episode of the session, or the OAK-D has since gone to sleep) so the
    cold boot finishes before T0. Conservative: unknown/old → cold."""
    if session.last_stop_at is not None and (time.time() - session.last_stop_at) < OAK_WARM_WINDOW_S:
        return GROUP_START_LEAD_WARM_S
    return GROUP_START_LEAD_COLD_S


def _named_members(owner: str, members: dict[str, str]) -> dict[str, dict[str, str]]:
    """role → {device_id, name}. Sent to devices so each records who its peers
    were (by stable device_id + display name), letting a device name an offline
    peer later — the device is the durable source of truth for episode membership."""
    fleet = _fleet_of(owner)
    return {
        r: {"device_id": d, "name": (fleet[d].name if d in fleet else d)}
        for r, d in members.items()
    }


def _schedule_episode_start(
    owner: str, task_name: str, members: dict[str, str], lead_s: float,
    exclude_device_id: Optional[str] = None, signature: Optional[list[str]] = None,
    session: Optional["Session"] = None,
) -> str:
    """Enqueue a synchronized start_capture to every member device (except
    exclude_device_id, which self-schedules from the returned T0). Raises 409
    if any member is offline. Returns the shared start_at_utc (ISO).

    Each command carries the episode's full membership (role → device_id+name)
    and the task's device signature so every device persists who recorded with
    it — the fleet no longer needs its own HF-backed record of this."""
    fleet = _fleet_of(owner)
    device_ids = list(members.values())
    offline = [d for d in device_ids if d not in fleet or not fleet[d].online]
    if offline:
        raise HTTPException(409, {"message": "one or more devices are offline", "offline": offline})
    # A device in a hardware fault REFUSES the start locally. Dispatching anyway
    # would let its peers record a take that is unusable the moment it is missing
    # an arm — and this endpoint never looks at the results, so nobody would find
    # out until the dataset build counted the episode as incomplete. Refuse here,
    # naming the device and the fault, while the take can still be saved.
    faulted = [{"device_id": d, "name": fleet[d].name, "error": fleet[d].hardware_error}
               for d in device_ids if fleet[d].hardware_error]
    if faulted:
        raise HTTPException(409, {
            "message": "one or more devices cannot record",
            "faulted": faulted})

    # A new episode starts from a clean slate: last take's start failures are
    # history. Cleared HERE rather than at each call site so the two entry points
    # (operator button, physical button) can't drift apart on it.
    if session is not None:
        session.start_errors.clear()

    named = _named_members(owner, members)
    target_iso = (datetime.now(timezone.utc) + timedelta(seconds=lead_s)).isoformat()
    for device_id in device_ids:
        if device_id == exclude_device_id:
            continue
        cmd = Command(
            id=uuid.uuid4().hex[:12], type="start_capture",
            args={"task_name": task_name, "start_at_utc": target_iso,
                  "members": named, "signature": signature or []},
        )
        _enqueue(fleet[device_id], cmd)
    return target_iso


def _dispatch_episode_stop(owner: str, members: dict[str, str],
                           exclude_device_id: Optional[str] = None) -> dict[str, str]:
    """Fan out an IMMEDIATE stop to every member device except exclude_device_id
    (the acting device, which has already stopped locally). Each peer stops as
    soon as it receives the command, i.e. within ~1 poll interval — the pressed
    device is instant and peers trail only by the short delivery latency, no
    scheduled lead. Chaining fast stays safe because the next episode's start
    still uses a lead (GROUP_START_LEAD_*) that the peer's stop+mux fits inside.

    Returns {command_id: device_id} for the dispatched stops so the caller can wait
    on their results (see Session.pending_stop_cmds) and end the "stopping" phase
    exactly when the devices report done, rather than after a fixed guess. Each of
    these commands is retryable, so a stop whose delivery is lost is re-sent until
    the device acknowledges it (see RETRYABLE_CMDS / _rearm_unacked) — a peer can
    no longer be left recording by a single dropped response.

    (A synchronized, lead-based stop — shared future T_stop via GROUP_STOP_LEAD_S
    + the device-side CaptureScheduler.schedule_stop path — remains wired up but
    unused here; it'd matter only if delivery latency grew. With long-polling
    delivery drops to a network round-trip, making even this spread negligible.)"""
    fleet = _fleet_of(owner)
    ids: dict[str, str] = {}
    for device_id in members.values():
        if device_id == exclude_device_id:
            continue
        dev = fleet.get(device_id)
        if dev is None:
            continue
        # A device never needs more than ONE outstanding stop: the command is
        # idempotent, so a second copy buys nothing — and since retryable commands
        # now survive the stale-queue purge, copies would pile up in the queue of a
        # device that never comes back (operator pressing stop again, close_session
        # re-stopping…). Re-arm the one already queued instead, resetting its retry
        # budget: an explicit re-dispatch is a fresh attempt, not a continuation.
        cmd = next((c for c in dev.queue if c.type == "stop_capture"), None)
        if cmd is not None:
            cmd.status = "pending"
            cmd.attempts = 0
            dev.wakeup.set()
        else:
            cmd = Command(id=uuid.uuid4().hex[:12], type="stop_capture", args={})
            _enqueue(dev, cmd)
        ids[cmd.id] = device_id
    return ids


def _begin_episode_stop(s: "Session", stop_cmd_ids: dict[str, str]) -> None:
    """Mark a session as no longer recording and enter the 'stopping' phase,
    awaiting the dispatched stops' results (see Session.stopping). Only claims
    'stopping' when there are commands to wait on — a stop with no tracked device
    (e.g. a solo device that stopped locally) goes straight to idle."""
    s.recording = False
    s.last_stop_at = time.time()  # also drives the warm/cold lead for the next episode
    s.pending_stop_cmds = dict(stop_cmd_ids)
    s.stopping = bool(stop_cmd_ids)
    # The episode is over, so "idle" is now the expected report from every member
    # — start the lost-stop watch from scratch for the next one.
    s.idle_since.clear()


# --- lost-stop safety net ----------------------------------------------------
# A stop fan-out can be lost BEFORE it ever reaches the fleet: the acting device
# POSTs /sync/stop once, best-effort, and if that call fails (slow Space, WiFi
# blip, device reboot) the peers are never told and the session stays "recording"
# forever. None of the machinery above protects against this — RETRYABLE_CMDS,
# _rearm_unacked and pending_stop_cmds only cover stops that were DISPATCHED.
#
# The heartbeat already carries each device's self-reported activity, so the
# fleet can see the contradiction for free: a member reports "idle" while its
# session believes it is recording ⇒ that device stopped without the fleet ever
# hearing about it. Fan the stop out to its peers instead of leaving them rolling.

# Grace after the episode's T0 before an "idle" report means anything. Covers the
# pre-T0 warm-up — every device is legitimately idle until the scheduled start
# fires — plus up to one stale heartbeat (DEVICE_HEARTBEAT_S).
LOST_STOP_GRACE_S = 15.0
# How long "idle" must persist before acting. At least two heartbeats, so a single
# stale or racing report can never cut a live episode short.
LOST_STOP_CONFIRM_S = 12.0


def _episode_t0_epoch(s: "Session") -> Optional[float]:
    """Epoch of the current episode's shared T0, or None if it can't be read."""
    ep = s.episodes[-1] if s.episodes else None
    iso = (ep or {}).get("start_at_utc")
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError):
        return None


def _reconcile_lost_stop(owner: str, dev: Device) -> None:
    """Spot a stop whose fan-out never reached the fleet, and dispatch it now.

    Called on every heartbeat: no extra traffic, no background loop, and at most
    one open session per owner to inspect. Deliberately reads dev.reported_status
    — the device's OWN view — and never _device_activity(), whose fallback derives
    from the session's own recording flag and would therefore always agree with the
    very session it is supposed to contradict.
    """
    s = _open_session_for_device(owner, dev.device_id)
    if s is None or not s.recording:
        return
    if (dev.reported_status or "").strip().lower() != "idle":
        s.idle_since.pop(dev.device_id, None)
        return
    # Before T0 + grace, idle is the NORMAL report (devices are warming up for the
    # scheduled start), so it carries no information yet.
    t0 = _episode_t0_epoch(s)
    now = time.time()
    if t0 is None or now < t0 + LOST_STOP_GRACE_S:
        return
    first_idle = s.idle_since.setdefault(dev.device_id, now)
    if now - first_idle < LOST_STOP_CONFIRM_S:
        return
    peers = [d for d in s.members.values() if d != dev.device_id]
    logger.warning(
        "session %s: device %s has reported idle for %.0fs while the session is still "
        "recording — its stop never reached the fleet. Stopping %d peer(s) now.",
        s.id, dev.device_id, now - first_idle, len(peers),
    )
    # Exclude the reporting device: it has already stopped, that's the whole
    # premise. With no peers left (solo session) _begin_episode_stop still clears
    # the stale "recording" flag, so the UI stops lying either way.
    ids = _dispatch_episode_stop(owner, s.members, exclude_device_id=dev.device_id)
    _begin_episode_stop(s, ids)


def _episode_id_for_target(target_iso: str) -> str:
    """Mirror the device's episode_id_for(T0): same UTC second → same id, so
    the manifest entry matches the folder every device actually creates."""
    return datetime.fromisoformat(target_iso).astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")


# === sessions (operator-facing, session OAuth) ===============================
# The recording unit. Launch = pick a task + a target (a group, or a single
# device treated as a group of one), validate the target's roles match the
# task's required devices, then record one or more synchronized episodes into
# the session's manifest, and close it.
class SessionLaunchReq(BaseModel):
    task_id: str
    # A device id per role, picked from the fleet at launch time. The device
    # selection IS the grouping — no Group object is created (the session owns
    # its members map).
    left: str = ""
    right: str = ""
    casquette: str = ""


def _resolve_members(owner: str, req: SessionLaunchReq) -> dict[str, str]:
    """Build the session's role→device map from the per-role device ids,
    checking each device actually fills the role it was placed in."""
    fleet = _fleet_of(owner)
    members: dict[str, str] = {}
    for role, device_id in (("left", req.left), ("right", req.right), ("casquette", req.casquette)):
        if not device_id:
            continue
        dev = fleet.get(device_id)
        if dev is None:
            raise HTTPException(404, f"Device {device_id} not found")
        if _device_slot(dev) != role:
            raise HTTPException(400, f"Device {device_id} cannot fill the {role} role")
        if not dev.online:
            # A session can't be launched with an offline device — it couldn't
            # receive the synchronized start and would fail at the first episode.
            raise HTTPException(409, {"message": f"{dev.name} is offline", "offline": [device_id]})
        members[role] = device_id
    if not members:
        raise HTTPException(400, "Select at least one device")
    return members


# Safety cap: if a device never reports its stop result (e.g. it dropped offline
# mid-stop), don't leave the "stopping" phase stuck — expose it as ended after this.
STOP_PHASE_MAX_S = 30.0


def _session_dict(owner: str, s: Session) -> dict[str, Any]:
    fleet = _fleet_of(owner)
    def dname(d):
        dev = fleet.get(d)
        return dev.name if dev else d
    # "stopping" ends when every dispatched stop reported (s.stopping cleared);
    # the time cap is only a fallback for a device that never reports.
    capped = s.stopping and s.last_stop_at is not None and (time.time() - s.last_stop_at) >= STOP_PHASE_MAX_S
    stopping = s.stopping and not capped
    # Past the cap with stops still unaccounted for, the fleet does NOT get to
    # pretend the session is idle: a device that never acknowledged its stop may
    # still be recording, out of sync with the rest. Name it so the operator can
    # act (and re-send the stop) instead of discovering it on the device later.
    unconfirmed = [{"device_id": d, "name": dname(d)}
                   for d in dict.fromkeys(s.pending_stop_cmds.values())] if capped else []
    return {
        "id": s.id, "status": s.status, "task_id": s.task_id,
        "task_name": _task_name(owner, s.task_id),
        "recording": s.recording,
        "stopping": stopping,  # accurate "stopping" phase — ends when the devices really stop
        # Devices whose stop_capture was never acknowledged (see above). Non-empty
        # ⇒ the UI warns and offers to re-send the stop.
        "stop_unconfirmed": unconfirmed,
        # Devices whose start_capture came back failed for the current episode:
        # they are NOT recording while their peers are. Non-empty ⇒ the UI says
        # so, because the fleet's start dispatch never waits for these results
        # and nothing else would ever surface them.
        "start_errors": [{"device_id": d, "name": dname(d), "error": e}
                         for d, e in s.start_errors.items()],
        "started_at": s.started_at,
        "members": {r: {"device_id": d, "name": dname(d)} for r, d in s.members.items()},
        "episode_count": len(s.episodes),
        "episodes": s.episodes,
    }


@app.get("/api/fleet/sessions")
async def list_sessions(request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    return {"sessions": [_session_dict(owner, s) for s in _sessions_of(owner).values()]}


@app.post("/api/fleet/sessions")
async def launch_session(req: SessionLaunchReq, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    task = _tasks_of(owner).get(req.task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    members = _resolve_members(owner, req)
    # The target must provide EXACTLY the roles the task requires.
    _validate_task_signature(owner, req.task_id, set(members.keys()))
    # A single open session at a time (fleet-wide). Launching is done from the
    # task list; the running one must be closed before another can start.
    if any(s.status == "open" for s in _sessions_of(owner).values()):
        raise HTTPException(409, {"message": "a session is already running; close it first"})
    # A device tied up by a dataset upload/conversion can't also record.
    _raise_if_recording_blocked(owner, members.values())
    sid = uuid.uuid4().hex[:8]
    _sessions_of(owner)[sid] = Session(id=sid, task_id=req.task_id, members=members)
    # No warm-up on launch: keeping the OAK-D on until the first episode would
    # drain the battery if the operator waits. Instead the first episode uses
    # the COLD lead (long enough for the scheduler to warm the OAK-D during the
    # lead), and after that the device keepalive keeps it warm between episodes.
    return {"status": "ok", "id": sid, "session": _session_dict(owner, _sessions_of(owner)[sid])}


def _get_open_session(owner: str, session_id: str) -> Session:
    s = _sessions_of(owner).get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    if s.status != "open":
        raise HTTPException(409, "Session is closed")
    return s


@app.post("/api/fleet/sessions/{session_id}/episode/start")
async def session_episode_start(session_id: str, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    s = _get_open_session(owner, session_id)
    # A member pulled into a dataset upload/conversion can't start a new episode.
    _raise_if_recording_blocked(owner, s.members.values())
    _task = _tasks_of(owner).get(s.task_id)
    target_iso = _schedule_episode_start(
        owner, _task_name(owner, s.task_id), s.members, _lead_for(s),
        signature=_task.device_signature if _task else [], session=s,
    )
    episode_id = _episode_id_for_target(target_iso)
    # Record the manifest entry up front from the deterministic id; per-device
    # success is reconciled later via /api/devices/result (phase 3). start_at_utc
    # is the shared T0 — the UI shows "initializing" until then, "recording" after.
    s.episodes.append({"episode_id": episode_id, "roles": dict(s.members),
                       "started_at": datetime.now(timezone.utc).isoformat(),
                       "start_at_utc": target_iso})
    s.recording = True
    return {"status": "scheduled", "scheduled_start_utc": target_iso, "episode_id": episode_id,
            "devices": list(s.members.values())}


@app.post("/api/fleet/sessions/{session_id}/episode/stop")
async def session_episode_stop(session_id: str, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    s = _get_open_session(owner, session_id)
    ids = _dispatch_episode_stop(owner, s.members)
    _begin_episode_stop(s, ids)
    return {"status": "ok", "devices": list(s.members.values())}


@app.post("/api/fleet/sessions/{session_id}/episode/delete-last")
async def session_delete_last_episode(session_id: str, request: Request) -> dict[str, Any]:
    """Drop the session's most recent episode from the manifest and tell each
    device that recorded it to delete its local files. Refused while recording
    (the last manifest entry is then the in-progress episode)."""
    owner = await _operator_loaded(request)
    s = _get_open_session(owner, session_id)
    if s.recording:
        raise HTTPException(409, {"message": "stop the current episode before deleting"})
    if not s.episodes:
        raise HTTPException(409, {"message": "no episode to delete"})
    ep = s.episodes.pop()  # most recent
    episode_id = ep["episode_id"]
    fleet = _fleet_of(owner)
    devices = list((ep.get("roles") or {}).values())
    for dev_id in devices:
        dev = fleet.get(dev_id)
        if dev is not None:
            _enqueue(dev, Command(id=uuid.uuid4().hex[:12], type="delete_episode",
                                  args={"episode_id": episode_id}))
    return {"status": "ok", "episode_id": episode_id, "devices": devices}


@app.post("/api/fleet/sessions/{session_id}/close")
async def close_session(session_id: str, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    s = _sessions_of(owner).get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    # Best-effort: make sure nothing is left recording on the members.
    _dispatch_episode_stop(owner, s.members)
    if not s.episodes:
        # A session that recorded nothing isn't worth keeping — drop it so it
        # never clutters the task's history.
        del _sessions_of(owner)[session_id]
        return {"status": "ok", "discarded": True}
    s.recording = False
    s.last_stop_at = time.time()
    s.status = "closed"
    return {"status": "ok"}


@app.delete("/api/fleet/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict[str, str]:
    owner = await _operator_loaded(request)
    if session_id not in _sessions_of(owner):
        raise HTTPException(404, "Session not found")
    del _sessions_of(owner)[session_id]
    return {"status": "ok"}


# === LeRobot dataset generation (operator OAuth) ============================
# Selected tasks (all sharing one device signature) → each involved device
# pushes its own streams to a shared raw dataset (by role) → a processing Space
# converts raw → LeRobot. The fleet only orchestrates; devices upload with their
# own write tokens (the fleet can't reach them, and needs no token for uploads).
class DatasetReq(BaseModel):
    task_ids: list[str]
    name: str = ""  # target dataset name; bare name → {owner}/{name}
    private: bool = False  # create the resulting LeRobot dataset as private
    # If True, include only episodes whose EVERY role-device is currently online
    # (skip episodes that reference an offline device) instead of requiring the
    # whole set of devices to be up.
    only_available: bool = False
    # "Use only" advanced option: a subset of roles (e.g. ["left"] or
    # ["left","casquette"]). When set, ANY task whose signature is a superset is
    # eligible, and only these roles' devices are uploaded → build a dataset from
    # just this sub-combination. Empty → use each selected task's full signature.
    roles: list[str] = []
    # Per-task episode picking ("advanced selection"): the episodes the operator
    # kept, across all the selected tasks. Ids are unique (a UTC timestamp), so one
    # flat allow-list says it without a task→episodes map. EMPTY MEANS EVERY
    # episode of the selected tasks — the default, and what every caller that
    # doesn't pick sends. A build restricted to nothing is therefore not
    # expressible here, by design: the dashboard disables the button instead (see
    # dsCanGen), and an empty plan is refused below either way.
    episode_ids: list[str] = []


def _command_status(dev: Device, cmd_id: str) -> Optional[Command]:
    """Find a dispatched command by id, whether still queued or completed
    (results move a command from queue to history)."""
    for c in list(dev.queue) + list(dev.history):
        if c.id == cmd_id:
            return c
    return None


def _resolve_dataset_plan(owner: str, task_ids: list[str], only_available: bool = False,
                          roles_override: Optional[list[str]] = None,
                          only_episodes: Optional[set[str]] = None,
                          ) -> tuple[list[str], dict[str, dict], dict[str, Any]]:
    """Validate the selection and build the per-device upload plan from the
    device reports. Returns (roles, plan, report) where plan maps device_id ->
    {"role": role, "episode_ids": set[str]} and report accounts for every episode
    left out: {"included": int, "incomplete": [...], "unavailable": [...]}.

    An episode is only usable when EVERY requested role has a device to upload it.
    Partial ones used to slip through — the test was merely "has at least one of
    the requested roles" — so a bimanual dataset could receive a take carrying only
    the left stream, and nothing said so. They are now excluded AND counted, since
    a dataset quietly built from 17 of 20 episodes is worse than one that says so.

    roles_override ("Use only"): a subset of roles. When given, every selected
    task must merely CONTAIN these roles (superset), and only these roles' devices
    are uploaded — so you can build, say, a left-only dataset from bimanual tasks.
    When absent, the selected tasks must share one signature and all its roles are
    used.

    only_available: skip episodes that reference a device which is currently
    offline — the resulting dataset covers only what can be fully uploaded now.

    only_episodes: keep just these episode ids (the operator's per-task picking).
    None = no restriction. Excluded ids are NOT reported as skipped: `skipped`
    accounts for what the FLEET dropped on its own, and that accounting is what
    the job message shows the operator — folding their own choice into it would
    read as a fault."""
    _reconcile_tasks(owner)
    tasks = _tasks_of(owner)
    sel = [tasks[t] for t in task_ids if t in tasks]
    if not sel:
        raise HTTPException(404, "no matching tasks")
    override = [r for r in (roles_override or []) if r in VALID_SLOTS]
    if override:
        oset = set(override)
        bad = [t.name for t in sel if not oset.issubset(set(t.device_signature))]
        if bad:
            raise HTTPException(409, {"message": "some selected tasks don't have the chosen devices",
                                      "tasks": bad})
        roles = sorted(oset)
    else:
        sigs = {tuple(sorted(t.device_signature)) for t in sel}
        if len(sigs) != 1:
            raise HTTPException(409, {"message": "selected tasks must share the same device signature"})
        roles = sorted(sel[0].device_signature)
    agg = _reported_tasks(owner)
    fleet = _fleet_of(owner)

    def _online(d):
        return d in fleet and fleet[d].online

    # Episodes come from the devices' reports (the source of truth), matched by
    # task name. members' device_id per role tells us which device to ask to
    # upload that role's data for each episode. With an override, restrict to
    # just those roles so only the requested devices upload.
    role_set = set(roles)
    plan: dict[str, dict] = {}
    included: set[str] = set()
    incomplete: set[str] = set()
    unavailable: set[str] = set()
    for t in sel:
        for ep in _episodes_from_entry(agg.get(t.name)):
            eid = ep["episode_id"]
            if only_episodes is not None and eid not in only_episodes:
                continue  # left out by the operator — see only_episodes
            # A role with no device_id counts as missing: nobody can upload it.
            roles_map = {r: d for r, d in ep["roles"].items() if r in role_set and d}
            if not role_set.issubset(roles_map):
                incomplete.add(eid)
                continue
            if only_available and not all(_online(d) for d in roles_map.values()):
                unavailable.add(eid)
                continue
            included.add(eid)
            for role, dev_id in roles_map.items():
                p = plan.setdefault(dev_id, {"role": role, "episode_ids": set()})
                p["episode_ids"].add(eid)
    if not plan:
        if unavailable:
            msg = "no episodes have all their devices online"
        elif incomplete:
            # Distinguished from "nothing recorded": the episodes exist, they were
            # just recorded with fewer devices than this dataset needs.
            msg = (f"none of the {len(incomplete)} recorded episode(s) was recorded with "
                   f"all of {roles}")
        else:
            msg = "selected tasks have no recorded episodes"
        raise HTTPException(409, {"message": msg, "incomplete": sorted(incomplete),
                                  "unavailable": sorted(unavailable)})
    return roles, plan, {"included": len(included), "incomplete": sorted(incomplete),
                         "unavailable": sorted(unavailable)}


def _slam_check_repo(owner: str, task_name: str, stamp: str) -> str:
    """Target repo for a check: {task}_trajectorycheck_{YYYYmmdd_HHMMSS}.

    The marker is not decoration — it is HOW the Space is told this run is a check
    (see SLAM_CHECK_MARKER). The task and the timestamp are what make the dataset a
    flagged check pushes identifiable on the Hub, and the stamp is UTC like the
    episode ids, so a dataset and the takes inside it read on the same clock."""
    slug = re.sub(r"[^a-z0-9]+", "-", task_name.lower()).strip("-")[:32] or "task"
    return f"{owner}/{slug}{SLAM_CHECK_MARKER}{stamp}"


def _episode_unavailable(reported: dict[str, set[str]], device_id: str, episode_id: str) -> bool:
    """Can this device NOT hand over this episode? Two cases only: we hold no
    report for it at all (offline — it can't upload anything), or it has since
    reported a NEWER episode while still not listing this one, which means the
    episode really is gone (deleted or refiled) rather than merely not reported
    yet. Episode ids are UTC 'YYYYmmdd_HHMMSS' stamps, so comparing them as
    strings compares them in time."""
    ids = reported.get(device_id)
    if ids is None:
        return True
    if episode_id in ids:
        return False
    return bool(ids) and max(ids) > episode_id


def _slam_check_plan(owner: str, s: Session, count: int,
                     ) -> tuple[list[str], dict[str, dict], list[str], list[str]]:
    """The session's last `count` finished episodes, and the per-device upload plan
    for them: (roles, plan, episode_ids, skipped) — plan being device_id ->
    {"role", "episode_ids"}, the same shape _resolve_dataset_plan returns, so the
    existing dataset machinery carries a check without a second code path.

    `skipped` names the manifest episodes passed over on the way to filling the
    count, and it exists because of a real failure: a session can hold a manifest
    entry no device has any data for (an entry is appended when a start is
    DISPATCHED, so a start that never produced a recording leaves one behind).
    Skipping it is right; skipping it silently made a check look like it had
    ignored the take the operator just recorded.

    The session MANIFEST is the authority on which episodes belong to this run and
    which device held each role. A device's report can only VETO an episode, and
    only on a positive contradiction (see _episode_unavailable).

    That asymmetry is the whole point. The fleet learns a device's episode list
    only when the device re-registers, which happens a heartbeat AFTER the episode
    is muxed — so for several seconds after a stop the take exists, the manifest
    knows it, and the report does not. Reading that absence as "the device hasn't
    got it" is what made a check silently run on the three takes recorded BEFORE
    the one the operator had just finished: the answer looked fine and was about
    the wrong episodes, which is worse than no check at all."""
    reported: dict[str, set[str]] = {}  # ONLINE device -> every episode id it reports
    for dev in _fleet_of(owner).values():
        if dev.online:
            reported[dev.device_id] = {eid for t in (dev.tasks or [])
                                       for eid, _members in _task_episode_items(t)}
    roles = sorted(s.members)
    # Newest first, and never the in-progress take: while the session is recording,
    # the last manifest entry is an episode whose data isn't muxed on the device yet.
    entries = s.episodes[:-1] if s.recording else s.episodes
    picked: list[tuple[str, dict[str, str]]] = []
    skipped: list[str] = []
    for ep in reversed(entries):
        eid = str(ep.get("episode_id") or "")
        got: dict[str, str] = {r: d for r, d in (ep.get("roles") or {}).items()
                               if r in roles and d}
        if not eid:
            continue
        if set(got) != set(roles):
            skipped.append(eid)  # not recorded with the session's full set of roles
            continue
        if any(_episode_unavailable(reported, d, eid) for d in got.values()):
            skipped.append(eid)  # no device holds it (or its device is offline)
            continue
        picked.append((eid, got))
        if len(picked) >= count:
            break
    if not picked:
        raise HTTPException(409, {"message": "none of this session's episodes is complete on its "
                                             "devices right now — nothing to check"})
    picked.reverse()  # chronological, so the report reads in recording order
    skipped.reverse()
    plan: dict[str, dict] = {}
    for eid, got in picked:
        for role, dev_id in got.items():
            p = plan.setdefault(dev_id, {"role": role, "episode_ids": set()})
            p["episode_ids"].add(eid)
    return roles, plan, [eid for eid, _ in picked], skipped


def _why_not_ok(res: dict, fallback: str) -> str:
    """Why a device's command result is not ok — never empty.

    A MISSING message is one thing; a message that is present but EMPTY is what
    actually bit us: dict.get(key, default) keeps the empty string, so a device-side
    timeout (str(TimeoutError()) is "") reached the operator as "Failed:" with
    nothing after it. Devices no longer send empty messages, but the fleet must not
    depend on that — a device in the field can be running an older build. Falling
    back on the reported status also distinguishes an error from a cancel."""
    msg = (res.get("message") or "").strip()
    if msg:
        return msg
    return f"{fallback} (device reported '{res.get('status') or 'no status'}')"


# --- SLAM check: reading the Space's per-episode tracking report -------------
# The fleet never talks to the Space to make it WORK (a device does that, with its
# own token — see _run_dataset_job). It only READS the report afterwards, which
# needs no token at all: the Space exposes it per target repo, and the fleet is the
# one that named that repo. So the token boundary stays exactly where it was.
def _fetch_slam_quality_blocking(target_repo: str) -> dict[str, Any]:
    r = requests.get(f"{LEROBOT_SPACE_URL}/api/quality/{target_repo}",
                     timeout=SLAM_QUALITY_TIMEOUT_S)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, dict) else {}


async def _slam_quality(target_repo: str) -> Optional[dict[str, Any]]:
    """The Space's report for this repo, or None when it can't be read.

    Never raises: a missing report is a missing table, not a failed check. Runs in
    a thread because requests is blocking and this is called from the event loop."""
    try:
        return await asyncio.to_thread(_fetch_slam_quality_blocking, target_repo)
    except Exception as e:  # noqa: BLE001
        logger.info("SLAM report fetch for %s failed: %s", target_repo, e)
        return None


# Worst-first ordering for merging several verdicts about the same episode. An
# unknown verdict ranks as WARN rather than GOOD: a value this fleet doesn't know
# must never be reported as clean. "" is "not judged yet" and loses to everything.
_VERDICT_RANK = {"": -1, "GOOD": 0, "WARN": 1, "BAD": 2, "ERROR": 3, "FAIL": 4}
# The entry kinds that say something about the SLAM OUTCOME. A pre-check WARNING
# does not: it is about the recording (a joint that barely moved, say), and this
# check is about tracking — so it rides along as a note and never becomes the
# verdict. A pre-check ERROR is different: it stopped the take from reaching SLAM
# at all, so it IS the outcome.
_SLAM_KINDS = ("trajectory", "slam")


def _summarize_slam_quality(quality: list[dict]) -> list[dict[str, Any]]:
    """One row per episode from the Space's flat list of quality entries.

    The Space reports one entry per CHECK, not per episode: a pre-check warning and
    a trajectory verdict for the same take are two entries. Rendered as-is that
    shows the same episode twice, with two verdicts, and reads as a contradiction.
    So merge by episode name: one row, whose verdict is the SLAM outcome (see
    _SLAM_KINDS — a recording warning does not overrule a trajectory that tracked
    fine) plus the tracking figures from whichever entry carries them. The prose
    each check produced is deliberately dropped: the figures are what say what
    happened, and the Space's own report still holds the rest.

    A row can come back with an empty verdict: while the run is still working
    through the set, an episode may have been pre-checked and not yet SLAM'd. That
    is "not judged yet", which is neither GOOD nor a warning."""
    rows: dict[str, dict[str, Any]] = {}
    for q in quality:
        if not isinstance(q, dict):
            continue
        name = str(q.get("name") or "?")
        row = rows.setdefault(name, {"episode": name, "verdict": "", "tracking_pct": None,
                                     "n_lost": None, "n_frames": None, "n_jumps": None})
        kind = str(q.get("kind") or "")
        verdict = str(q.get("verdict") or "").upper()
        # Anything but a benign pre-check WARNING gets to set the verdict — so a
        # severity this fleet has never heard of is never demoted to a note.
        judges = kind in _SLAM_KINDS or verdict not in ("", "GOOD", "WARN")
        if judges and _VERDICT_RANK.get(verdict, 1) > _VERDICT_RANK.get(row["verdict"], 1):
            row["verdict"] = verdict
        for key in ("tracking_pct", "n_lost", "n_frames", "n_jumps"):
            if q.get(key) is not None:
                row[key] = q[key]
        # No messages at all: a row is the metrics and the verdict. The figures are
        # what say what happened (61% tracked, 351 frames lost), the prose behind
        # them was noise per episode, and the Space's report still carries it.
    # Episode names are timestamps, so sorting them IS chronological order.
    return [rows[k] for k in sorted(rows)]


def _episode_of(label: str) -> str:
    """The episode a report row belongs to. The Space names a row by its path in the
    raw dataset — "{episode_id}/{role}" — because it runs SLAM per ARM: two
    grabettes means two rows for one take. The table wants them both (per-arm
    tracking is worth seeing); every COUNT wants the take counted once, or a
    bimanual check ends up saying "2 of 3 flagged" for one bad take and "6 of 3
    came back" for a clean one."""
    return label.split("/")[0] or label


def _slam_check_note(job: DatasetJob, asked: int, judged: int) -> str:
    """What the check did NOT cover, in one clause — empty when it covered
    everything. A take the device had no data for is named (that answer comes from
    the device itself); short of that, a count that doesn't add up is still worth
    saying, because "all clear" must never quietly stand for fewer takes than the
    operator asked about."""
    gone = job.missing_episodes
    if gone:
        names = ", ".join(gone[:3]) + ("…" if len(gone) > 3 else "")
        return (f" {len(gone)} take(s) had no data on their device and were not "
                f"checked: {names}.")
    if judged < asked:
        return f" ({judged} of {asked} came back.)"
    return ""


async def _apply_slam_quality(job: DatasetJob, final: bool) -> None:
    """Pull the Space's report onto the job.

    Called repeatedly while the run works (the Space publishes its report live, so
    the pre-check findings and the log line show up as they happen) and once at the
    end, where it also writes the verdict the operator actually reads.

    A report that can't be fetched is never fatal: mid-run it's a no-op, and at the
    end it degrades to "the run finished, its report didn't come back" — which is
    the truth, and is NOT the same statement as "nothing was flagged"."""
    rep = await _slam_quality(job.target_repo)
    n = len(job.episode_ids)
    if rep is None or rep.get("status") == "not_found":
        if final:
            job.message = ("SLAM ran, but its report could not be read back from the Space "
                           "(a Space restart drops it — its job list is in memory). "
                           "Re-run the check.")
        return
    job.quality = _summarize_slam_quality(rep.get("quality") or [])
    # Which EPISODES are flagged, read off the merged rows — never off the Space's
    # own `flagged`, which is a list of ENTRIES: a take with two findings is in it
    # twice and a recording warning is in it at all, which is how a 4-episode check
    # announced "5 of 4 episode(s) flagged". The line the operator reads has to
    # agree with the table printed under it — and it counts takes, not arms (see
    # _episode_of), so an arm that lost tracking flags its take once.
    judged = [r for r in job.quality if r["verdict"]]
    eps_judged = {_episode_of(r["episode"]) for r in judged}
    if judged:
        job.flagged = sorted({_episode_of(r["episode"]) for r in judged
                              if r["verdict"] != "GOOD"})
    # A clean check pushes nothing, so both of these stay empty on purpose.
    job.result_url = rep.get("result") or None
    job.visualizer_url = rep.get("visualizer") or None
    if not final:
        tail = str(rep.get("log_tail") or "").strip()[:120]
        job.message = f"Checking SLAM on {n} episode(s)…" + (f" {tail}" if tail else "")
        return
    note = _slam_check_note(job, n, len(eps_judged))
    if job.flagged:
        names = ", ".join(job.flagged[:3]) + ("…" if len(job.flagged) > 3 else "")
        job.message = (f"{len(job.flagged)} of {n} episode(s) flagged: {names}. "
                       + ("The tested episodes were pushed — open them in the visualizer."
                          if job.result_url else
                          "Nothing could be pushed to look at — see the verdicts below.")
                       + note)
    elif job.flagged is None:
        # The run completed but the Space named no verdicts (an empty report):
        # say that instead of inventing a pass.
        job.message = f"SLAM ran on {n} episode(s) but reported no verdict — re-run the check."
    else:
        # Say when a dataset went out despite a clean result — a link with no
        # explanation reads as a contradiction — and never let "all clear" cover
        # takes that were not checked (see _slam_check_note).
        pushed = " The tested set was pushed — open it in the visualizer." if job.result_url else ""
        job.message = (f"No tracking loss on the {len(eps_judged)} episode(s) checked — "
                       f"recording can continue the same way.{pushed}{note}")


async def _run_dataset_job(owner: str, job: DatasetJob, private: bool,
                           processor_device_id: str) -> None:
    """Background: wait for every device's upload to complete, then have ONE
    device trigger the processing Space (raw → LeRobot, mono/bimanual per the
    device set) with ITS OWN long-lived token and report the result back.

    The fleet never handles an HF token for the Space call: the device→Space
    channel is the same one the SLAM flow already uses, so no token is cached on
    the fleet nor forwarded through it (avoids widening the token's blast radius
    and the short-lived-OAuth-token expiry problem). The Space downloads the raw and
    builds + pushes the dataset; it is asked to KEEP the raw afterwards (see
    KEEP_RAW_DATASET), so nothing on the raw side is ever deleted for now."""
    fleet = _fleet_of(owner)
    deadline = time.time() + 1800.0  # 30-min cap for the whole upload phase
    try:
        pending = dict(job.upload_cmds)  # device_id -> command id
        n = len(pending)
        while pending:
            if time.time() > deadline:
                raise RuntimeError("upload timed out")
            await asyncio.sleep(2.0)
            if job.cancelled:
                return  # the cancel path owns the status/message — don't touch them
            for dev_id, cmd_id in list(pending.items()):
                dev = fleet.get(dev_id)
                c = _command_status(dev, cmd_id) if dev else None
                if c is not None and c.status == "done":
                    res = c.result or {}
                    if res.get("status") != "ok":
                        # Even a FAILED upload can name episodes it screened out
                        # before giving up — keep them, the error alone doesn't say
                        # which takes are unusable.
                        _note_excluded(job, [
                            {**e, "stage": "upload", "device": (dev.name if dev else dev_id),
                             "reason": f"missing {', '.join(e.get('missing') or []) or 'required files'}"}
                            for e in (res.get("incomplete") or [])])
                        raise RuntimeError(f"{(dev.name if dev else dev_id)}: {_why_not_ok(res, 'upload failed')}")
                    # The device names the episodes it had nothing for. Keep them:
                    # they are the difference between what was asked for and what
                    # can possibly be checked, and an omission nobody sees is what
                    # makes a check untrustworthy.
                    for eid in (res.get("missing") or []):
                        if str(eid) not in job.missing_episodes:
                            job.missing_episodes.append(str(eid))
                    # Episodes this device refused to push: present on its card but
                    # missing a file the conversion needs. The device screens them
                    # out precisely so gigabytes aren't wasted — which also makes
                    # this the only place their names exist. Distinct from `missing`
                    # above: those were never recorded, these exist and were
                    # screened out, and only the ledger knows why.
                    _note_excluded(job, [
                        {**e, "stage": "upload", "device": (dev.name if dev else dev_id),
                         "reason": f"missing {', '.join(e.get('missing') or []) or 'required files'}"}
                        for e in (res.get("incomplete") or [])])
                    pending.pop(dev_id)
                    job.progress = (n - len(pending)) / n if n else 1.0
                    job.message = f"Uploaded {n - len(pending)}/{n} device(s)… (this can take several minutes)"
                    continue
                # Not done yet — bail fast if the device dropped offline (it went
                # dark before finishing, so its upload will never complete). An
                # actively-uploading device keeps polling → stays online, so this
                # only fires on a real disconnect (within ~ONLINE_WINDOW).
                if dev is None or not dev.online:
                    who = dev.name if dev else dev_id
                    raise RuntimeError(f"device '{who}' went offline before finishing its upload — reconnect it and retry")

        # Raw dataset complete → one device runs the processing (calls the Space
        # with its own token; the command completes when the Space is done).
        if job.cancelled:
            return  # cancelled between the last upload and here — don't start work
        job.raw_uploaded = True  # raw complete on HF → any later failure is conversion-only
        job.status = "processing"
        job.progress = None  # opaque Space conversion → indeterminate bar
        job.message = "Converting to LeRobot… (this can take several minutes)"
        dev = fleet.get(processor_device_id)
        if dev is None or not dev.online:
            raise RuntimeError("no online device available to run processing")
        proc = Command(id=uuid.uuid4().hex[:12], type="process_dataset",
                       args={"space_url": LEROBOT_SPACE_URL, "source_repo": job.raw_repo,
                             "target_repo": job.target_repo, "roles": job.roles,
                             "private": private,
                             "task": job.task_desc or job.target_repo.split("/")[-1],
                             # Ask the Space to LEAVE the raw dataset in place (see
                             # KEEP_RAW_DATASET). Only the Space can delete it.
                             # A SLAM check is the exception: its raw exists only to
                             # feed that one run, so it goes once the run completes
                             # (the Space deletes it on success only, so a failed
                             # check still leaves it there to inspect).
                             "keep_raw": KEEP_RAW_DATASET and not job.check,
                             # State outright that this is a check. The Space can
                             # also infer it from SLAM_CHECK_MARKER in the target
                             # repo, but the device cannot — and the device is where
                             # "done with no dataset" gets judged, so it has to be
                             # told rather than left to read the name.
                             "check_only": job.check})
        _enqueue(dev, proc)
        job.proc_cmd = proc.id  # so a cancel can reach the conversion too
        proc_deadline = time.time() + 3600.0  # 60-min cap for processing
        ticks = 0
        while True:
            if time.time() > proc_deadline:
                raise RuntimeError("processing timed out")
            await asyncio.sleep(3.0)
            if job.cancelled:
                return
            ticks += 1
            # A check exists FOR its report, so show it as it comes rather than
            # after the fact. Every 4th tick (~12s) against a run that takes
            # minutes — and a failed fetch here changes nothing.
            if job.check and ticks % 4 == 0:
                await _apply_slam_quality(job, final=False)
            c = _command_status(dev, proc.id)
            if c is None or c.status != "done":
                # Bail fast if the processing device dropped offline mid-run
                # (it keeps polling while working, so this is a real disconnect).
                d = fleet.get(processor_device_id)
                if d is None or not d.online:
                    who = d.name if d else processor_device_id
                    raise RuntimeError(f"device '{who}' went offline during processing — reconnect it and retry")
                continue
            res = c.result or {}
            _note_excluded(job, [{**e, "stage": "conversion"}
                                 for e in (res.get("excluded") or [])])
            if res.get("status") != "ok":
                raise RuntimeError(_why_not_ok(res, "processing failed"))
            if job.check:
                # No result_url fallback here: a clean check pushes NOTHING, so
                # guessing the dataset URL would hand the operator a dead link.
                # The report decides what to say (and whether there's a link).
                await _apply_slam_quality(job, final=True)
                job.status, job.progress = "done", 1.0
                return
            job.result_url = res.get("result_url") or f"https://huggingface.co/datasets/{job.target_repo}"
            # The excluded episodes are reported by the panel below the result
            # line, not here: saying it in both places made the operator read the
            # same count twice before reaching the link they actually wanted.
            # The ledger is the single source — see `excluded` in the status.
            job.status, job.message, job.progress = "done", "Dataset ready.", 1.0
            return
    except Exception as e:  # noqa: BLE001
        if job.cancelled:
            return  # a cancel racing with a failing device is a cancel, not an error
        job.status, job.error = "error", str(e)
        job.message = f"Failed: {e}"  # exclusions ride in the panel, not here
        logger.warning("dataset job %s failed: %s", job.id, e, exc_info=True)


def _cancel_dataset_job(owner: str, job: DatasetJob) -> dict[str, Any]:
    """Stop a running build EVERYWHERE, then report what was done per device.

    Flag the job first — the background runner checks job.cancelled on every tick
    and bows out — then deal with each device the job put to work. Two cases, and
    the difference matters:
      • the work command is still queued, undelivered → drop it. The device never
        heard of it, so there is nothing to abort and nothing to wait for.
      • it was already handed over → the device may be mid-upload or mid-conversion,
        so send a cancel_dataset naming the command to abort. The original command
        stays queued until the device reports on it, so the operator keeps seeing
        that device as busy while it really winds down (rather than a premature
        "idle" that would let a recording start on top of a live upload).

    Cancelling from ANY entry point — the dataset bar, or one device in the fleet
    list — always covers the WHOLE job: a raw dataset uploaded by only some of the
    devices is unusable, so a per-device cancel would be a trap."""
    job.cancelled = True
    fleet = _fleet_of(owner)
    dropped: list[str] = []    # never-delivered work, thrown away
    aborting: list[str] = []   # already working — asked to stop
    finished: list[str] = []   # already pushed their part before the cancel
    for dev_id, cmd_id in job.device_cmds().items():
        dev = fleet.get(dev_id)
        if dev is None:
            continue
        cmd = next((c for c in dev.queue if c.id == cmd_id), None)
        if cmd is None:
            finished.append(dev_id)  # result already came back — its data is pushed
            continue
        if cmd.status == "pending":
            dev.queue.remove(cmd)
            dropped.append(dev_id)
        else:
            # raw_repo is passed for context/logging only — the device must NOT
            # delete it. Nothing deletes the raw for now (see KEEP_RAW_DATASET).
            _enqueue(dev, Command(id=uuid.uuid4().hex[:12], type="cancel_dataset",
                                  args={"job_id": job.id, "raw_repo": job.raw_repo,
                                        "keep_raw": True, "command_ids": [cmd_id]}))
            aborting.append(dev_id)
    job.status = "cancelled"
    job.progress = None
    # Whatever reached the raw repo before the cancel stays there — deliberately:
    # nothing deletes a raw dataset for now (see KEEP_RAW_DATASET), and the fleet
    # couldn't anyway (devices push with their own tokens, the fleet holds none).
    # Name the repo so the partial upload is findable rather than a mystery.
    leftover = bool(aborting or finished)
    names = ", ".join((fleet[d].name if d in fleet else d) for d in aborting)
    job.message = f"Cancelled — asked to stop: {names}." if aborting else "Cancelled."
    if leftover:
        job.message += f" Whatever was already uploaded is kept in {job.raw_repo} (raw datasets are never deleted for now)."
    logger.info("dataset job %s cancelled (dropped=%s aborting=%s finished=%s)",
                job.id, dropped, aborting, finished)
    return {"status": "ok", "job_id": job.id, "job_status": job.status,
            "dropped": dropped, "aborting": aborting, "finished": finished,
            "raw_repo": job.raw_repo if leftover else ""}


@app.post("/api/fleet/lerobot-dataset")
async def create_lerobot_dataset(req: DatasetReq, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    # An empty list means "no restriction" (see DatasetReq.episode_ids), which is
    # what `or None` turns it into.
    roles, plan, skipped = _resolve_dataset_plan(owner, req.task_ids, req.only_available,
                                                 req.roles or None,
                                                 set(_clean_episode_ids(req.episode_ids)) or None)
    fleet = _fleet_of(owner)
    # With only_available the plan already excludes offline-device episodes; this
    # guard then bites only in the default mode (require the whole set up).
    offline = [d for d in plan if d not in fleet or not fleet[d].online]
    if offline:
        raise HTTPException(409, {"message": "some recording devices are offline", "devices": offline})
    # One member device runs the processing (calls the Space with its own token).
    processor = next(iter(plan))
    job_id = uuid.uuid4().hex[:8]
    name = (req.name or f"grabette-dataset-{job_id}").strip()
    target_repo = name if "/" in name else f"{owner}/{name}"
    # Intermediate raw dataset named after the target (…-raw) instead of random,
    # so it's easy to spot and correlate. Same namespace as the target.
    raw_repo = f"{target_repo}-raw"
    # What was left out rides along in the job message, which is what the dataset
    # bar polls and shows: an omission the operator never sees is the failure mode
    # this accounting exists to prevent.
    left_out = []
    if skipped["incomplete"]:
        left_out.append(f"{len(skipped['incomplete'])} recorded with fewer devices")
    if skipped["unavailable"]:
        left_out.append(f"{len(skipped['unavailable'])} with an offline device")
    note = f" Skipped {' and '.join(left_out)}." if left_out else ""
    job = DatasetJob(id=job_id, task_ids=list(req.task_ids), roles=roles,
                     raw_repo=raw_repo, target_repo=target_repo, processor=processor,
                     episodes_requested=skipped["included"],
                     message=f"Uploading {skipped['included']} episode(s) from {len(plan)} "
                             f"device(s)… (this can take several minutes).{note}")
    _dataset_jobs_of(owner)[job_id] = job
    # Seed the ledger with what the PLAN already dropped, so the final report
    # accounts for every stage rather than only the ones that happen later. These
    # also ride in the opening message (note, above); the ledger is what makes
    # them still visible once the build finishes and the message is rewritten.
    _note_excluded(job, [
        {"episode_id": eid, "stage": "plan",
         "reason": f"recorded without all of {'+'.join(roles)}"}
        for eid in skipped["incomplete"]])
    _note_excluded(job, [
        {"episode_id": eid, "stage": "plan",
         "reason": "a device it was recorded with is offline"}
        for eid in skipped["unavailable"]])
    for dev_id, p in plan.items():
        cmd = Command(id=uuid.uuid4().hex[:12], type="upload_episodes",
                      args={"raw_repo": raw_repo, "role": p["role"],
                            "episode_ids": sorted(p["episode_ids"]),
                            "private": req.private})
        _enqueue(fleet[dev_id], cmd)
        job.upload_cmds[dev_id] = cmd.id
    t = asyncio.create_task(_run_dataset_job(owner, job, req.private, processor))
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return {"status": "ok", "job_id": job_id, "raw_repo": raw_repo,
            "target_repo": target_repo, "roles": roles, "devices": list(plan.keys()),
            "episodes": skipped["included"], "skipped": skipped}


@app.get("/api/fleet/lerobot-dataset/{job_id}")
async def lerobot_dataset_status(job_id: str, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    job = _dataset_jobs_of(owner).get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {"id": job.id, "status": job.status, "message": job.message,
            "progress": job.progress,
            "raw_repo": job.raw_repo, "target_repo": job.target_repo,
            "result_url": job.result_url, "error": job.error,
            # On a conversion failure the UI points the operator at the Space that
            # does the raw → LeRobot step, so it must know which one is configured
            # (LEROBOT_SPACE_URL is overridable — never hard-code it client-side).
            "raw_uploaded": job.raw_uploaded, "space_url": LEROBOT_SPACE_URL,
            # Every episode left out, with its reason. The per-episode list is
            # only sent once the build is over: this endpoint is polled every 2s
            # for the whole build, the list is unchanged between polls, and the UI
            # renders it only in the terminal branches — so shipping it early was
            # re-sending several kilobytes a second for a fold nobody could open.
            # The summary is a short string and rides along throughout.
            "excluded": [] if job.status in DATASET_LIVE else job.excluded,
            "excluded_summary": _excluded_summary(job),
            # How many episodes the dataset actually holds (None = unknown), so
            # the result line can say what was BUILT and not only what was lost.
            "episodes": _episodes_in_dataset(job),
            # Devices still working for this job → the UI can offer Cancel while
            # any of them is busy, even on a page reloaded mid-build.
            "devices": sorted(job.device_cmds()),
            "cancellable": job.status in DATASET_LIVE and not job.cancelled,
            # SLAM check: same poll, different readout. `flagged` is three-valued
            # (null = no report) and the UI must keep those three apart.
            "check": job.check, "episode_ids": job.episode_ids,
            "quality": job.quality, "flagged": job.flagged,
            "visualizer_url": job.visualizer_url}


@app.get("/api/fleet/lerobot-datasets")
async def list_lerobot_datasets(request: Request) -> dict[str, Any]:
    """The owner's LIVE builds. Lets a reloaded page (or a second operator) pick the
    running build back up — and cancel it — instead of only being able to watch a
    build it started itself in this browser session."""
    owner = await _operator_loaded(request)
    return {"jobs": [{"id": j.id, "status": j.status, "message": j.message,
                      "progress": j.progress, "target_repo": j.target_repo,
                      "devices": sorted(j.device_cmds()), "cancellable": True,
                      # A SLAM check is picked up by the session panel, a build by
                      # the dataset bar — neither may adopt the other's job.
                      "check": j.check, "quality": j.quality, "flagged": j.flagged,
                      "episode_ids": j.episode_ids, "visualizer_url": j.visualizer_url}
                     for j in _dataset_jobs_of(owner).values()
                     if j.status in DATASET_LIVE and not j.cancelled]}


@app.post("/api/fleet/lerobot-dataset/{job_id}/cancel")
async def cancel_lerobot_dataset(job_id: str, request: Request) -> dict[str, Any]:
    """Cancel a running build on every device involved (see _cancel_dataset_job).
    Idempotent: cancelling an already-finished or already-cancelled job is a no-op
    rather than an error, so a double click — or two operators clicking at once —
    can't corrupt anything."""
    owner = await _operator_loaded(request)
    job = _dataset_jobs_of(owner).get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status not in DATASET_LIVE or job.cancelled:
        return {"status": "ok", "job_id": job.id, "job_status": job.status,
                "dropped": [], "aborting": [], "finished": [], "raw_repo": "",
                "already": True}
    return _cancel_dataset_job(owner, job)


@app.post("/api/fleet/devices/{device_id}/cancel-dataset")
async def cancel_dataset_for_device(device_id: str, request: Request) -> dict[str, Any]:
    """Cancel the dataset build this device is working for — the fleet-list entry
    point, for when the operator has left (or reloaded) the build page. It cancels
    the WHOLE job, not just this device's part: see _cancel_dataset_job."""
    owner = await _operator_loaded(request)
    job_id = _live_dataset_job_by_device(owner).get(device_id)
    if job_id is None:
        raise HTTPException(409, {"message": "this device isn't working for a dataset build "
                                             "started from the fleet"})
    return _cancel_dataset_job(owner, _dataset_jobs_of(owner)[job_id])


class SlamCheckReq(BaseModel):
    session_id: str
    # How many of the session's most recent episodes to run through SLAM. A handful
    # is the whole point: the check has to fit between two takes.
    count: int = SLAM_CHECK_DEFAULT_N


@app.post("/api/fleet/slam-check")
async def start_slam_check(req: SlamCheckReq, request: Request) -> dict[str, Any]:
    """Run the session's last few episodes through the SLAM Space and report, per
    episode, whether tracking held — the mid-session question "can I keep recording
    this task the same way?".

    It is a dataset build in every mechanical respect (upload per device → one
    device triggers the Space), so it IS a DatasetJob: progress, the per-device
    Cancel, the status endpoint and the gate that stops a recording starting on a
    busy device all work on it unchanged. What differs is the outcome — a
    per-episode tracking report — and that a clean check pushes nothing.

    The devices must be free: uploading is the same blocking work as a build, so a
    check is a deliberate pause between takes, never something running underneath a
    recording."""
    owner = await _operator_loaded(request)
    s = _get_open_session(owner, req.session_id)
    if s.recording or s.stopping:
        raise HTTPException(409, {"message": "stop the current episode before running a trajectory check"})
    count = max(1, min(SLAM_CHECK_MAX_N, int(req.count or SLAM_CHECK_DEFAULT_N)))
    roles, plan, episode_ids, skipped = _slam_check_plan(owner, s, count)
    fleet = _fleet_of(owner)
    offline = [d for d in plan if d not in fleet or not fleet[d].online]
    if offline:
        raise HTTPException(409, {"message": "a device that recorded these episodes is offline",
                                  "devices": offline})
    busy = _recording_blockers(owner, list(plan))
    if busy:
        raise HTTPException(409, {"message": "a device is already busy with a dataset build or check",
                                  "devices": busy})
    job_id = uuid.uuid4().hex[:8]
    task_name = _task_name(owner, s.task_id)
    target_repo = _slam_check_repo(owner, task_name,
                                  datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    job = DatasetJob(id=job_id, task_ids=[s.task_id], roles=roles,
                     raw_repo=f"{target_repo}-raw", target_repo=target_repo,
                     processor=next(iter(plan)), check=True, task_desc=task_name,
                     episode_ids=list(episode_ids),
                     message=f"Uploading {len(episode_ids)} episode(s) from {len(plan)} "
                             f"device(s) for the trajectory check…"
                             # An omission the operator can't see is what makes a
                             # check untrustworthy: say which takes were passed
                             # over and why, rather than quietly checking fewer.
                             + (f" Skipped {len(skipped)} take(s) no device holds: "
                                f"{', '.join(skipped[:3])}." if skipped else ""))
    _dataset_jobs_of(owner)[job_id] = job
    for dev_id, p in plan.items():
        # The raw goes up PRIVATE — it is the recording itself, and it exists only
        # to feed this one run (see keep_raw in _run_dataset_job). What may become
        # public is the LeRobot dataset a flagged check pushes, because the LeRobot
        # visualizer can only open a public one.
        cmd = Command(id=uuid.uuid4().hex[:12], type="upload_episodes",
                      args={"raw_repo": job.raw_repo, "role": p["role"],
                            "episode_ids": sorted(p["episode_ids"]), "private": True})
        _enqueue(fleet[dev_id], cmd)
        job.upload_cmds[dev_id] = cmd.id
    t = asyncio.create_task(_run_dataset_job(owner, job, False, job.processor))
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    logger.info("trajectory check %s on %s: %s episode(s) from %s device(s)",
                job_id, task_name, len(episode_ids), len(plan))
    return {"status": "ok", "job_id": job_id, "episodes": list(episode_ids),
            "devices": list(plan), "roles": roles, "target_repo": target_repo,
            "skipped": skipped}


# === device-facing sync API (Bearer auth) — physical-button episode start/stop
# A button press records an episode into the OPEN session containing this
# device (found by membership). It behaves exactly like the operator starting
# an episode from the dashboard: same task, same synchronized T0 across the
# session's members. No open session → solo (the device records locally). ======
@app.post("/api/devices/{device_id}/sync/start")
async def device_sync_start(device_id: str, auth: tuple[str, str] = Depends(device_auth)) -> dict[str, Any]:
    owner, _token = auth
    if device_id not in _fleet_of(owner):
        raise HTTPException(404, "Device not registered")
    s = _open_session_for_device(owner, device_id)
    if s is None:
        return {"status": "solo"}
    # The physical button and the local dashboard reach a group start ONLY through
    # here, so this is where they meet the same gate as the fleet's own button. It
    # was missing: an operator told "recording paused until the check finishes"
    # could press the button and get an episode anyway — a half-rig one, since the
    # busy peer refuses its own start_capture locally. A 409 travels back as
    # "refused", which the device turns into an error buzz and no recording.
    _raise_if_recording_blocked(owner, s.members.values())
    task_name = _task_name(owner, s.task_id)
    _task = _tasks_of(owner).get(s.task_id)
    signature = _task.device_signature if _task else []
    target_iso = _schedule_episode_start(
        owner, task_name, s.members, _lead_for(s),
        exclude_device_id=device_id, signature=signature, session=s,
    )
    episode_id = _episode_id_for_target(target_iso)
    s.episodes.append({"episode_id": episode_id, "roles": dict(s.members),
                       "started_at": datetime.now(timezone.utc).isoformat(),
                       "start_at_utc": target_iso})
    s.recording = True
    peers = [d for d in s.members.values() if d != device_id]
    # Full membership + signature so the pressing device persists who it recorded
    # with (it self-schedules from scheduled_start_utc, so it's excluded above).
    return {"status": "scheduled", "scheduled_start_utc": target_iso, "task_name": task_name,
            "peers": peers, "members": _named_members(owner, s.members), "signature": signature}


@app.post("/api/devices/{device_id}/sync/stop")
async def device_sync_stop(device_id: str, auth: tuple[str, str] = Depends(device_auth)) -> dict[str, Any]:
    owner, _token = auth
    if device_id not in _fleet_of(owner):
        raise HTTPException(404, "Device not registered")
    s = _open_session_for_device(owner, device_id)
    if s is None:
        return {"status": "solo"}
    ids = _dispatch_episode_stop(owner, s.members, exclude_device_id=device_id)
    _begin_episode_stop(s, ids)
    return {"status": "ok", "peers": [d for d in s.members.values() if d != device_id]}


@app.post("/api/fleet/dispatch")
async def dispatch(req: DispatchReq, request: Request) -> dict[str, Any]:
    owner = await _operator_loaded(request)
    dev = _fleet_of(owner).get(req.device_id)
    if dev is None:
        raise HTTPException(404, "Device not found in your fleet")
    cmd = Command(id=uuid.uuid4().hex[:12], type=req.type, args=dict(req.args))
    _enqueue(dev, cmd)
    return {"status": "queued", "command_id": cmd.id}


# === OAuth relay for grabette devices ========================================
@app.get("/oauth/grabette/callback", response_model=None)
async def grabette_oauth_relay(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> RedirectResponse | HTMLResponse:
    """Relay the HF OAuth callback to the originating grabette on the local network.

    The grabette encodes its mDNS hostname into the OAuth state as
    ``{hostname}|{session_id}``. This endpoint splits that apart and issues a
    302 redirect so the user's browser (on the same LAN as the grabette) reaches
    ``http://{hostname}.local:8000/api/hf-auth/oauth/callback``.

    Only this Space URL needs to be registered as a redirect_uri in the HF
    OAuth app — one entry covers every grabette regardless of hostname.
    """
    if not state or "|" not in state:
        return HTMLResponse("Missing or invalid state parameter.", status_code=400)

    hostname, session_id = state.split("|", 1)
    from urllib.parse import urlencode
    base = f"http://{hostname}.local:8000/api/hf-auth/oauth/callback"

    if error:
        # Forward the error to the grabette so it can mark the session as failed
        # and stop the polling loop on the frontend.
        params: dict = {"error": error, "state": session_id}
        if error_description:
            params["error_description"] = error_description
        return RedirectResponse(f"{base}?{urlencode(params)}", status_code=302)

    if not code:
        return HTMLResponse("Missing code.", status_code=400)

    return RedirectResponse(
        f"{base}?{urlencode({'code': code, 'state': session_id})}",
        status_code=302,
    )


# === UI ======================================================================
@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX)


_INDEX = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grabette fleet</title><style>
 *{box-sizing:border-box}
 body{font-family:-apple-system,system-ui,sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);
  color:#fff;min-height:100vh;margin:0;padding:2rem;display:flex;justify-content:center}
 .wrap{width:100%;max-width:640px}h1{font-size:1.3rem;margin:0 0 .3rem}
 h2{font-size:1rem;margin:0;display:flex;align-items:center;gap:.5rem}
 .intro{color:#c3cbe0;font-size:.88rem;line-height:1.5;margin:0 0 1.4rem}
 .card{background:rgba(255,255,255,.06);padding:1.2rem;border-radius:14px;margin-bottom:1rem}
 .card.groups{background:linear-gradient(135deg,rgba(139,92,246,.18),rgba(59,130,246,.10));
  border:1px solid rgba(167,139,250,.4);margin-bottom:1.4rem}
 .card.groups h2{color:#c4b5fd}
 .card.groups .count{background:rgba(167,139,250,.22);color:#e9d5ff}
 .card-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:.9rem}
 .count{background:rgba(255,255,255,.12);color:#e2e8f0;font-size:.72rem;font-weight:600;
  padding:.12rem .55rem;border-radius:999px}
 a.btn,button{padding:.55rem 1rem;border:0;border-radius:8px;cursor:pointer;font-weight:600;text-decoration:none;display:inline-block}
 .primary{background:#ffcc4d;color:#1a1a2e}.logout{background:#ef4444;color:#fff}
 .who-row{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap}
 .logout-icon{background:#16213e;color:#fff;border:1px solid #2d4a7a;padding:.3rem .5rem;line-height:1}
 .del-icon{background:#16213e;color:#ef4444;border:1px solid #ef4444;padding:.3rem .5rem;line-height:1}
 .rec-icon{background:#16213e;color:#10b981;border:1px solid #2d4a7a;padding:.3rem .5rem;line-height:1}
 .stop-icon{background:#16213e;color:#ef4444;border:1px solid #2d4a7a;padding:.3rem .5rem;line-height:1}
 .dash-icon{background:#16213e;color:#7dd3fc;border:1px solid #2d4a7a;padding:.3rem .5rem;line-height:1}
 button:disabled{opacity:.4;cursor:not-allowed}
 .muted{color:#a0aec0;font-size:.82rem}
 /* Field labels on a device's identity line — present enough to name the value
    next to them, dim enough that the values stay what you read. */
 .idlab{color:#6b7a90;text-transform:uppercase;font-size:.68rem;letter-spacing:.04em}
 table{width:100%;border-collapse:collapse;font-size:.85rem;table-layout:fixed}
 td,th{text-align:left;padding:.45rem .35rem;border-bottom:1px solid #334;vertical-align:top}
 .c-dot{width:1.2rem}.c-act{width:30%}.c-tools{width:6.6rem;text-align:right}
 .name b{word-break:break-word}
 .result{font-size:.8rem;color:#cbd5e0;
  background:rgba(0,0,0,.25);border-radius:6px;padding:.5rem .6rem;margin-top:.4rem}
 .result.empty{background:none;padding:0}
 .rlabel{font-size:.74rem;font-weight:600;color:#8b98ad;margin-bottom:.35rem;display:flex;align-items:center;gap:.5rem}
 .rclose{margin-left:auto;background:none;border:0;color:#8b98ad;font-size:1.05rem;line-height:1;cursor:pointer;padding:0 .15rem;font-weight:400}
 .rclose:hover{color:#fff}
 .kv{display:grid;grid-template-columns:auto 1fr;gap:.2rem .7rem}
 .kv .k{color:#8b98ad}.kv .v{overflow-wrap:anywhere;word-break:break-word}
 .pill{display:inline-block;padding:.03rem .5rem;border-radius:999px;font-size:.72rem;font-weight:600}
 .pill.rec{background:#ef4444;color:#fff}.pill.idle{background:rgba(255,255,255,.14);color:#cbd5e0}
 .pill.init{background:rgba(245,158,11,.22);color:#fcd34d}
 .pill.stopping{background:rgba(239,68,68,.22);color:#fca5a5}
 .pill.hand{background:rgba(125,211,252,.16);color:#7dd3fc;margin-left:.4rem;vertical-align:middle}
 .rec-dur{font-variant-numeric:tabular-nums;margin-left:.15rem}
 .batt{display:inline-block;margin-left:.4rem;padding:.03rem .45rem;border-radius:999px;font-size:.7rem;font-weight:700;vertical-align:middle}
 .batt::before{content:"🔋 "}
 .batt.batt-ok{background:rgba(16,185,129,.2);color:#a7f3d0}
 .batt.batt-mid{background:rgba(234,179,8,.2);color:#fde68a}
 .batt.batt-low{background:rgba(239,68,68,.22);color:#fca5a5}
 /* Audio signal: a label facing an On/Off switch. State selectors go through
    aria-checked, never a .on/.off class — those two are taken globally by the
    online dot and would repaint the switch red. */
 .sp-audio{display:inline-flex;align-items:center;gap:.55rem;color:#cbd5e1;font-size:.85rem}
 .sp-audio-lbl{display:inline-flex;align-items:center;gap:.35rem;font-weight:600}
 /* Sized in em and stroked in currentColor so it tracks the label, unlike the
    bell emoji, which lands at a different size and colour per platform. */
 .bell-ico{width:1em;height:1em;flex-shrink:0}
 .sw{position:relative;display:inline-flex;padding:2px;border-radius:999px;line-height:1;
   background:#16213e;border:1px solid rgba(255,255,255,.22);cursor:pointer}
 .sw:hover{border-color:rgba(255,255,255,.5)}
 .sw-knob{position:absolute;top:2px;bottom:2px;left:2px;width:calc(50% - 2px);border-radius:999px;
   background:#3b82f6;transition:transform .16s ease,background .16s ease}
 .sw[aria-checked="false"] .sw-knob{transform:translateX(100%);background:#3a4358}
 .sw-opt{position:relative;z-index:1;min-width:2.5rem;text-align:center;padding:.28rem .5rem;
   font-size:.78rem;font-weight:600;color:#8b98ad;transition:color .16s ease}
 .sw[aria-checked="true"] .sw-opt-on,.sw[aria-checked="false"] .sw-opt-off{color:#fff}
 /* device activity badge (capturing / uploading / converting) */
 .act-badge{display:inline-block;margin-left:.4rem;padding:.03rem .5rem;border-radius:999px;font-size:.7rem;font-weight:700;vertical-align:middle;white-space:nowrap}
 .act-badge.act-cap{background:rgba(239,68,68,.22);color:#fca5a5}
 .act-badge.act-up{background:rgba(59,130,246,.2);color:#93c5fd}
 .act-badge.act-proc{background:rgba(139,92,246,.22);color:#d6c9fb}
 /* Hardware fault: the device refuses to record. Deliberately the loudest badge
    in the row — it is the only one that means "this grabette is out". */
 .act-badge.act-fault{background:rgba(239,68,68,.3);color:#fecaca;border:1px solid rgba(248,113,113,.55)}
 .dev-fault{margin:.35rem 0 0;font-size:.76rem;color:#fca5a5;line-height:1.4}
 .err{color:#fca5a5}
 .raw-d{margin-top:.45rem}.raw-d summary{cursor:pointer;color:#8b98ad;font-size:.72rem}
 .raw-d pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;margin:.3rem 0 0;font-size:.72rem;
  max-height:14rem;overflow:auto;background:rgba(0,0,0,.25);border-radius:5px;padding:.4rem .5rem}
 .acts{display:flex;flex-wrap:wrap;gap:.3rem}.acts button{padding:.22rem .7rem;font-size:.8rem}
 .acts button.act-on{box-shadow:inset 0 0 0 2px #1a1a2e}
 .tools{display:flex;gap:.3rem;justify-content:flex-end}
 .dot{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;margin-top:.3rem}
 .on{background:#10b981}.off{background:#ef4444}
 fieldset:disabled{opacity:.45}fieldset{border:0;padding:0;margin:0}code{color:#ffcc4d}
 .c-plus{width:2.4rem}
 .plus-btn{width:1.8rem;height:1.8rem;border-radius:50%;padding:0;font-size:1.1rem;font-weight:700;
  line-height:1;display:inline-flex;align-items:center;justify-content:center}
 .plus-btn.blue{background:#3b82f6;color:#fff}
 .plus-btn.green{background:#10b981;color:#fff}
 .plus-btn.grey{background:#3a4358;color:#6b7280;cursor:not-allowed}
 .pill.group{color:#fff}
 .task-row{padding:.5rem .4rem;border-radius:8px;cursor:pointer}
 .task-row:hover{background:rgba(255,255,255,.06)}
 /* Top line: name takes the full width, action buttons pinned right. */
 .task-top{display:flex;align-items:center;gap:.5rem}
 .task-top .tname{font-weight:600;flex:1;min-width:0;overflow-wrap:anywhere}
 .task-acts{display:flex;gap:.3rem;flex-shrink:0;margin-left:auto}
 /* Second line: "Required devices" tags + description wrap below the name. */
 .task-sub{color:#a0aec0;font-size:.78rem;margin-top:.3rem;display:flex;flex-wrap:wrap;align-items:center;gap:.3rem}
 .task-sub .pill.hand{margin:0}
 #task-editor,.task-edit-panel,.task-del-panel{display:flex;flex-direction:column;gap:.6rem;align-items:stretch}
 .subpanel{margin-top:.9rem;padding:.9rem 1rem;border-radius:10px;border:1px solid rgba(16,185,129,.35);
  background:rgba(0,0,0,.18)}
 .subpanel-title{font-weight:600;font-size:.85rem;color:#6ee7b7}
 .subpanel.task-del-panel{border-color:rgba(239,68,68,.4)}
 .subpanel-title.danger{color:#fca5a5}
 .del-warn{font-size:.85rem;color:#c3cbe0}
 button.del-confirm{background:#ef4444;color:#fff}
 #orphan-bar{margin-bottom:1.4rem}
 .orphan-acc{border:1px solid rgba(245,158,11,.4);border-radius:10px;background:rgba(245,158,11,.08)}
 .orphan-acc-head{display:flex;justify-content:space-between;align-items:center;gap:.5rem;
  padding:.65rem .9rem;cursor:pointer;color:#fcd34d;font-weight:600;font-size:.9rem;user-select:none}
 .orphan-acc-head:hover{background:rgba(245,158,11,.06)}
 .orphan-caret{color:#fcd34d}
 .orphan-acc-body{padding:0 .8rem .8rem;display:flex;flex-direction:column;gap:.7rem}
 .orphan-item{padding:.8rem .9rem;border-radius:10px;border:1px solid rgba(245,158,11,.4);
  background:rgba(245,158,11,.1);display:flex;flex-direction:column;gap:.4rem;align-items:flex-start}
 .orphan-head{font-size:.92rem;color:#fcd34d}
 .orphan-detail{font-size:.82rem;color:#c3cbe0}
 /* Unassigned-recordings inbox. Structurally the orphan accordion, but in the
    blue "info" family rather than amber: these are intact takes waiting to be
    filed, not an anomaly to repair. An inbox that looked like a warning would
    cry wolf and train the operator to ignore the amber one that matters. */
 /* Below the pinned Fleet card and above the mode tabs — outside the sticky
    container, so it scrolls away once the operator has moved past it. */
 #unassigned-bar{margin-bottom:.9rem}
 .unassigned-acc{border:1px solid rgba(59,130,246,.4);border-radius:10px;background:rgba(59,130,246,.08)}
 .card.triage .count{background:rgba(59,130,246,.22);color:#bfdbfe}
 #triage-body,#manage-body{display:flex;flex-direction:column;gap:.7rem}
 /* The manage page is the task-side twin of triage: same page furniture, same
    sticky toolbar, so the two read as one gesture in two places. */
 .card.manage .count{background:rgba(59,130,246,.22);color:#bfdbfe}
 .triage-back{align-self:flex-start;margin-bottom:1.1rem}
 .triage-row button{padding:.4rem .8rem;font-size:.82rem;border-radius:8px;line-height:1.25}
 .triage-toolbar{position:sticky;top:var(--sticky-top,72px);z-index:20;
  display:flex;flex-direction:column;gap:.5rem;
  padding:.6rem .7rem;border-radius:10px;border:1px solid rgba(59,130,246,.35);
  background:#1b2545;box-shadow:0 6px 12px -8px rgba(0,0,0,.7)}
 .triage-row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
 .triage-label{font-size:.82rem;color:#93c5fd;font-weight:600}
 .triage-count{font-size:.82rem;color:#c3cbe0}
 /* Rows are plain divs, and the checkbox is decorative (pointer-events:none):
    selection is driven by mousedown/mouseenter on the row so a drag can sweep
    many takes at once. A <label> would toggle the box a second time on click. */
 .uep{user-select:none}
 .uep input{pointer-events:none}
 .unassigned-acc-head{display:flex;justify-content:space-between;align-items:center;gap:.5rem;
  padding:.65rem .9rem;cursor:pointer;color:#93c5fd;font-weight:600;font-size:.9rem;user-select:none}
 .unassigned-acc-head:hover{background:rgba(59,130,246,.06)}
 .unassigned-caret{color:#93c5fd}
 .unassigned-item{padding:.8rem .9rem;border-radius:10px;border:1px solid rgba(59,130,246,.35);
  background:rgba(59,130,246,.1);display:flex;flex-direction:column;gap:.4rem}
 .unassigned-head{font-size:.92rem;color:#93c5fd}
 .unassigned-detail{font-size:.82rem;color:#c3cbe0}
 /* Shown in full: this is a page of its own now, so the takes are read straight
    down instead of through a 220px porthole. What stays put while scrolling is the
    toolbar above (grabette, select-all, destination). */
 .unassigned-eps{display:flex;flex-direction:column;gap:.15rem}
 .uep{display:flex;align-items:center;gap:.4rem;cursor:pointer}
 .uep input{flex-shrink:0}
 .uall{display:flex;align-items:center;gap:.3rem;font-size:.82rem;color:#93c5fd;cursor:pointer}
 .uassign-sel{background:#16213e;color:#e2e8f0;border:1px solid rgba(59,130,246,.4);
  border-radius:8px;padding:.35rem .5rem;font-size:.85rem;max-width:16rem}
 .split-actions{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.3rem}
 .inbox-ico{width:15px;height:15px;flex-shrink:0;vertical-align:-2px}
 .warn-ico{width:15px;height:15px;flex-shrink:0}
 .orphan-acc-head .warn-ico,.hist-toggle.warn .warn-ico{width:14px;height:14px;vertical-align:-2px}
 .task-warn{display:inline-flex;align-items:center;margin-left:.4rem;color:#fbbf24;vertical-align:middle}
 .task-warn .warn-ico{width:15px;height:15px}
 .hist-toggle.warn{color:#fcd34d}
 .hist-toggle.warn:hover{color:#fde68a}
 select.role-select{width:100%;padding:.5rem .6rem;border-radius:8px;border:1px solid #334;
  background:#fff;color:#111;font:inherit}
 select.role-select option{background:#fff;color:#111}
 .role-pick{display:flex;align-items:center;gap:.6rem}
 .role-pick .sig-label{min-width:5.5rem}
 .fleet-group{margin-top:1rem}.fleet-group:first-of-type{margin-top:.3rem}
 .fleet-group h3{font-size:.9rem;margin:0 0 .4rem;color:#c3cbe0;display:flex;align-items:center;gap:.5rem}
 .subcount{background:rgba(255,255,255,.12);color:#e2e8f0;font-size:.7rem;font-weight:600;padding:.1rem .5rem;border-radius:999px}
 .session-panel{margin-top:.9rem;padding:1rem;border-radius:12px;border:1px solid rgba(239,68,68,.4);
  background:rgba(0,0,0,.22)}
 .sp-head{display:flex;align-items:center;justify-content:space-between;gap:.6rem;margin-bottom:.4rem}
 .sp-task{font-weight:700;font-size:1.05rem}
 /* Big recording timer, centered just above the record button. */
 .sp-timer-wrap{display:flex;justify-content:center;min-height:1.9rem;margin:.6rem 0 .2rem}
 .sp-timer{font-variant-numeric:tabular-nums;font-size:1.9rem;font-weight:800;line-height:1;color:#fff}
 @keyframes sp-pulse{50%{opacity:.35}}
 /* Phone-camera-style record toggle: red circle (start) ↔ red square (stop). */
 .sp-rec{display:flex;justify-content:center;margin:0 0 1rem}
 .rec-toggle{width:72px;height:72px;border-radius:999px;border:4px solid rgba(255,255,255,.55);background:transparent;display:inline-flex;align-items:center;justify-content:center;cursor:pointer;padding:0}
 .rec-toggle:hover{border-color:rgba(255,255,255,.85)}
 .rec-toggle .rt-inner{background:#ef4444;transition:width .15s,height .15s,border-radius .15s}
 .rec-toggle.idle .rt-inner{width:52px;height:52px;border-radius:999px}
 .rec-toggle.recording .rt-inner{width:28px;height:28px;border-radius:8px}
 .rec-toggle.busy{cursor:default;border-color:rgba(255,255,255,.35)}
 .rec-toggle.busy .rt-inner{width:34px;height:34px;border-radius:999px;opacity:.6;animation:sp-pulse 1s infinite}
 /* record circle blocked because a member is busy with a dataset job */
 .rec-toggle.blocked{cursor:not-allowed;opacity:.4;border-color:rgba(255,255,255,.3)}
 .sp-busy-note{margin-top:.6rem;text-align:center;font-size:.8rem;color:#d6c9fb}
 /* a device never confirmed its stop — it may still be recording */
 .sp-fault-warn{margin-top:.6rem;text-align:center;font-size:.8rem;color:#fca5a5;
  background:rgba(239,68,68,.12);border:1px solid rgba(248,113,113,.4);
  border-radius:10px;padding:.5rem .7rem;line-height:1.45}
 .sp-stop-warn{margin-top:.6rem;text-align:center;font-size:.8rem;color:#fcd34d;
   border:1px solid rgba(245,158,11,.4);background:rgba(245,158,11,.08);
   border-radius:8px;padding:.5rem .6rem}
 .sp-stop-warn .warn-ico,.sp-fault-warn .warn-ico{width:14px;height:14px;vertical-align:-2px}
 .restop-btn{margin-top:.45rem;display:block;width:100%;background:#f59e0b;color:#1a1a2e;
   border:0;padding:.4rem;border-radius:7px;font-weight:700;font-size:.82rem;cursor:pointer}
 .sp-meta{display:flex;align-items:baseline;gap:.6rem;margin-top:.35rem;font-size:.9rem}
 .sp-label{color:#8b98ad;font-size:.78rem;font-weight:600;min-width:5rem}
 .sp-devs{display:flex;flex-wrap:wrap;gap:.8rem}
 .sp-dev{display:inline-flex;align-items:center;gap:.35rem}
 .sp-actions{display:flex;gap:.5rem;margin-top:.8rem;flex-wrap:wrap}
 .sp-actions button{display:inline-flex;align-items:center;gap:.4rem;padding:.4rem .8rem;font-size:.85rem}
 .sp-actions .del-ep{background:#16213e;color:#ef4444;border:1px solid #ef4444}
 .sp-actions .del-ep:disabled{opacity:.45;cursor:not-allowed}
 .sp-close{display:block;width:100%;margin-top:1rem;padding:.6rem;font-size:.9rem;border:0;border-radius:8px;cursor:pointer;font-weight:600}
 /* ===== Session panel — two zones: recording box vs management box ===== */
 .del-ep{background:#16213e;color:#ef4444;border:1px solid #ef4444;padding:.45rem .8rem;border-radius:8px;font-size:.85rem;cursor:pointer}
 .del-ep:disabled{opacity:.45;cursor:not-allowed}
 .close-btn{background:#ffcc4d;color:#1a1a2e;border:0;padding:.55rem;border-radius:8px;font-weight:700;font-size:.9rem;cursor:pointer;width:100%;margin-top:.8rem}
 .var-c{padding:0;overflow:hidden}
 .sp-rec-zone{padding:1rem;background:rgba(239,68,68,.08)}
 .sp-manage-zone{padding:1rem;background:rgba(255,255,255,.04);border-top:1px solid rgba(255,255,255,.12)}
 /* Bottom row of the recording zone: the chime toggle on the left, facing
    "Delete last episode" on the right. Wraps rather than squeezing on a narrow
    panel. */
 .sp-rec-del{display:flex;justify-content:space-between;align-items:center;
   gap:.5rem;flex-wrap:wrap;margin-top:.8rem}
 /* Recorded-episode count: one large number */
 .sp-section-label{font-size:.72rem;font-weight:700;color:#8b98ad;letter-spacing:.05em;text-transform:uppercase;margin:.2rem 0 .3rem}
 .ep-big{display:flex;align-items:baseline;gap:.5rem;margin:.15rem 0 .6rem}
 .ep-big-num{font-size:2.1rem;font-weight:800;line-height:1;font-variant-numeric:tabular-nums;
  background:linear-gradient(135deg,#10b981,#3b82f6);-webkit-background-clip:text;background-clip:text;color:transparent}
 .ep-big-lbl{font-size:.82rem;color:#8b98ad}
 /* SLAM check: the between-takes "is tracking still holding?" readout */
 .sp-slam{margin-top:.9rem;padding-top:.7rem;border-top:1px solid rgba(255,255,255,.1)}
 .slam-row{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap}
 .slam-lbl{font-size:.82rem;color:#8b98ad}
 /* Pills, not a <select>: renderSessionList rebuilds this panel every second
    (the timer, the phase), and innerHTML kills a native dropdown the instant it
    opens. A click is atomic, so a pill survives the same re-render. */
 .slam-n{background:rgba(255,255,255,.06);color:#c3cbe0;border:1px solid rgba(255,255,255,.18);
   border-radius:6px;padding:.15rem .5rem;font-size:.8rem;cursor:pointer;
   font-variant-numeric:tabular-nums}
 .slam-n:hover:not(:disabled){border-color:rgba(255,255,255,.45)}
 .slam-n.on{background:#3b82f6;border-color:#3b82f6;color:#fff;font-weight:700}
 .slam-n:disabled{opacity:.45;cursor:not-allowed}
 .slam-btn{background:#3b82f6;color:#fff;border:0;padding:.4rem .8rem;border-radius:8px;
   font-weight:600;font-size:.82rem;cursor:pointer;margin-left:auto}
 /* Wraps under the row on a narrow panel rather than squeezing the select. */
 .slam-cancel,.slam-btn{flex:0 0 auto}
 .slam-btn:disabled{opacity:.45;cursor:not-allowed}
 .slam-cancel{background:#16213e;color:#fca5a5;border:1px solid #ef4444;padding:.35rem .7rem;
   border-radius:8px;font-size:.8rem;cursor:pointer}
 .slam-hint{font-size:.75rem;margin-top:.35rem;line-height:1.35}
 .slam-job{margin-top:.5rem;font-size:.82rem;color:#c3cbe0;line-height:1.4}
 /* The section label IS the fold's handle. */
 .slam-head{display:flex;align-items:center;gap:.3rem;cursor:pointer;user-select:none}
 .slam-head:hover{color:#c3cbe0}
 .slam-caret{display:inline-block;width:.9rem;color:#8b98ad}
 /* Closed-header state, so a running or finished check is never out of sight. */
 .slam-badge{margin-left:.15rem;padding:.02rem .4rem;border-radius:999px;font-size:.65rem;
   font-weight:700;text-transform:none;letter-spacing:0}
 .slam-badge.run{background:rgba(59,130,246,.22);color:#bfdbfe}
 .slam-badge.ok{background:rgba(16,185,129,.2);color:#a7f3d0}
 .slam-badge.warn{background:rgba(234,179,8,.2);color:#fde68a}
 .slam-job.ok{color:#a7f3d0}
 .slam-job.warn{color:#fcd34d}
 .slam-job.err{color:#fca5a5}
 .slam-link{margin-top:.4rem;font-size:.82rem}
 .slam-link a{color:#7dd3fc}
 /* Per-episode report. Numbers right-aligned and tabular so a bad take stands out
    down the column instead of having to be read row by row.
    The episode ids don't wrap (a broken timestamp is unreadable), so on a narrow
    panel the table SCROLLS instead of squeezing the first column until its text
    runs under the verdict. */
 .sq-wrap{overflow-x:auto}
 /* table-layout:fixed + declared column widths. The automatic algorithm squeezes
    columns past their own content to honour width:100%, which is what kept drawing
    the episode label under the verdict — and min-width does NOT apply to table
    cells, so it could not be fixed from the cell side. Fixed widths also guarantee
    the header and body stay in the same columns. The floor equals the declared
    total, so a narrower panel scrolls instead of compressing. */
 .sq-table{width:100%;min-width:30rem;table-layout:fixed;border-collapse:collapse;
   margin-top:.45rem;font-size:.78rem}
 .sq-table col.c-ep{width:12rem}
 .sq-table col.c-v{width:4.8rem}
 .sq-table col.c-n{width:4.4rem}
 .sq-table th{text-align:left;color:#8b98ad;font-weight:600;padding:.2rem .3rem;
   border-bottom:1px solid rgba(255,255,255,.12)}
 .sq-table td{padding:.22rem .3rem;border-bottom:1px solid rgba(255,255,255,.06)}
 /* The numbers stay right-aligned (a bad take then stands out down the column),
    so their HEADER has to sit on the same edge — otherwise the value reads as
    belonging to the column on its right. Needs to outrank `.sq-table th`. */
 .sq-num{text-align:right;font-variant-numeric:tabular-nums}
 .sq-table th.sq-num{text-align:right}
 /* The full label ("20260826_114634/right") is 21 characters of nowrap text. Give
    the column that width outright: a table honours width:100% by squeezing columns
    past their content, which is what drew the episode text under the verdict. With
    a floor here and on the table, a narrow panel scrolls instead. */
 .sq-ep{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-right:1.5rem}
 .sq-v{padding-left:0}
 .v-pill{display:inline-block;padding:.02rem .45rem;border-radius:999px;font-size:.68rem;font-weight:700}
 .v-good{background:rgba(16,185,129,.2);color:#a7f3d0}
 .v-warn{background:rgba(234,179,8,.2);color:#fde68a}
 .v-bad{background:rgba(239,68,68,.22);color:#fca5a5}
 .v-pending{background:rgba(255,255,255,.08);color:#8b98ad}
 .task-block{border-bottom:1px solid #223;}
 .task-detail{padding:.2rem .4rem .7rem 1.6rem}
 .task-launch{display:flex;flex-direction:column;gap:.5rem;margin:.2rem 0 .6rem}
 .launch-hint{font-size:.78rem;align-self:center}
 .section-note{font-size:.8rem;line-height:1.35;margin:.1rem 0 .8rem}
 .task-eps{margin-top:.35rem}
 .ep-count{display:inline-block;font-size:.72rem;color:#a7f3d0;background:rgba(16,185,129,.18);padding:.08rem .55rem;border-radius:999px;white-space:nowrap}
 .ghost{background:rgba(255,255,255,.08);color:#c3cbe0;border:1px solid rgba(255,255,255,.18);padding:.3rem .7rem;font-size:.8rem;border-radius:8px;cursor:pointer}
 .ghost:hover{background:rgba(255,255,255,.14)}
 .task-row.select{cursor:pointer}
 .task-row.select .task-top{gap:.6rem}
 .task-row.disabled{opacity:.4;cursor:not-allowed}
 .task-check{width:16px;height:16px;accent-color:#10b981;pointer-events:none;flex-shrink:0}
 /* Vertical stack: content recap, then the labelled name field, then Private,
    then the button — each on its own line. */
 #dataset-bar{display:none;flex-direction:column;align-items:flex-start;gap:.6rem;
  margin-top:.8rem;padding:.7rem .8rem;border-radius:10px;background:linear-gradient(135deg,rgba(139,92,246,.14),rgba(59,130,246,.10));border:1px solid rgba(139,92,246,.32)}
 #top-sticky{position:sticky;top:0;z-index:30;background:#181e36;padding:.7rem 0;margin-bottom:.9rem;
  display:flex;flex-direction:column;gap:.7rem;box-shadow:0 6px 12px -8px rgba(0,0,0,.7)}
 #top-sticky .card{margin-bottom:0}
 .fleet-head{cursor:pointer;user-select:none;margin-bottom:0}
 .fleet-head h2{display:inline}
 .fleet-caret{color:#8b98ad;font-size:.9rem}
 .fleet-recap{display:inline-flex;flex-wrap:wrap;gap:.3rem;align-items:center}
 .fleet-chip{font-size:.68rem;font-weight:600;color:#c3cbe0;background:rgba(255,255,255,.1);padding:.06rem .45rem;border-radius:999px;white-space:nowrap}
 .fleet-chip.zero{color:#6b7280;background:rgba(255,255,255,.05)}
 .fleet-chip.busy{color:#d6c9fb;background:rgba(139,92,246,.22)}
 #fleet-body{margin-top:.9rem;max-height:42vh;overflow:auto}
 /* Pinned just under the Fleet card, at the offset measured in syncStickyOffset
    (the Fleet accordion changes height, so the offset can't be a constant). The
    banner between them is deliberately NOT sticky: it scrolls away once used. */
 #tabs-sticky{position:sticky;top:var(--sticky-top,72px);z-index:20;
  background:#181e36;padding-bottom:.9rem}
 .page-tabs{display:flex;gap:4px;padding:4px;border-radius:999px;
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1)}
 .page-tabs .seg-btn{flex:1;justify-content:center}
 .seg-btn{display:inline-flex;align-items:center;justify-content:center;gap:.5rem;border:0;background:transparent;
  color:#aeb8cc;font:inherit;font-size:1rem;font-weight:700;padding:.7rem 1.1rem;border-radius:999px;cursor:pointer;
  white-space:nowrap;transition:background .12s,color .12s,box-shadow .12s}
 .seg-btn svg{width:16px;height:16px}
 .seg-btn:hover{color:#e2e8f0}
 #seg-record.active{background:linear-gradient(135deg,rgba(16,185,129,.5),rgba(59,130,246,.42));color:#fff;box-shadow:0 1px 8px rgba(16,185,129,.3)}
 #seg-dataset.active{background:linear-gradient(135deg,rgba(139,92,246,.5),rgba(59,130,246,.42));color:#fff;box-shadow:0 1px 8px rgba(139,92,246,.3)}
 .mode-dataset .step-num{background:linear-gradient(135deg,#8b5cf6,#3b82f6);color:#fff}
 .mode-dataset .ep-count{background:rgba(139,92,246,.2);color:#d6c9fb}
 #ds-gen{background:linear-gradient(135deg,#8b5cf6,#3b82f6);color:#fff}
 .ds-info{font-size:.82rem;color:#c3cbe0}
 .ds-step{display:flex;align-items:center;gap:.55rem;font-weight:700;font-size:1.05rem;color:#e2e8f0}
 #ds-step1{margin:.2rem 0 .6rem}
 #ds-advanced{margin:-.2rem 0 .7rem}
 .adv-toggle{font-size:.8rem;color:#8b98ad;cursor:pointer;user-select:none;padding:.2rem 0}
 .adv-toggle:hover{color:#c3cbe0}
 .adv-body{display:flex;align-items:center;flex-wrap:wrap;gap:.4rem;padding:.3rem 0 .1rem;font-size:.82rem}
 .use-only{border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.05);color:#c3cbe0;
  font:inherit;font-size:.8rem;font-weight:600;padding:.22rem .6rem;border-radius:7px;cursor:pointer}
 .use-only:hover{background:rgba(255,255,255,.12)}
 .use-only.active{background:rgba(16,185,129,.22);color:#a7f3d0;border-color:rgba(16,185,129,.5)}
 .adv-hint{flex-basis:100%;font-size:.75rem;margin-top:.15rem}
 .step-num{display:inline-flex;align-items:center;justify-content:center;width:1.7rem;height:1.7rem;flex-shrink:0;
  border-radius:999px;background:linear-gradient(135deg,#10b981,#3b82f6);color:#fff;font-size:.85rem;font-weight:700}
 .ds-field{display:flex;flex-direction:column;gap:.3rem;width:100%}
 .ds-label{font-size:.75rem;font-weight:600;color:#8b98ad}
 .ds-req{color:#fca5a5}
 .ds-devlabel{font-size:.75rem}
 .ds-devlist{display:flex;flex-wrap:wrap;gap:.3rem .8rem;margin-top:.2rem}
 .ds-dev{display:inline-flex;align-items:center;gap:.35rem;font-size:.82rem;color:#c3cbe0}
 .ds-dot{width:.55rem;height:.55rem;border-radius:999px;flex-shrink:0}
 .ds-dot.on{background:#10b981}
 .ds-dot.off{background:#ef4444}
 .ds-avail{display:inline-flex;align-items:center;gap:.3rem;font-size:.8rem;color:#c3cbe0;cursor:pointer}
 .ds-avail input{width:auto;min-width:0;margin:0}
 /* Destination group: the naming + Private + action live below a rule, so they
    read as "where it lands" rather than as more of the selection controls above
    (the availability toggle). The bar is a flex column, hence width:100%. */
 .ds-dest{width:100%;display:flex;flex-direction:column;align-items:flex-start;gap:.5rem;
  margin-top:.5rem;padding-top:.85rem;border-top:1px solid rgba(139,92,246,.3)}
 .ds-dest-label{font-size:.7rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#8b98ad}
 .ds-target{display:flex;align-items:center;gap:.3rem;flex-wrap:wrap}
 .ds-target select,.ds-target #ds-name{padding:.35rem .5rem;border-radius:8px;border:1px solid #334;font:inherit;font-size:.82rem}
 .ds-target select{background:#fff;color:#111}
 .ds-target #ds-name{background:rgba(255,255,255,.05);color:#fff;min-width:8rem}
 .ds-slash{color:#8b98ad}
 /* Keep the checkbox tight to its label (don't inherit the text-input width). */
 .ds-private{display:inline-flex;align-items:center;gap:.25rem;font-size:.8rem;color:#c3cbe0;cursor:pointer;white-space:nowrap}
 .ds-private input{width:auto;min-width:0;margin:0}
 .ds-job{width:100%;font-size:.8rem;color:#c3cbe0}
 .ds-job.err{color:#fca5a5}
 .ds-job.ok{color:#a7f3d0}
 .ds-job.warn{color:#fcd34d}
 .ds-progress{width:100%;height:6px;border-radius:999px;background:rgba(255,255,255,.12);overflow:hidden}
 .ds-progress-fill{height:100%;width:0;border-radius:999px;background:#10b981;transition:width .3s ease}
 .ds-progress.indet .ds-progress-fill{width:40%;background:linear-gradient(90deg,rgba(16,185,129,0),#10b981,rgba(16,185,129,0));animation:ds-indet 1.2s linear infinite}
 @keyframes ds-indet{0%{transform:translateX(-120%)}100%{transform:translateX(320%)}}
 .ds-job a{color:#7dd3fc}
 /* Recovery note under a failed build — explanatory, so it stays neutral-toned
    rather than inheriting the red of the error line above it. */
 /* Excluded-episode report on a finished build: the count is part of the result
    line, the per-episode list sits behind a <details> so a 40-episode task
    doesn't push the whole page down. */
 .ds-excl{margin-top:.5rem;font-size:.78rem;color:#fcd34d;text-align:left}
 .ds-excl summary{cursor:pointer;font-weight:700}
 .ds-excl ul{margin:.4rem 0 0;padding-left:1.1rem;color:#e5e7eb;font-weight:400}
 .ds-excl li{margin:.12rem 0;line-height:1.35}
 .ds-note{margin-top:.35rem;color:#c3cbe0;line-height:1.45}
 .hist-toggle{font-size:.8rem;color:#8b98ad;cursor:pointer;user-select:none;padding:.25rem 0}
 .hist-toggle:hover{color:#c3cbe0}
 .hist-row{font-size:.82rem;padding:.25rem 0;color:#cbd5e0}
 .hist-head{display:flex;align-items:center;justify-content:space-between;gap:.6rem;flex-wrap:wrap}
 .hist-acts{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap}
 .fill-task{background:#16213e;color:#7dd3fc;border:1px solid #2d4a7a;padding:.2rem .6rem;border-radius:7px;font-size:.76rem;cursor:pointer;white-space:nowrap}
 .fill-ep-hint{font-size:.75rem;color:#8b98ad;margin-left:.3rem;white-space:nowrap}
 /* Selection is driven by the ROW (see epDragStart), so the box is decorative —
    same rule, and same reason, as .uep in the inbox. */
 .tep,.dsep{display:flex;align-items:center;gap:.4rem;cursor:pointer;user-select:none}
 .tep input,.dsep input{flex-shrink:0;pointer-events:none}
 /* Advanced selection, expanded under a task row in Build dataset mode. Same
    furniture as the episode page's toolbar, without the sticky: this one sits
    inside a task block a few rows tall, so pinning it would only detach it from
    the list it acts on. */
 .ds-pick{padding:.1rem .4rem .7rem 1.6rem;display:flex;flex-direction:column;gap:.45rem}
 .ds-pick-bar{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;
  padding:.5rem .6rem;border-radius:10px;border:1px solid rgba(59,130,246,.35);background:#1b2545}
 .act-edit.on{background:#1e3a5f;color:#bfdbfe}
 .act-edit{background:#16213e;color:#7dd3fc;border:1px solid #2d4a7a;padding:.3rem .5rem;line-height:1}
 #task-name-input,#task-desc-input,.edit-input{width:100%;padding:.5rem .6rem;border-radius:8px;border:1px solid #334;
  background:rgba(255,255,255,.05);color:#fff;font:inherit}
 .sig-row{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap}
 .sig-label{color:#c3cbe0;font-size:.85rem;font-weight:600}
 .sig-label.task-label{font-size:1.05rem;font-weight:700;color:#e2e8f0}
 .sig-checks{display:flex;gap:.8rem;align-items:center;color:#c3cbe0;font-size:.85rem}
 .sig-checks label{display:flex;align-items:center;gap:.3rem;cursor:pointer}
 .editor-actions{display:flex;gap:.5rem}
 .pill.task{background:rgba(255,204,77,.16);color:#ffcc4d}
 .card.tasks{margin-bottom:1.4rem}
 .card.sessions{background:linear-gradient(135deg,rgba(239,68,68,.14),rgba(139,92,246,.10));
  border:1px solid rgba(239,68,68,.32);margin-bottom:1.4rem}
 .card.sessions h2{color:#fca5a5}
 .card.sessions .count{background:rgba(239,68,68,.22);color:#fecaca}
 .pill.open{background:#10b981;color:#fff}
 .pill.closed{background:rgba(255,255,255,.14);color:#cbd5e0}
 button.validate{background:linear-gradient(135deg,#10b981,#3b82f6);color:#fff}
 button.cancel{background:rgba(255,255,255,.12);color:#e2e8f0}
 /* Generate + Cancel side by side; Cancel only appears while a build is live. */
 .ds-actions{display:flex;gap:.5rem;width:100%}
 .ds-actions button{flex:1}
 button.ds-cancel{background:#ef4444;color:#fff;flex:0 0 auto;padding-left:1rem;padding-right:1rem}
 /* Cancel-build button in the fleet device list, next to the activity badge. */
 .ds-cancel-dev{background:#ef4444;color:#fff;border:0;border-radius:999px;
  font-size:.68rem;font-weight:700;padding:.1rem .5rem;margin-left:.35rem;cursor:pointer}
 /* Phone: reclaim the wide desktop padding and let side-by-side rows stack. */
 @media (max-width:600px){
  body{padding:.7rem}
  .card{padding:.9rem}
  .role-pick{flex-wrap:wrap}
  .role-pick select{flex:1;min-width:0}
  .sig-row{align-items:flex-start}
 }
</style></head><body><div class="wrap">
<h1>Grabette fleet <span class="muted">— operator dashboard</span></h1>
<p class="intro">This dashboard lists every Grabette, Gripette and Casquette that has been
 detected and is connected to your HuggingFace account. Each device is grouped by type below;
 sign in with HuggingFace to see and control the devices that report to your fleet.</p>
<div class="card"><h2>HuggingFace login</h2><div id="who" style="margin-top:.7rem">Checking…</div></div>
<fieldset id="gated" disabled>
 <!-- Only the Fleet card is pinned. The triage banner and the mode tabs below it
      scroll away: what has to stay reachable while working down a long page is the
      fleet's state, not an entry point you have already used. -->
 <div id="top-sticky">
  <div class="card fleet-acc" id="fleet-card">
   <div class="card-head fleet-head" onclick="toggleFleet()">
    <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap"><h2>Fleet</h2><span class="fleet-recap" id="fleet-recap"></span></div>
    <span class="fleet-caret" id="fleet-caret">▸</span>
   </div>
   <div id="fleet-body" style="display:none">
    <div class="fleet-group" data-kind="grabette">
     <h3>Grabettes <span class="subcount" id="count-grabette">0</span></h3>
     <table><tbody id="tb-grabette"></tbody></table>
     <div class="muted empty" id="empty-grabette">No Grabettes detected yet.</div>
    </div>
    <div class="fleet-group" data-kind="gripette">
     <h3>Gripettes <span class="subcount" id="count-gripette">0</span></h3>
     <table><tbody id="tb-gripette"></tbody></table>
     <div class="muted empty" id="empty-gripette">No Gripettes detected yet.</div>
    </div>
    <div class="fleet-group" data-kind="casquette">
     <h3>Casquettes <span class="subcount" id="count-casquette">0</span></h3>
     <table><tbody id="tb-casquette"></tbody></table>
     <div class="muted empty" id="empty-casquette">No Casquettes detected yet.</div>
    </div>
   </div>
  </div>
 </div>
 <div id="unassigned-bar" style="display:none"></div>
 <div id="tabs-sticky">
  <div class="page-tabs" id="page-tabs" role="tablist" aria-label="Mode">
  <button class="seg-btn active" id="seg-record" role="tab" onclick="setSelectMode(false)"><svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="7"/></svg>Record</button>
  <button class="seg-btn" id="seg-dataset" role="tab" onclick="setSelectMode(true)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/></svg>Build dataset</button>
  </div>
 </div>
 <div id="orphan-bar" style="display:none"></div>
 <div class="card sessions" id="sessions-card" style="display:none">
  <div class="card-head"><div style="display:flex;align-items:center;gap:.5rem"><h2><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>Session</h2></div></div>
  <div id="session-list"></div>
 </div>
 <div class="card triage" id="triage-card" style="display:none">
  <button class="cancel triage-back" onclick="closeTriage()">← Back</button>
  <div class="card-head">
   <div style="display:flex;align-items:center;gap:.5rem"><h2><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg>Unassigned recordings</h2><span class="count" id="count-unassigned">0</span></div>
  </div>
  <p class="section-note muted">Takes recorded outside any session, or left over from a deleted task. Tick the ones that belong to a task and file them; only grabettes online now report theirs.</p>
  <div id="triage-body"></div>
 </div>
 <div class="card manage" id="manage-card" style="display:none">
  <button class="cancel triage-back" onclick="closeManage()">← Back</button>
  <div class="card-head">
   <div style="display:flex;align-items:center;gap:.5rem"><h2><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M7 12h14"/><path d="M11 18h10"/></svg>Manage episodes</h2><span class="count" id="count-manage">0</span></div>
  </div>
  <p class="section-note muted" id="manage-note"></p>
  <div id="manage-body"></div>
 </div>
 <div class="card tasks mode-record" id="tasks-card">
  <div class="card-head"><div style="display:flex;align-items:center;gap:.5rem"><h2><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>Tasks</h2><span class="count" id="count-tasks">0</span></div></div>
  <p class="section-note muted">These tasks come from the grabettes connected to this account — those online now, plus any connected earlier in this session. Episode counts reflect only the grabettes currently online.</p>
  <div id="ds-step1" class="ds-step" style="display:none"></div>
  <div id="ds-advanced" style="display:none"></div>
  <div id="task-list" class="muted empty">No tasks yet.</div>
  <div id="dataset-bar"></div>
  <button class="primary" id="btn-create-task" onclick="startCreateTask()" style="margin-top:.9rem">Create task</button>
  <div id="task-editor" class="subpanel" style="display:none">
   <div class="subpanel-title" id="task-editor-title">New task</div>
   <input id="task-name-input" type="text" placeholder="Task name">
   <input id="task-desc-input" type="text" placeholder="Description (optional)">
   <div class="sig-row">
    <span class="sig-label">Required devices</span>
    <span class="sig-checks">
     <label><input type="checkbox" id="task-sig-left">lgrabette</label>
     <label><input type="checkbox" id="task-sig-right">rgrabette</label>
     <label><input type="checkbox" id="task-sig-casquette">casquette</label>
    </span>
   </div>
   <div class="editor-actions">
    <button class="validate" onclick="validateTask()">Validate task</button>
    <button class="cancel" onclick="cancelTaskEdit()">Cancel</button>
   </div>
  </div>
 </div>
</fieldset></div><script>
const $=id=>document.getElementById(id);let loggedIn=false;
const KINDS=['grabette','gripette','casquette'];
let DEVICES=[];let TASKS=[];let SESSIONS=[];let ORPHANS=[];let SPLITS=[];let UNASSIGNED=[];
let uSel={};  // device_id -> Set of picked episode ids, for per-episode filing
let epSel={};  // task id -> Set of its episode ids picked on the manage page
let epMoveTask='';  // destination task NAME picked on the manage page
let manageTaskId='';  // task whose episodes the manage page is showing, '' = closed
// Advanced selection, in Build dataset mode: which task's episode list is
// expanded under its row ('' = none), and, per task, the episodes kept for the
// build. No entry for a task means the whole task — the default, and what the
// build sends when nothing was picked.
let dsPickOpen='';
let dsEpSel={};
let triageDev='';let triageTask='';  // grabette being triaged, and its destination
let taskEditId=null;let taskDeleteId='';
let launchRolePick={left:'',right:'',casquette:''};
let expandedTaskId='';let historyOpenId='';let issuesOpenId='';let orphansOpen=false;let triageOpen=false;let groupsOpenId='';let fleetOpen=false;
// Dataset selection mode: pick several tasks (same device signature) to build a
// LeRobot dataset. The first pick locks the signature; incompatible tasks are
// disabled. Selection persists across the 3s refresh (module-level state).
let selectMode=false;let datasetSel=new Set();let datasetJob=null;let datasetPollTimer=null;
// Id of the build this page started, kept apart from datasetJob so Cancel works
// from the very first click — before the first status poll comes back.
let datasetJobId='';
let datasetName='';let datasetOwner='';let datasetNamespaces=null;let datasetPrivate=false;let datasetOnlyAvailable=false;
// "Use only" advanced option: a role subset (empty = each task's full signature).
let datasetUseOnly=[];let datasetAdvOpen=false;
const USE_ONLY_OPTS=[{label:'L',roles:['left']},{label:'R',roles:['right']},{label:'LC',roles:['left','casquette']},{label:'RC',roles:['right','casquette']}];
const ROLE_LABEL={left:'lgrabette',right:'rgrabette',casquette:'casquette'};
function deviceName(device_id){const d=DEVICES.find(x=>x.device_id===device_id);return d?d.name:device_id;}
function deviceBattery(device_id){const d=DEVICES.find(x=>x.device_id===device_id);return d?d.battery:null;}
function batteryPill(b){
 if(b===null||b===undefined)return '';
 const p=Math.round(b);const cls=p<=15?'batt-low':p<=40?'batt-mid':'batt-ok';
 return `<span class="batt ${cls}" title="Battery ${p}%">${p}%</span>`;}
function fmtDur(ms){const s=Math.max(0,Math.floor(ms/1000));return Math.floor(s/60)+':'+String(s%60).padStart(2,'0');}
function tickRecDur(){document.querySelectorAll('.rec-dur').forEach(el=>{const st=+el.dataset.start;if(st)el.textContent=fmtDur(Date.now()-st);});}
function slotOf(d){const k=kindOf(d);if(k==='casquette')return 'casquette';if(k==='grabette'&&(d.hand==='left'||d.hand==='right'))return d.hand;return null;}
// device activity ('idle'|'capturing'|'uploading'|'processing') reported by
// the fleet (self-reported by the device, or inferred from dispatched work).
function deviceActivity(id){const d=DEVICES.find(x=>x.device_id===id);return (d&&d.activity)||'idle';}
// Busy with dataset work → cannot (re)start a recording (mirrors the server gate).
// Mirrors the server's _recording_blockers: enrolled in a live build/check counts
// even while this particular device reads idle, or the panel says "paused" during
// the gaps and the record button stays live.
function deviceBusyForRec(id){
 const d=DEVICES.find(x=>x.device_id===id);if(!d)return false;
 if(d.dataset_job)return true;
 const a=d.activity||'idle';return a==='uploading'||a==='processing';}
const ACT_LABEL={capturing:'● Recording',uploading:'↑ Uploading',processing:'⚙ Converting'};
function activityBadge(id){
 const a=deviceActivity(id);
 // Enrolled in a live build/check but doing nothing this second (its upload is
 // done, another device is converting): still held, so it must not read as free.
 if(a==='idle'){
  const d=DEVICES.find(x=>x.device_id===id);
  if(!d||!d.dataset_job)return '';
  const what=d.dataset_check?'trajectory check':'dataset build';
  return `<span class="act-badge act-proc" title="Held by a ${what} — can't record until it finishes">⏸ In ${d.dataset_check?'check':'build'}</span>`;}
 const cls={capturing:'act-cap',uploading:'act-up',processing:'act-proc'}[a]||'';
 return `<span class="act-badge ${cls}" title="Device is ${a}">${ACT_LABEL[a]||a}</span>`;}
// Hardware fault self-reported by the device: it REFUSES to record (no OAK-D
// calibration, no angle sensors). Shown next to the activity badge rather than
// instead of it — a faulted grabette can also be uploading, and hiding either
// fact behind the other is how you go hunting for the wrong problem.
function deviceFault(id){const d=DEVICES.find(x=>x.device_id===id);return (d&&d.hardware_error)||'';}
function faultBadge(id){
 const e=deviceFault(id);
 if(!e)return '';
 return `<span class="act-badge act-fault">⚠ Cannot record</span>`;}
// The badge is the alarm; this is the instruction. The device's message names
// the fault, the consequence AND the fix, so print it rather than making the
// operator hover a tooltip or walk to the grabette to read its LED.
function faultNote(d){
 if(!d.online||!d.hardware_error)return '';
 return `<div class="dev-fault">⚠ ${esc(d.hardware_error)}</div>`;}
function errText(j){const d=j&&j.detail;if(!d)return '';if(typeof d==='string')return d;return d.message||JSON.stringify(d);}
// ── Tasks ──
function startCreateTask(){taskEditId=null;$('task-editor-title').textContent='New task';$('task-name-input').value='';$('task-desc-input').value='';
 ['left','right','casquette'].forEach(s=>$('task-sig-'+s).checked=false);
 $('task-editor').style.display='flex';$('btn-create-task').style.display='none';}
// Edit happens INLINE, in the task's own block (see taskEditorHtml in
// renderTaskList) — not in the bottom editor — so it's visible even far down a
// long list. Just flag the task and re-render.
function editTask(id){taskEditId=(taskEditId===id)?'':id;expandedTaskId='';renderTaskList();}
// Inline editor for one task. The required devices are LOCKED once they're KNOWN
// and the task has episodes (changing them would misdescribe the recordings). An
// old task with episodes but no recorded signature can be set once to backfill it.
function sigLocked(t){return (t.episode_count||0)>0 && (t.device_signature||[]).length>0;}
function taskEditorHtml(t){
 const locked=sigLocked(t);
 const backfill=(t.episode_count||0)>0 && !locked;  // old task: set the devices once
 const chk=s=>`<label><input type="checkbox" class="edit-sig" value="${s}" ${(t.device_signature||[]).includes(s)?'checked':''} ${locked?'disabled':''}>${ROLE_LABEL[s]}</label>`;
 let note='';
 if(locked)note=`<div class="muted" style="font-size:.8rem">Locked — this task already has ${t.episode_count} recorded episode(s), so its required devices can't change.</div>`;
 else if(backfill)note=`<div class="muted" style="font-size:.8rem">This task has recordings but no devices set — choose them now to fix it. This can only be set once.</div>`;
 return `<div class="subpanel task-edit-panel">
   <div class="subpanel-title">Edit task</div>
   <input id="edit-name" class="edit-input" type="text" placeholder="Task name" value="${esc(t.name)}">
   <input id="edit-desc" class="edit-input" type="text" placeholder="Description (optional)" value="${esc(t.description||'')}">
   <div class="sig-row"><span class="sig-label">Required devices</span>
    <span class="sig-checks">${chk('left')}${chk('right')}${chk('casquette')}</span></div>
   ${note}
   <div class="editor-actions">
    <button class="validate" onclick="validateTask()">Save</button>
    <button class="cancel" onclick="editTask('${t.id}')">Cancel</button>
   </div>
  </div>`;}
function cancelTaskEdit(){taskEditId=null;$('task-editor').style.display='none';$('btn-create-task').style.display='inline-block';}
async function validateTask(){
 let name,description,device_signature;
 if(taskEditId){  // inline editor
  name=$('edit-name').value.trim();description=$('edit-desc').value.trim();
  device_signature=[...document.querySelectorAll('.edit-sig:checked')].map(c=>c.value);
 }else{           // bottom "create task" editor
  name=$('task-name-input').value.trim();description=$('task-desc-input').value.trim();
  device_signature=['left','right','casquette'].filter(s=>$('task-sig-'+s).checked);
 }
 if(!name){alert('Task needs a name');return;}
 if(!device_signature.length){alert('Select at least one required device');return;}
 const body={name,description,device_signature};
 const url=taskEditId?`/api/fleet/tasks/${taskEditId}`:'/api/fleet/tasks';
 const method=taskEditId?'PUT':'POST';
 const r=await fetch(url,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!r.ok){const j=await r.json().catch(()=>({}));alert(errText(j)||'Failed to save task');return;}
 taskEditId=null;$('task-editor').style.display='none';$('btn-create-task').style.display='inline-block';
 await refresh();}
// Delete asks for confirmation in an inline panel (taskDeleteHtml) that spells
// out the consequences (episodes + which devices lose their recordings), rather
// than a bare confirm(). Clicking delete again cancels.
function deleteTask(id){taskDeleteId=(taskDeleteId===id)?'':id;taskEditId='';expandedTaskId='';renderTaskList();}
function taskDeleteHtml(t){
 const eps=t.episodes||[];
 const devs=new Map();  // device_id -> {name, online} across all this task's episodes
 for(const ep of eps)for(const m of Object.values(ep.members||{}))if(m&&m.device_id)devs.set(m.device_id,{name:m.name,online:deviceOnline(m.device_id)});
 const list=[...devs.values()].map(d=>`<span class="ds-dev"><span class="ds-dot ${d.online?'on':'off'}"></span>${esc(d.name)}${d.online?'':' <span class="muted">offline</span>'}</span>`).join('');
 const offline=[...devs.values()].filter(d=>!d.online).length;
 return `<div class="subpanel task-del-panel">
   <div class="subpanel-title danger">Delete “${esc(t.name)}”?</div>
   <div class="del-warn">This permanently deletes the task and its <b>${eps.length}</b> recorded episode(s) on:</div>
   <div class="ds-devlist">${list||'<span class="muted">no recording devices</span>'}</div>
   ${offline?`<div class="del-warn muted">${offline} device(s) offline — they keep their copy until they reconnect, then you'll be prompted to finish the cleanup.</div>`:''}
   <div class="editor-actions">
    <button class="del-confirm" onclick='confirmDeleteTask("${t.id}")'>Delete permanently</button>
    <button class="cancel" onclick='deleteTask("${t.id}")'>Cancel</button>
   </div>
  </div>`;}
async function confirmDeleteTask(id){
 const r=await fetch(`/api/fleet/tasks/${id}`,{method:'DELETE'});
 if(!r.ok){const j=await r.json().catch(()=>({}));alert(errText(j)||'Failed to delete task');return;}
 taskDeleteId='';await refresh();}
function sigBadges(sig){return (sig||[]).map(s=>`<span class="pill hand" title="${s} role">${s[0].toUpperCase()}</span>`).join(' ');}
function inboxIcon(){return '<svg class="inbox-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg>';}
function warnIcon(){return '<svg class="warn-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';}
function toggleIssues(id){issuesOpenId=issuesOpenId===id?'':id;renderTaskList();}
function toggleGroups(id){groupsOpenId=groupsOpenId===id?'':id;renderTaskList();}
function toggleFleet(){fleetOpen=!fleetOpen;$('fleet-body').style.display=fleetOpen?'':'none';$('fleet-caret').textContent=fleetOpen?'▾':'▸';syncStickyOffset();}  // its height drives --sticky-top
// Header recap (accordion collapsed): count of each device type ONLINE now.
function renderFleetRecap(){
 let lg=0,rg=0,grip=0,casq=0,busy=0;
 for(const d of DEVICES){
  if(!d.online)continue;
  if(d.activity==='uploading'||d.activity==='processing')busy++;
  const k=kindOf(d);
  if(k==='grabette'){if(d.hand==='left')lg++;else if(d.hand==='right')rg++;}
  else if(k==='gripette')grip++;
  else if(k==='casquette')casq++;
 }
 const chip=(label,n,title)=>`<span class="fleet-chip${n?'':' zero'}" title="${title}">${label} ${n}</span>`;
 // Busy chip is visible even with the Fleet accordion collapsed (e.g. from Record).
 const busyChip=busy?`<span class="fleet-chip busy" title="${busy} device(s) busy with a dataset upload/conversion">⚙ ${busy} busy</span>`:'';
 $('fleet-recap').innerHTML=chip('L',lg,'lgrabettes online')+chip('R',rg,'rgrabettes online')+chip('C',casq,'casquettes online')+chip('Grip',grip,'gripettes online')+busyChip;}
// Distinct device combinations used to record a task, with episode count each.
// Built from the connected devices' reported episodes (their members map).
function taskCombos(t){
 const m=new Map();
 for(const ep of (t.episodes||[])){
  const entries=Object.entries(ep.members||{}).sort((a,b)=>a[0].localeCompare(b[0]));
  if(!entries.length)continue;
  const key=entries.map(([r,mm])=>r+':'+(mm.device_id||'?')).join('|');
  let g=m.get(key);
  if(!g){g={roles:entries.map(([r,mm])=>({role:r,name:mm.name,online:deviceOnline(mm.device_id)})),count:0};m.set(key,g);}
  g.count++;
 }
 return [...m.values()];}
function comboRowHtml(g){
 const devs=g.roles.map(r=>`<span class="pill hand" title="${r.role}">${r.role[0].toUpperCase()}</span> ${esc(r.name)}${r.online?'':' <span class="muted">offline</span>'}`).join(' · ');
 return `<div class="hist-row">${devs} — <b>${g.count}</b> episode${g.count>1?'s':''}</div>`;}
function toggleTaskDetail(id){
 if(expandedTaskId===id){expandedTaskId='';}
 else{expandedTaskId=id;launchRolePick={left:'',right:'',casquette:''};historyOpenId='';}
 renderTaskList();}
function toggleHistory(id){historyOpenId=historyOpenId===id?'':id;renderTaskList();}
function fmtTs(sec){try{return new Date(sec*1000).toLocaleString();}catch(e){return '';}}
// ── Dataset selection ──
function sigKey(sig){return (sig||[]).slice().sort().join('+');}
function datasetLockedSig(){for(const t of TASKS){if(datasetSel.has(t.id))return sigKey(t.device_signature);}return '';}
function useOnlySet(){return datasetUseOnly.length>0;}
function sameRoles(a,b){return a.length===b.length&&a.every(r=>b.includes(r));}
function taskHasRoles(t,roles){const sig=new Set(t.device_signature||[]);return roles.every(r=>sig.has(r));}
// A task can be picked for the dataset if: (with "use only") its signature is a
// superset of the chosen roles; else (default) it matches the first pick's exact
// signature. This is what lets a bimanual task be used for a left-only dataset.
function datasetCompatible(t){
 if(useOnlySet())return taskHasRoles(t,datasetUseOnly);
 const locked=datasetLockedSig();
 return !locked||sigKey(t.device_signature)===locked;}
// Roles actually used to build the dataset: the "use only" subset, or null=all.
function activeRoles(){return useOnlySet()?datasetUseOnly:null;}
function setUseOnly(i){
 const roles=USE_ONLY_OPTS[i].roles;
 datasetUseOnly=sameRoles(datasetUseOnly,roles)?[]:roles.slice();  // click active = clear
 // Drop any now-incompatible selection under the new constraint.
 for(const id of [...datasetSel]){const t=TASKS.find(x=>x.id===id);if(!t||!datasetCompatible(t))datasetSel.delete(id);}
 renderTaskList();}
function renderAdvanced(){
 const el=$('ds-advanced');
 if(!selectMode||!TASKS.length){el.style.display='none';return;}
 el.style.display='block';
 const opts=USE_ONLY_OPTS.map((o,i)=>`<button class="use-only${sameRoles(datasetUseOnly,o.roles)?' active':''}" onclick="setUseOnly(${i})">${o.label}</button>`).join('');
 el.innerHTML=`<div class="adv-toggle" onclick="datasetAdvOpen=!datasetAdvOpen;renderAdvanced()">${datasetAdvOpen?'▾':'▸'} Advanced options</div>`+
  (datasetAdvOpen?`<div class="adv-body"><span class="muted">Use only:</span> ${opts}`+
   `<div class="adv-hint muted">Includes any task that has at least these devices, and uploads only these to build the dataset.</div></div>`:'');}
// Switch the task list between the two segmented modes. Idempotent: re-clicking
// the active segment is a no-op (so a dataset in progress isn't reset).
function setSelectMode(on){
 on=!!on;
 if(on===selectMode)return;
 selectMode=on;datasetSel.clear();expandedTaskId='';datasetJob=null;
 // Reset any open task editor/delete/create panel so it doesn't linger across modes.
 taskEditId=null;taskDeleteId='';$('task-editor').style.display='none';
 if(datasetPollTimer){clearTimeout(datasetPollTimer);datasetPollTimer=null;}
 dsPickOpen='';dsEpSel={};  // no panel left open, no stale picking
 if(selectMode){datasetName='';datasetPrivate=false;datasetOnlyAvailable=false;datasetUseOnly=[];datasetAdvOpen=false;if(datasetNamespaces===null)fetchNamespaces();
  datasetJobId='';adoptLiveDatasetJob();}  // a build may already be running (reload / other operator)
 $('seg-record').classList.toggle('active',!selectMode);
 $('seg-dataset').classList.toggle('active',selectMode);
 // Drive the per-mode accent colour of the numbers/counts/encart below.
 $('tasks-card').classList.toggle('mode-dataset',selectMode);
 $('tasks-card').classList.toggle('mode-record',!selectMode);
 $('btn-create-task').style.display=selectMode?'none':'inline-block';
 renderTaskList();}  // renderTaskList sets the step-1 header text for the mode
async function fetchNamespaces(){
 try{
  const r=await fetch('/api/fleet/namespaces');
  if(r.ok){const j=await r.json();datasetNamespaces=j.namespaces||[];
   if(!datasetOwner)datasetOwner=j.default||datasetNamespaces[0]||'';
   renderDatasetBar();}
 }catch(e){}
}
function toggleDatasetPick(id){
 const t=TASKS.find(x=>x.id===id);if(!t)return;
 if(datasetSel.has(id)){datasetSel.delete(id);}
 else{if(!datasetCompatible(t))return;datasetSel.add(id);}
 renderTaskList();}
// ── Dataset device involvement (computed live from the tasks' reported episodes + DEVICES) ──
function deviceOnline(id){const d=DEVICES.find(x=>x.device_id===id);return !!(d&&d.online);}
function dsIncluded(t){return dsEpSel[t.id]||null;}
function dsEpisodesOf(t){
 const s=dsIncluded(t),eps=t.episodes||[];
 return s?eps.filter(ep=>s.has(ep.episode_id)):eps;}
// Restricted = the build would take fewer than the task holds. A set that still
// covers everything is NOT a restriction: it says the same as no set at all, and
// the row must not claim otherwise.
function dsRestricted(t){return !!dsIncluded(t)&&dsEpisodesOf(t).length<(t.episodes||[]).length;}
// Expand/collapse the panel. Opens with EVERYTHING ticked when this task has no
// restriction yet: the default is the whole task, so the operator starts from what
// a build would take today and subtracts from it.
function toggleDsPick(tid){
 const t=TASKS.find(x=>x.id===tid);if(!t)return;
 if(dsPickOpen===tid){dsPickOpen='';renderTaskList();return;}
 dsPickOpen=tid;
 if(!dsEpSel[tid])dsEpSel[tid]=new Set((t.episodes||[]).map(ep=>ep.episode_id));
 renderTaskList();}
function dsPicked(t){const s=dsEpSel[t.id]||new Set();return (t.episodes||[]).filter(ep=>s.has(ep.episode_id));}
function dsToggleAll(tid,on){
 const t=TASKS.find(x=>x.id===tid);if(!t)return;
 dsEpSel[tid]=on?new Set((t.episodes||[]).map(ep=>ep.episode_id)):new Set();
 renderTaskList();}
function dsLiveLabel(t,n){
 const total=(t.episodes||[]).length;
 return `${n} of ${total} episode${total>1?'s':''} will go into the dataset`;}
// Patches the panel mid-sweep, so the count and the select-all follow the drag
// without rebuilding the list under the cursor (see epApplyPick).
function dsSyncToolbar(tid){
 const t=TASKS.find(x=>x.id===tid);if(!t)return;
 const n=dsPicked(t).length,all=$('ds-all'),live=$('ds-live');
 if(all)all.checked=n>0&&n===(t.episodes||[]).length;
 if(live)live.textContent=dsLiveLabel(t,n);}
// The panel itself. Deliberately buttonless: ticking IS the action here — the
// build reads the selection when it starts — so there is nothing to confirm.
// Every take is tickable, including one whose grabette is offline: including an
// episode is not an order to a device, and the build has its own accounting for
// what it then can't upload (see _resolve_dataset_plan's incomplete/unavailable).
function dsPickHtml(t){
 const eps=(t.episodes||[]).slice().sort((a,b)=>b.episode_id.localeCompare(a.episode_id));
 const sel=dsEpSel[t.id]||new Set();
 const n=eps.filter(ep=>sel.has(ep.episode_id)).length;
 const rows=eps.map(ep=>`<div class="hist-row dsep" data-eid="${ep.episode_id}"
    onmousedown="epDragStart('dataset','${t.id}','${ep.episode_id}',event)"
    onmouseenter="epDragOver('dataset','${t.id}','${ep.episode_id}')">
    <input type="checkbox" ${sel.has(ep.episode_id)?'checked':''}>${epLineHtml(ep)}</div>`).join('')
  ||'<div class="muted" style="font-size:.82rem">No episode recorded yet.</div>';
 return `<div class="ds-pick">
   <div class="ds-pick-bar">
    <label class="uall"><input type="checkbox" id="ds-all" ${n&&n===eps.length?'checked':''} onchange="dsToggleAll('${t.id}',this.checked)"> Select all</label>
    <span class="triage-count" id="ds-live">${esc(dsLiveLabel(t,n))}</span>
   </div>
   <div class="unassigned-eps">${rows}</div>
  </div>`;}
function datasetEpisodes(){ // every PICKED episode across the selected tasks (from device reports)
 const eps=[];
 for(const t of TASKS){if(!datasetSel.has(t.id))continue;for(const ep of dsEpisodesOf(t))eps.push(ep);}
 return eps;}
function datasetDevices(){ // Map device_id -> role, union over selected episodes (only the active roles)
 const m=new Map();const only=activeRoles();
 for(const ep of datasetEpisodes())for(const [role,dev] of Object.entries(ep.roles||{})){
  if(only&&!only.includes(role))continue;
  m.set(dev,role);}
 return m;}
function datasetEpisodeCounts(){ // total, and how many have ALL their (active-role) devices online
 let total=0,avail=0;const only=activeRoles();
 for(const ep of datasetEpisodes()){
  const d=Object.entries(ep.roles||{}).filter(([r])=>!only||only.includes(r)).map(([,dev])=>dev);
  if(!d.length)continue;
  total++;if(d.every(deviceOnline))avail++;}
 return {total,avail};}
function datasetDevicesHtml(){
 const m=datasetDevices();
 if(!m.size)return '';
 const items=[...m.entries()].map(([id,role])=>{
  const on=deviceOnline(id);
  return `<span class="ds-dev"><span class="ds-dot ${on?'on':'off'}"></span>${esc(deviceName(id))}`+
   ` <span class="pill hand">${role[0].toUpperCase()}</span>${on?'':' <span class="muted">offline</span>'}</span>`;
 }).join('');
 return `<span class="ds-devlabel muted">Devices needed:</span><span class="ds-devlist">${items}</span>`;}
// A build that fails AFTER every device finished uploading lost only the conversion:
// the raw dataset is complete and kept on HF (see KEEP_RAW_DATASET). Say so, and
// send the operator to the Space that does the raw → LeRobot step so they can retry
// it there — re-running the whole build would re-upload gigabytes for nothing.
function datasetConvFailHtml(){
 if(!datasetJob||!datasetJob.raw_uploaded||!datasetJob.raw_repo)return '';
 const raw=esc(datasetJob.raw_repo);
 const rawUrl='https://huggingface.co/datasets/'+raw;
 const space=datasetJob.space_url?`<a href="${esc(datasetJob.space_url)}" target="_blank" rel="noopener">the SLAM Space</a>`:'the SLAM Space';
 return `<div class="ds-note">The episodes were all uploaded — only the LeRobot conversion failed.`+
  ` The raw dataset <a href="${esc(rawUrl)}" target="_blank" rel="noopener">${raw}</a> is kept on Hugging Face,`+
  ` so you can retry the generation from ${space} with it as source, without re-uploading anything from the devices.</div>`;}
// Episodes that did NOT make it into the dataset. This panel is the ONLY place
// that reports them: the result line stays "Dataset ready." so the operator
// isn't made to read the same count twice on the way to the link they wanted.
// The summary line carries how many and why; the names sit behind the fold,
// which is what you want only when you go fixing them.
//
// The fold's open/closed state lives OUTSIDE the DOM: renderDatasetBar rewrites
// the job block on every 3s refresh, so a <details> that only remembered its
// state in the element itself snapped shut a few seconds after being opened.
// Same idiom as openRaw/rawToggle for the device rows.
let dsExclOpen=false;
function dsExclToggle(open){dsExclOpen=open;}
function datasetExcludedHtml(){
 const ex=(datasetJob&&datasetJob.excluded)||[];
 if(!ex.length)return '';
 const items=ex.map(e=>{
  const who=e.role?` (${esc(e.role)})`:'';
  const dev=e.device?` — ${esc(e.device)}`:'';
  return `<li><code>${esc(e.episode_id)}</code>${who}${dev}: ${esc(e.reason)}</li>`;
 }).join('');
 const n=new Set(ex.map(e=>e.episode_id)).size;
 return `<details class="ds-excl"${dsExclOpen?' open':''} ontoggle='dsExclToggle(this.open)'>`
  +`<summary>${n} episode(s) not included in the dataset</summary><ul>${items}</ul></details>`;}
function renderDatasetBar(){
 const bar=$('dataset-bar');
 if(!selectMode){bar.style.display='none';bar.innerHTML='';return;}
 bar.style.display='flex';
 const sel=TASKS.filter(t=>datasetSel.has(t.id));
 const counts=datasetEpisodeCounts();
 const epText=datasetOnlyAvailable?`${counts.avail} / ${counts.total} episodes`:`${counts.total} episodes`;
 // With "use only" the dataset's roles are the chosen subset, not the task sig.
 const sig=useOnlySet()?datasetUseOnly:(sel.length?sel[0].device_signature:[]);
 const infoHtml=`${sel.length} task(s) · ${sigBadges(sig)||'<span class="muted">no device set</span>'} · ${epText}`;
 let jobCls='',jobHtml='';
 if(datasetJob){
  const m=esc(datasetJob.message||datasetJob.status||'');
  if(datasetJob.status==='error'){jobCls='err';jobHtml='✗ '+m+datasetExcludedHtml()+datasetConvFailHtml();}
  else if(datasetJob.status==='cancelled'){jobCls='warn';jobHtml='⊘ '+m;}
  // Name the dataset in the link: by the time a build ends the operator may have
  // several going / a stale bar on screen, and "open" alone doesn't say which one.
  else if(datasetJob.status==='done'){
   jobCls='ok';
   // The count of what IS in the dataset sits with the link, outside it: the
   // link text stays the repo name, and the number answers the question the
   // operator actually has once a build ends — how much did I get?
   const n=datasetJob.episodes;
   const nTxt=(n===null||n===undefined)?'':` (${n} episode${n===1?'':'s'})`;
   const link=datasetJob.result_url
    ?` — <a href="${datasetJob.result_url}" target="_blank" rel="noopener">Open ${esc(datasetJob.target_repo||'dataset')}</a>${nTxt}`:'';
   jobHtml='✓ '+m+link+datasetExcludedHtml();}
  else{jobHtml='… '+m;}
 }
 // Build the controls once (so the name <input> keeps focus/value across the 3s
 // refresh re-renders); afterwards only update the dynamic text/state.
 if(!$('ds-name')){
  bar.innerHTML=
   `<div class="ds-step"><span class="step-num">2</span> Name your destination repository</div>`+
   `<span class="ds-info" id="ds-info">${infoHtml}</span>`+
   `<div id="ds-devices"></div>`+
   `<label class="ds-avail"><input type="checkbox" id="ds-avail" ${datasetOnlyAvailable?'checked':''} onchange="datasetOnlyAvailable=this.checked;renderDatasetBar()"> Use only available devices</label>`+
   // Everything above describes WHAT goes in the dataset (selection recap + the
   // availability filter); everything below is WHERE it lands. Grouped behind a
   // rule so the two don't read as one undifferentiated list of controls.
   `<div class="ds-dest">`+
    `<div class="ds-dest-label">Destination</div>`+
    `<div class="ds-field"><label class="ds-label" for="ds-name">Dataset name <span class="ds-req">*</span></label>`+
    `<span class="ds-target"><select id="ds-owner" onchange="datasetOwner=this.value"></select>`+
    `<span class="ds-slash">/</span>`+
    `<input id="ds-name" placeholder="dataset-name" value="${esc(datasetName)}" oninput="datasetName=this.value;$('ds-gen').disabled=!dsCanGen()"></span></div>`+
    `<label class="ds-private"><input type="checkbox" id="ds-private" ${datasetPrivate?'checked':''} onchange="datasetPrivate=this.checked"> Private</label>`+
    `<div class="ds-actions"><button class="validate" id="ds-gen" onclick="generateDataset()">Generate LeRobot dataset</button>`+
    `<button class="ds-cancel" id="ds-cancel" style="display:none" onclick="cancelDataset()">Cancel</button></div>`+
   `</div>`+
   `<div class="ds-progress" id="ds-progress" style="display:none"><div class="ds-progress-fill"></div></div>`+
   `<div class="ds-job" id="ds-job"></div>`;
 }else{
  $('ds-info').innerHTML=infoHtml;
 }
 $('ds-devices').innerHTML=datasetDevicesHtml();
 // Owner options: (re)fill when the namespace list changes; keep the selection.
 const ownerSel=$('ds-owner');
 const opts=(datasetNamespaces&&datasetNamespaces.length)?datasetNamespaces:(datasetOwner?[datasetOwner]:[]);
 if(ownerSel.options.length!==opts.length)
  ownerSel.innerHTML=opts.map(n=>`<option value="${esc(n)}">${esc(n)}</option>`).join('');
 if(datasetOwner)ownerSel.value=datasetOwner;
 $('ds-gen').disabled=!dsCanGen();
 // Cancel sits next to Generate and only exists while devices are actually working
 // for the build (the server tells us: cancellable).
 const canCancel=!!(datasetJob&&datasetJob.cancellable&&datasetJobId);
 $('ds-cancel').style.display=canCancel?'':'none';
 const jobEl=$('ds-job');jobEl.className='ds-job'+(jobCls?' '+jobCls:'');
 // Only touch the DOM when the content actually changed. Compared against a
 // cached copy of what we last wrote, never against innerHTML: the browser
 // normalises markup (an opened <details> gains open=""), so reading it back
 // would never match and the block would be rebuilt on every refresh anyway.
 if(jobEl.dataset.html!==jobHtml){jobEl.innerHTML=jobHtml;jobEl.dataset.html=jobHtml;}
 // Progress bar: determinate during upload (fraction of devices done), animated
 // (indeterminate) during the opaque Space conversion, full & green when done.
 const prog=$('ds-progress'),fill=prog.querySelector('.ds-progress-fill');
 if(!datasetJob||datasetJob.status==='error'||datasetJob.status==='cancelled'){prog.style.display='none';}
 else{
  prog.style.display='block';
  const done=datasetJob.status==='done';
  // Same animated (indeterminate) bar for both work phases (upload + convert);
  // full green only when done. Each phase can take several minutes.
  const indet=!done;
  prog.className='ds-progress'+(indet?' indet':'')+(done?' done':'');
  fill.style.width=indet?'':'100%';
 }}
// Generate requires: ≥1 task, a name, no running job, at least one usable
// episode, and — in default mode — every needed device online (else the backend
// refuses). With "use only available devices", offline devices are tolerated
// (their episodes are dropped) as long as ≥1 episode remains.
function dsCanGen(){
 const busy=datasetJob&&(datasetJob.status==='uploading'||datasetJob.status==='processing');
 if(busy||datasetSel.size===0||!(datasetName||'').trim())return false;
 const c=datasetEpisodeCounts();
 if(datasetOnlyAvailable)return c.avail>0;
 if(c.total===0)return false;
 for(const id of datasetDevices().keys())if(!deviceOnline(id))return false;
 return true;}
async function generateDataset(){
 const ids=[...datasetSel];
 if(!ids.length)return;
 const name=(datasetName||'').trim();
 if(!name){alert('Enter a dataset name');return;}
 const target=datasetOwner?datasetOwner+'/'+name:name;
 // The allow-list only rides along when something was actually picked: an empty
 // list means "every episode" server-side, so sending one for an untouched
 // selection would be the same request with more bytes — and a curated one must
 // never arrive as empty, which is why dsCanGen refuses to build from nothing.
 const sel=TASKS.filter(t=>datasetSel.has(t.id));
 const episode_ids=sel.some(dsRestricted)?sel.flatMap(t=>dsEpisodesOf(t).map(ep=>ep.episode_id)):[];
 const r=await fetch('/api/fleet/lerobot-dataset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_ids:ids,name:target,private:datasetPrivate,only_available:datasetOnlyAvailable,roles:datasetUseOnly,episode_ids})});
 const j=await r.json().catch(()=>({}));
 if(!r.ok){alert(errText(j)||'Failed to start dataset generation');return;}
 datasetJobId=j.job_id;
 datasetJob={status:'uploading',message:'Starting… (this can take several minutes)',cancellable:true};renderDatasetBar();
 pollDatasetJob(j.job_id);}
function pollDatasetJob(jobId){
 if(datasetPollTimer)clearTimeout(datasetPollTimer);
 const step=async()=>{
  try{
   const r=await fetch('/api/fleet/lerobot-dataset/'+jobId);
   if(r.ok){datasetJob=await r.json();renderDatasetBar();
    if(datasetJob.status==='done'||datasetJob.status==='error'||datasetJob.status==='cancelled'||datasetJob.status==='raw_ready')return;}
  }catch(e){}
  datasetPollTimer=setTimeout(step,2000);
 };
 step();}
// Pick up a build that's already running server-side — after a page reload, or one
// started from another browser/operator. Without this the dataset bar could only
// offer Cancel for a build launched in THIS page's lifetime.
async function adoptLiveDatasetJob(){
 try{
  const r=await fetch('/api/fleet/lerobot-datasets');
  if(!r.ok)return;
  const live=((await r.json()).jobs||[]).find(j=>!j.check);  // a check belongs to the session panel
  if(!live)return;
  datasetJobId=live.id;datasetJob=live;renderDatasetBar();pollDatasetJob(live.id);
 }catch(e){}}
// Cancel from the build page. Covers every device involved — the server cancels the
// whole job, never one device's share (a partially-uploaded raw is unusable).
async function cancelDataset(){
 if(!datasetJobId)return;
 if(!confirm('Cancel this dataset build on ALL the devices involved?\\n\\nWhatever was already uploaded is kept in the raw repository on HF (nothing is deleted).'))return;
 const r=await fetch('/api/fleet/lerobot-dataset/'+datasetJobId+'/cancel',{method:'POST'});
 const j=await r.json().catch(()=>({}));
 if(!r.ok){alert(errText(j)||'Failed to cancel the build');return;}
 pollDatasetJob(datasetJobId);  // picks up the "cancelled" status and stops polling
 await refresh();}
// Cancel from the fleet device list — the way back in when the build page was left
// or reloaded. Acts on the whole build, not just this device.
async function cancelDatasetForDevice(id){
 const d=DEVICES.find(x=>x.device_id===id);
 if(!d||!d.dataset_job)return;
 const what=d.dataset_check?'trajectory check':'dataset build';
 if(!confirm(`Cancel the ${what} ${d.name} is working on?\\n\\nThis cancels it on ALL the devices involved, not just this one.`))return;
 const r=await fetch('/api/fleet/lerobot-dataset/'+d.dataset_job+'/cancel',{method:'POST'});
 if(!r.ok){const j=await r.json().catch(()=>({}));alert(errText(j)||'Failed to cancel the build');return;}
 if(datasetJobId===d.dataset_job)pollDatasetJob(datasetJobId);
 if(slamJobId===d.dataset_job)pollSlamCheck(slamJobId);
 await refresh();}
// Per-role device dropdowns for a task's required devices. Selected values are
// baked in as `selected` attributes so the 3s refresh re-render keeps the picks.
function pickRole(role,val){launchRolePick[role]=val;renderTaskList();}
function launchRolesHtml(t){
 return (t.device_signature||[]).map(role=>{
  const opts=DEVICES.filter(d=>slotOf(d)===role).map(d=>{
   const sel=launchRolePick[role]===d.device_id?' selected':'';
   // Offline devices can't be picked — a session needs every device online.
   // Nor can a device busy with a dataset upload/conversion.
   const busy=deviceBusyForRec(d.device_id);
   const dis=(!d.online||busy)?' disabled':'';
   const suffix=!d.online?' (offline)':busy?` (${deviceActivity(d.device_id)==='processing'?'converting':'uploading'})`:'';
   return `<option value="${d.device_id}"${sel}${dis}>${esc(d.name)}${suffix}</option>`;
  }).join('');
  const none=launchRolePick[role]?'':' selected';
  return `<label class="role-pick"><span class="sig-label">${ROLE_LABEL[role]}</span>`+
   `<select class="role-select" onchange="pickRole('${role}',this.value)">`+
   `<option value=""${none}>— select —</option>${opts}</select></label>`;
 }).join('');}
// Expanded task detail: the session launcher (role dropdowns + Launch button),
// then a collapsed-by-default session history.
function taskDetailHtml(t){
 const running=SESSIONS.some(s=>s.status==='open');
 // Episodes recorded for this task, newest first — sourced from the devices'
 // reports so they show (and name their peers, even offline ones) for ANY
 // operator, regardless of which account did the acquisition.
 const eps=(t.episodes||[]).slice().sort((a,b)=>b.episode_id.localeCompare(a.episode_id));
 const roles=t.device_signature||[];
 const missing=roles.filter(role=>!launchRolePick[role]);
 const offlinePick=roles.filter(role=>launchRolePick[role]&&!deviceOnline(launchRolePick[role]));
 const busyPick=roles.filter(role=>launchRolePick[role]&&deviceBusyForRec(launchRolePick[role]));
 let launchBtn;
 if(running){
  launchBtn=`<button class="validate" disabled>Launch session</button><span class="muted launch-hint">A session is already running — close it first.</span>`;
 }else if(missing.length){
  launchBtn=`<button class="validate" disabled>Launch session</button><span class="muted launch-hint">Select a device for every role — required to launch.</span>`;
 }else if(offlinePick.length){
  launchBtn=`<button class="validate" disabled>Launch session</button><span class="muted launch-hint">A selected device is offline — every device must be online.</span>`;
 }else if(busyPick.length){
  launchBtn=`<button class="validate" disabled>Launch session</button><span class="muted launch-hint">A selected device is busy processing a dataset — wait for it to finish.</span>`;
 }else{
  launchBtn=`<button class="validate" onclick='launchSession("${t.id}")'>Launch session</button>`;
 }
 const histOpen=historyOpenId===t.id;
 // Read-only here: reorganising takes (move to another task, delete) is its own
 // page, reached from the button in the head below — the same split as the inbox,
 // where the banner leads to the triage page rather than sorting in place.
 const epRows=eps.length?eps.map(ep=>`<div class="hist-row">${epLineHtml(ep)}</div>`).join('')
  :'<div class="muted" style="font-size:.82rem">No episode recorded yet.</div>';
 // Task-level repair: count incomplete episodes and how many can be filled now.
 const incN=eps.filter(e=>e.incomplete).length;
 const fillN=eps.filter(e=>e.fillable).length;
 const fillCtrl=fillN
  ?`<button class="fill-task" title="Backfill who recorded these episodes onto the connected grabettes" onclick='fillTaskDevices("${t.id}")'>Fill devices (${fillN})</button>`
  :(incN?`<span class="fill-ep-hint" title="Connect the grabettes that recorded these episodes to repair them">${warnIcon()} ${incN} incomplete — connect their grabettes</span>`:'');
 // Entry point to the manage page. In the head, not inside the accordion: it acts
 // on the task's episodes whether or not the list happens to be unfolded.
 const manageCtrl=eps.length?`<button class="fill-task" title="Move these episodes to another task, or delete them" onclick='openManage("${t.id}")'>Manage episodes</button>`:'';
 // Pairing-issues accordion — only for a task that currently has orphans. Same
 // style as the global banner, nested under the Episodes accordion.
 const g=orphanFor(t.name);
 const issuesOpen=issuesOpenId===t.id;
 const issues=g?`<div class="hist-toggle warn" onclick='toggleIssues("${t.id}")'>${issuesOpen?'▾':'▸'} ${warnIcon()} Pairing issues (${g.count})</div>${issuesOpen?orphanCardHtml(g):''}`:'';
 // Device groups: which device combinations recorded this task, and how many
 // episodes each. From connected devices' reports (same scope as Episodes).
 const combos=taskCombos(t);
 const groupsOpen=groupsOpenId===t.id;
 const groups=combos.length?`<div class="hist-toggle" onclick='toggleGroups("${t.id}")'>${groupsOpen?'▾':'▸'} Device groups (${combos.length})</div>${groupsOpen?combos.map(comboRowHtml).join(''):''}`:'';
 const connNote='<span class="muted" style="font-weight:400"> · from connected devices</span>';
 return `<div class="task-detail">
   <div class="ds-step" style="margin:.2rem 0 .6rem"><span class="step-num">2</span> Select the devices to use</div>
   <div class="task-launch">${launchRolesHtml(t)}<div class="editor-actions">${launchBtn}</div></div>
   ${groups}
   <div class="hist-head"><div class="hist-toggle" onclick='toggleHistory("${t.id}")'>${histOpen?'▾':'▸'} Episodes (${eps.length})${connNote}</div><div class="hist-acts">${fillCtrl}${manageCtrl}</div></div>
   ${histOpen?epRows:''}
   ${issues}
  </div>`;}
// One episode as a line: when it was recorded, who recorded it (with the marker
// for a member this fleet can't reach), and the incomplete flag. Shared by the
// read-only list in the task detail and the manage page's selectable rows.
function epLineHtml(ep){
 const names=Object.entries(ep.members||{}).map(([role,m])=>`<span class="pill hand" title="${role}">${role[0].toUpperCase()}</span> ${esc(m.name)}${deviceOnline(m.device_id)?'':' <span class="muted">offline</span>'}`).join(' ')||'<span class="muted">no devices recorded</span>';
 // Episodes saved before the device persisted their members show up incomplete;
 // repair is bulk, at the task level (see the "Fill devices" button).
 const mark=ep.incomplete?` <span class="fill-ep-hint" title="Recorded before members were saved — use “Fill devices” to repair">${warnIcon()} incomplete</span>`:'';
 return `<span class="muted">${fmtTs(ep.started_at)}</span> · ${names}${mark}`;}
// Both actions on this page reach EVERY grabette that holds the take, and that is
// what makes them safe: refiling (or deleting) one copy of a pair while its peer
// stays offline leaves the peer filing it under the old task — the split the
// fleet then reports (see splitCardHtml) — or holding an orphan. So a take with
// an unreachable member is listed but not selectable. One with no members
// recorded is out too: we don't know who holds it, which is what "Fill devices"
// repairs.
function epReachable(ep){
 const ids=Object.values(ep.members||{}).map(w=>w&&w.device_id).filter(Boolean);
 return ids.length>0&&ids.every(deviceOnline);}
function epHolders(eps){
 return [...new Set(eps.flatMap(ep=>Object.values(ep.members||{}).map(w=>w&&w.device_id).filter(Boolean)))];}
function epPickable(t){return (t.episodes||[]).filter(epReachable);}
function epPicked(t){const s=epSel[t.id]||new Set();return epPickable(t).filter(ep=>s.has(ep.episode_id));}
function epToggleAll(tid,on){
 const t=TASKS.find(x=>x.id===tid);if(!t)return;
 epSel[tid]=on?new Set(epPickable(t).map(e=>e.episode_id)):new Set();
 renderManageBody();}
// Patches the toolbar mid-sweep, so the counts and both buttons follow the drag
// without rebuilding the list under the cursor (see epApplyPick).
function epSyncToolbar(tid){
 const t=TASKS.find(x=>x.id===tid);if(!t)return;
 const n=epPicked(t).length;
 const mv=$('ep-move'),dl=$('ep-del'),all=$('ep-all');
 if(mv){mv.textContent=moveLabel(n,epMoveTask);mv.disabled=!n;}
 if(dl){dl.textContent=delLabel(n);dl.disabled=!n;}
 if(all)all.checked=n>0&&n===epPickable(t).length;}
// ── Manage episodes ──
// The task-side twin of the triage page: one task's takes, read straight down,
// with the toolbar pinned above. A page and not a panel in the accordion for the
// same reason triage is one — reorganising past recordings is a different
// activity from recording, so it takes the screen rather than hiding three levels
// down in a task row.
function manageTask(){return TASKS.find(t=>t.id===manageTaskId)||null;}
function openManage(tid){manageTaskId=tid;epSel={};epMoveTask='';renderManage();window.scrollTo(0,0);}
function closeManage(){manageTaskId='';epSel={};epMoveTask='';renderManage();}
function renderManage(){
 // The task can go while the page is open (deleted on a device, or its last
 // reporter dropped off) — leave rather than show an empty page.
 if(manageTaskId&&!manageTask())manageTaskId='';
 applyLayout();
 const t=manageTask();
 if(!t){$('manage-body').innerHTML='';return;}
 $('count-manage').textContent=(t.episodes||[]).length;
 // textContent, not a template: a task name can contain anything.
 $('manage-note').textContent=`Episodes recorded for “${t.name}”. Tick the takes to `
  +`file them under another task, or delete them for good. Both are carried out on every `
  +`grabette that holds a take, so only takes whose grabettes are all online can be picked.`;
 renderManageBody();}
function renderManageBody(){
 const t=manageTask();if(!t)return;
 const eps=(t.episodes||[]).slice().sort((a,b)=>b.episode_id.localeCompare(a.episode_id));
 const sel=epSel[t.id]||new Set();
 const pickable=eps.filter(epReachable);
 const picked=pickable.filter(ep=>sel.has(ep.episode_id));
 // Destination list, from the roles the SELECTED takes have in common (all
 // pickable ones, as a preview, while nothing is ticked) — same rule as the
 // inbox: a task whose signature those takes can't satisfy would be refused by
 // the fleet, so offering it is a button that only ever alerts. This task is left
 // out: "moving" a take onto itself is a no-op you'd discover by trying it.
 const basis=picked.length?picked:pickable;
 const roles=basis.reduce((acc,ep)=>{
  const r=Object.keys(ep.members||{});
  return acc===null?r:acc.filter(x=>r.includes(x));},null)||[];
 const fits=TASKS.filter(x=>x.id!==t.id&&x.name&&(x.device_signature||[]).every(s=>roles.includes(s)));
 // Keep the chosen destination across re-renders (the poll re-renders every few
 // seconds), falling back only when it stops being a valid one.
 if(!fits.some(x=>x.name===epMoveTask))epMoveTask=fits.length?fits[0].name:'';
 const taskOpts=fits.map(x=>
   `<option value="${esc(x.name)}" ${x.name===epMoveTask?'selected':''}>${esc(x.name)}</option>`).join('');
 // Who is holding back the takes that can't be picked: the members this fleet has
 // no CONNECTED device for. Named, not counted — "sim-test-3 is not connected" is
 // the whole diagnosis, where "1 unavailable" sends the operator hunting. A device
 // the fleet has never seen (retired, or re-flashed with a new id) shows as the
 // raw id, which is what makes a stale stamp recognisable at all.
 const blockers=[...new Set(eps.filter(ep=>!epReachable(ep))
   .flatMap(ep=>Object.values(ep.members||{}).map(w=>w&&w.device_id))
   .filter(id=>id&&!deviceOnline(id)))];
 const why=[];
 if(blockers.length)why.push(`${blockers.map(id=>esc(deviceName(id))).join(', ')} ${blockers.length>1?'are':'is'} not connected`);
 if(eps.some(ep=>!Object.values(ep.members||{}).some(w=>w&&w.device_id)))why.push('no grabette was recorded for it');
 const stuck=eps.length-pickable.length;
 const dest=!pickable.length
  ?`<span class="unassigned-detail">Nothing here can be moved or deleted yet — both act on every grabette that holds a take, so they all have to be connected.</span>`
  :taskOpts
   ?`<span class="triage-label">Select the destination task:</span>
     <select class="uassign-sel" onchange="epMoveTask=this.value;renderManageBody()">${taskOpts}</select>
     <button class="fill-task" id="ep-move" ${picked.length?'':'disabled'} onclick="moveTaskEpisodes('${t.id}')">${esc(moveLabel(picked.length,epMoveTask))}</button>`
   :`<span class="unassigned-detail">No other task matches ${roles.length?'these recordings’ devices ('+roles.map(esc).join(', ')+')':'these recordings'} — create one to move them there.</span>`;
 const rows=eps.map(ep=>{
  const line=epLineHtml(ep);
  // A take that can't be acted on keeps a plain row: its reason is already ON the
  // row (an "offline" member, or "no devices recorded"), so a disabled checkbox
  // would add a dead control and no information.
  if(!epReachable(ep))return `<div class="hist-row">${line}</div>`;
  return `<div class="hist-row tep" data-eid="${ep.episode_id}"
    onmousedown="epDragStart('manage','${t.id}','${ep.episode_id}',event)"
    onmouseenter="epDragOver('manage','${t.id}','${ep.episode_id}')">
    <input type="checkbox" ${sel.has(ep.episode_id)?'checked':''}>${line}</div>`;
 }).join('')||'<div class="muted" style="font-size:.82rem">No episode recorded yet.</div>';
 $('manage-body').innerHTML=`<div class="triage-toolbar">
    <div class="triage-row">
     <span class="triage-label">${esc(t.name)}</span>
     <span class="triage-count">${pickable.length} of ${eps.length} episode${eps.length>1?'s':''} can be acted on</span>
     <label class="uall"><input type="checkbox" id="ep-all" ${picked.length&&picked.length===pickable.length?'checked':''} onchange="epToggleAll('${t.id}',this.checked)"> Select all</label>
     ${stuck?`<span class="fill-ep-hint" title="Both actions reach every grabette that holds a take — otherwise the peer keeps it filed here, or holds an orphan">${warnIcon()} ${stuck} unavailable — ${why.join('; ')||'their grabettes are not connected'}</span>`:''}
    </div>
    <div class="triage-row">
     ${dest}
    </div>
    <div class="triage-row">
     <button class="del-confirm" id="ep-del" ${picked.length?'':'disabled'} onclick="deleteTaskEpisodes('${t.id}')">${delLabel(picked.length)}</button>
    </div>
   </div>
   <div class="unassigned-eps">${rows}</div>`;}
// Refile the picked takes under another task. Dispatched to every grabette that
// holds them, in ONE command each, so both sides of a pair land on the same name
// and no split is created — the same endpoint, and the same reason, as
// resolveSplit. The fleet refuses a destination the takes can't satisfy; the
// dropdown already filtered those out, so an alert here is a real disagreement
// (a device reporting fewer roles than it recorded) and is worth showing.
async function moveTaskEpisodes(tid){
 const t=TASKS.find(x=>x.id===tid);if(!t||!epMoveTask)return;
 const eps=epPicked(t);
 if(!eps.length)return;
 const ids=eps.map(ep=>ep.episode_id);
 const holders=epHolders(eps);
 if(!confirm(`Move ${ids.length} episode(s) from “${t.name}” to “${epMoveTask}” on ${holders.map(deviceName).join(', ')}?`))return;
 const r=await fetch('/api/fleet/episodes/assign',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({task_name:epMoveTask,device_ids:holders,episode_ids:ids})});
 if(!r.ok){const e=await r.json().catch(()=>({}));alert(errText(e)||'Could not move these episodes');}
 else epSel[tid]=new Set();  // moved: drop the selection, stay on the page
 await refresh();}
// The other half of managing a task's takes: discard them — registry entry AND
// files, on every grabette that holds them. Same endpoint as the inbox's delete;
// what differs is only where the ids come from. Irreversible, hence the wording
// of the confirmation: this is not "remove from the task" (that's the move).
async function deleteTaskEpisodes(tid){
 const t=TASKS.find(x=>x.id===tid);if(!t)return;
 const eps=epPicked(t);
 if(!eps.length)return;
 const ids=eps.map(ep=>ep.episode_id);
 const holders=epHolders(eps);
 if(!confirm(`Permanently delete ${ids.length} episode(s) of “${t.name}” on ${holders.map(deviceName).join(', ')}? The files go too.`))return;
 const r=await fetch('/api/fleet/episodes/delete',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({device_ids:holders,episode_ids:ids})});
 if(!r.ok){const e=await r.json().catch(()=>({}));alert(errText(e)||'Could not delete these episodes');}
 else epSel[tid]=new Set();
 await refresh();}
function taskEpsHtml(total){return `<div class="task-eps"><span class="ep-count" title="Total episodes recorded for this task">${total} ${total===1?'episode':'episodes'}</span></div>`;}
function taskSubHtml(t){return `<div class="task-sub"><span class="muted">Required devices:</span> ${sigBadges(t.device_signature)}${t.description?` · ${esc(t.description)}`:''}</div>`;}
function renderTaskList(){
 // Don't clobber an inline edit-in-progress on the periodic refresh — rebuilding
 // innerHTML would wipe what the user is typing. Skip while its editor is live.
 if(taskEditId && $('edit-name'))return;
 const el=$('task-list');$('count-tasks').textContent=TASKS.length;
 const step1=$('ds-step1');
 if(!TASKS.length){el.className='muted empty';el.textContent='No tasks yet.';step1.style.display='none';renderAdvanced();renderDatasetBar();return;}
 el.className='';
 // Step 1 header, styled the same in both modes but worded for the task at hand.
 step1.style.display='flex';
 step1.innerHTML='<span class="step-num">1</span> '+(selectMode?'Select tasks to include':'Select the task to record');
 renderAdvanced();
 if(selectMode){
  el.innerHTML=TASKS.map(t=>{
   const checked=datasetSel.has(t.id);
   const disabled=!checked&&!datasetCompatible(t);
   const all=(t.episodes||[]).length;
   const open=dsPickOpen===t.id;
   // Advanced selection, per task. stopPropagation because the row itself toggles
   // the task in or out of the build — the icon must not do that too. Offered on
   // any task with episodes, picked or not: the restriction is remembered and
   // applies if the task goes in.
   const pick=all?`<div class="task-acts">
       <button class="act-edit${open?' on':''}" title="Advanced selection — choose which episodes of this task go into the dataset" onclick='event.stopPropagation();toggleDsPick("${t.id}")'>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="21" y1="4" x2="14" y2="4"/><line x1="10" y1="4" x2="3" y2="4"/><line x1="21" y1="12" x2="12" y2="12"/><line x1="8" y1="12" x2="3" y2="12"/><line x1="21" y1="20" x2="16" y2="20"/><line x1="12" y1="20" x2="3" y2="20"/><line x1="14" y1="2" x2="14" y2="6"/><line x1="8" y1="10" x2="8" y2="14"/><line x1="16" y1="18" x2="16" y2="22"/></svg>
       </button>
      </div>`:'';
   // A curated task says so on its row: "20 episodes" on a task the build will
   // take 12 of is the one number that must not be left standing.
   const epsHtml=dsRestricted(t)
    ?`<div class="task-eps"><span class="ep-count" title="Advanced selection: only some of this task's episodes go into the dataset">${dsEpisodesOf(t).length} of ${all} episodes</span></div>`
    :taskEpsHtml(all);
   return `<div class="task-block">
    <div class="task-row select${disabled?' disabled':''}" ${disabled?'':`onclick='toggleDatasetPick("${t.id}")'`}>
     <div class="task-top">
      <input type="checkbox" class="task-check" ${checked?'checked':''} ${disabled?'disabled':''}>
      <span class="tname">${esc(t.name)}</span>
      ${pick}
     </div>
     ${epsHtml}
     ${taskSubHtml(t)}
    </div>${open?dsPickHtml(t):''}
   </div>`;
  }).join('');
  renderDatasetBar();return;
 }
 el.innerHTML=TASKS.map(t=>{
  const editing=taskEditId===t.id;
  const deleting=taskDeleteId===t.id;
  const detail=editing?taskEditorHtml(t):(deleting?taskDeleteHtml(t):(expandedTaskId===t.id?taskDetailHtml(t):''));
  const hasIssues=!!orphanFor(t.name);
  return `<div class="task-block">
    <div class="task-row" ${(editing||deleting)?'':`onclick='toggleTaskDetail("${t.id}")'`}>
     <div class="task-top">
      <span class="tname">${esc(t.name)}${hasIssues?`<span class="task-warn" title="Episode pairing issues">${warnIcon()}</span>`:''}</span>
      <div class="task-acts">
       <button class="act-edit" title="Edit task" onclick='event.stopPropagation();editTask("${t.id}")'>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
       </button>
       <button class="del-icon" title="Delete task" onclick='event.stopPropagation();deleteTask("${t.id}")'>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
       </button>
      </div>
     </div>
     ${taskEpsHtml(t.episode_count||0)}
     ${taskSubHtml(t)}
    </div>${detail}
   </div>`;
 }).join('');
 renderDatasetBar();}
// ── Sessions ──
// The launcher now lives inside each task's expanded detail (taskDetailHtml):
// pick a device per required role, then Launch. A single session runs at a time.
async function launchSession(taskId){
 const t=TASKS.find(x=>x.id===taskId);if(!t)return;
 const body={task_id:taskId};
 for(const role of (t.device_signature||[])){
  const id=launchRolePick[role];
  if(!id){alert('Select a device for '+ROLE_LABEL[role]);return;}
  if(!deviceOnline(id)){alert(deviceName(id)+' is offline — a session needs every device online.');return;}
  body[role]=id;
 }
 const r=await fetch('/api/fleet/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!r.ok){const j=await r.json().catch(()=>({}));alert(errText(j)||'Failed to launch session');return;}
 expandedTaskId='';launchRolePick={left:'',right:'',casquette:''};await refresh();}
// Single record toggle (phone-camera style): start when idle, stop when
// recording/initializing; ignored mid-stop.
function episodeToggle(id){
 const s=SESSIONS.find(x=>x.id===id);if(!s)return;
 const p=sessionPhase(s);
 if(p==='idle'){
  // Don't let a new episode start on top of a device that may still be recording
  // the previous one — that desyncs the pair for good. Warn, but leave it to the
  // operator (a device on older firmware may simply never report its stop).
  const un=s.stop_unconfirmed||[];
  if(un.length&&!confirm(`${un.map(m=>m.name).join(', ')} never confirmed the stop and may still be recording. Re-send the stop first — start a new episode anyway?`))return;
  episodeStart(id);
 }
 else if(p==='recording'||p==='initializing')episodeStop(id);}
async function episodeStart(id){
 const r=await fetch(`/api/fleet/sessions/${id}/episode/start`,{method:'POST'});
 const j=await r.json().catch(()=>({}));
 if(!r.ok){alert(errText(j)||'Failed to start episode');return;}
 await refresh();}  // the "Initializing…" state shows the pre-T0 wait — no alert needed
async function episodeStop(id){await fetch(`/api/fleet/sessions/${id}/episode/stop`,{method:'POST'});await refresh();}
async function deleteLastEpisode(id){
 if(!confirm('Delete the last episode of this session on ALL its devices? This cannot be undone.'))return;
 const r=await fetch(`/api/fleet/sessions/${id}/episode/delete-last`,{method:'POST'});
 if(!r.ok){const j=await r.json().catch(()=>({}));alert(errText(j)||'Failed to delete episode');return;}
 await refresh();
 const s=SESSIONS.find(x=>x.id===id);if(s)epChimed.set(id,epTens(s));}  // deletion walked the count back
async function fillTaskDevices(taskId){
 const r=await fetch(`/api/fleet/tasks/${encodeURIComponent(taskId)}/fill-devices`,{method:'POST'});
 if(!r.ok){const j=await r.json().catch(()=>({}));alert(errText(j)||'Failed to fill devices');return;}
 await refresh();}
async function closeSession(id){if(!confirm('Close this session?'))return;await fetch(`/api/fleet/sessions/${id}/close`,{method:'POST'});await refresh();}
async function deleteSession(id){if(!confirm('Delete this session and its manifest?'))return;await fetch(`/api/fleet/sessions/${id}`,{method:'DELETE'});await refresh();}
// The session's capture phase, mirroring the grabette LEDs (off/blink/solid/
// fast-blink): idle → nothing running; initializing → episode dispatched, still
// before the shared T0 (devices warming/waiting); recording → past T0; stopping
// → devices tearing down + muxing. initializing↔recording is derived from T0;
// stopping is reported by the fleet and ends when the devices actually finish
// (s.stopping is cleared as each stop_capture reports its result).
function sessionPhase(s){
 if(s.recording){
  const ep=(s.episodes&&s.episodes.length)?s.episodes[s.episodes.length-1]:null;
  const t0=ep&&ep.start_at_utc?Date.parse(ep.start_at_utc):0;
  return (t0&&Date.now()<t0)?'initializing':'recording';
 }
 return s.stopping?'stopping':'idle';}
function phasePill(s){
 // State pill for the panel header — the recording duration lives above the
 // record button, not in the pill.
 const p=sessionPhase(s);
 if(p==='recording')return '<span class="pill rec">● Recording</span>';
 if(p==='initializing')return '<span class="pill init">◌ Initializing…</span>';
 if(p==='stopping')return '<span class="pill stopping">◍ Stopping…</span>';
 return '<span class="pill idle">○ Idle</span>';}
// Episodes that lost their pair (a peer deleted the task while this device was
// offline; on reconnect its copies are orphaned). Cleanup deletes only the ones
// whose deleting peer is back online — those still paired with an offline device
// are left untouched.
function orphanFor(name){return ORPHANS.find(g=>g.task===name)||null;}  // group for a task, or null
function orphanCardHtml(g){
 const i=ORPHANS.indexOf(g);
 const dels=g.deleted_by.map(esc).join(', ')||'a peer';
 const hold=g.holders.map(esc).join(', ');
 const head=g.whole_task
  ?`<b>${esc(g.task)}</b> — task deleted on <b>${dels}</b>; ${g.count} episode(s) still here`
  :`<b>${esc(g.task)}</b> — ${g.count} episode(s) lost their pair`;
 const detail=g.whole_task
  ?`This task was deleted (with its recordings) on <b>${dels}</b>, but ${g.count} orphaned episode(s) remain on <b>${esc(hold)}</b>.`
  :`Deleted on <b>${dels}</b>, still present on <b>${esc(hold)}</b>.`;
 const btn=g.whole_task
  ?`Delete the task and its ${g.count} episode(s) on ${esc(hold)}`
  :`Delete the ${g.count} orphaned episode(s) on ${esc(hold)}`;
 return `<div class="orphan-item">
    <div class="orphan-head">${head}</div>
    <div class="orphan-detail">${detail}</div>
    <button class="del-confirm" onclick="cleanupOrphans(${i})">${btn}</button>
   </div>`;}
// An episode two grabettes file under DIFFERENT tasks. Same amber family as the
// orphans — it is an inconsistency to repair, not an inbox. Repair = pick which
// task name is right; the others' holders are told to refile onto it.
function splitCardHtml(g){
 const i=SPLITS.indexOf(g);
 const lines=g.tasks.map(t=>`<div class="orphan-detail">“${esc(t)}” on <b>${(g.by_task[t]||[]).map(d=>esc(d.name)).join(', ')}</b></div>`).join('');
 const btns=g.tasks.map(t=>`<button class="fill-task" onclick='resolveSplit(${i},${JSON.stringify(t)})'>File all under “${esc(t)}”</button>`).join(' ');
 return `<div class="orphan-item">
    <div class="orphan-head">${g.count} episode${g.count>1?'s':''} filed under ${g.tasks.length} different tasks</div>
    ${lines}
    <div class="orphan-detail">The fleet merges reports by task name, so these show up under every one of those tasks and a dataset built from any of them would pull the same takes.</div>
    <div class="split-actions">${btns}</div>
   </div>`;}
async function resolveSplit(i,task){
 const g=SPLITS[i];if(!g)return;
 const holders=Object.values(g.by_task||{}).flat();
 const ids=[...new Set(holders.map(d=>d.device_id))];
 const names=[...new Set(holders.map(d=>d.name))].join(', ');
 if(!confirm(`File ${g.episode_ids.length} episode(s) under “${task}” on every grabette that holds them (${names})?`))return;
 const r=await fetch('/api/fleet/episodes/assign',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({task_name:task,device_ids:ids,episode_ids:g.episode_ids})});
 if(!r.ok){const e=await r.json().catch(()=>({}));alert(errText(e)||'Could not refile these episodes');}
 await refresh();}
// Collapsed accordion under the tabs summarising all pairing issues.
function renderOrphans(){
 const bar=$('orphan-bar');
 if(!ORPHANS.length&&!SPLITS.length){bar.style.display='none';bar.innerHTML='';orphansOpen=false;return;}
 bar.style.display='block';
 const n=ORPHANS.length,sn=SPLITS.reduce((a,g)=>a+g.count,0);
 const parts=[];
 if(n)parts.push(`${n} task${n>1?'s':''} with episode pairing issues`);
 if(sn)parts.push(`${sn} episode${sn>1?'s':''} filed under different tasks`);
 bar.innerHTML=`<div class="orphan-acc">
   <div class="orphan-acc-head" onclick="orphansOpen=!orphansOpen;renderOrphans()">
    <span>${warnIcon()} ${parts.join(' · ')}</span>
    <span class="orphan-caret">${orphansOpen?'▾':'▸'}</span>
   </div>
   ${orphansOpen?`<div class="orphan-acc-body">${ORPHANS.map(orphanCardHtml).join('')}${SPLITS.map(splitCardHtml).join('')}</div>`:''}
  </div>`;}
// Unassigned recordings — takes filed under no task. Three provenances land here,
// which is why the wording stays neutral: a button press outside any session, the
// episodes of a deleted task falling back, and episodes migrated from the old
// on-disk layout. Read-only for now: filing them to a task is the next step.
// One grabette at a time, picked in the toolbar. Stacking every device's list made
// the page a scroll marathon, and the takes of one grabette are what an operator
// actually works through in one go.
function triageGroup(){
 if(!UNASSIGNED.length)return null;
 return UNASSIGNED.find(g=>g.device_id===triageDev)||UNASSIGNED[0];}
function renderTriageBody(){
 const g=triageGroup();
 const body=$('triage-body');
 if(!g){body.innerHTML='';return;}
 triageDev=g.device_id;
 const sel=uSel[g.device_id]||new Set();
 const picked=g.episodes.filter(ep=>sel.has(ep.episode_id));
 // Roles of the SELECTED takes (all shown, as a preview, while nothing is ticked).
 // An inbox can hold mixed shapes — a mono button press next to a pair that fell
 // back from a deleted task — so the compatible targets depend on the selection.
 const basis=picked.length?picked:g.episodes;
 const roles=basis.reduce((acc,ep)=>{
  const r=Object.keys(ep.members||{});
  return acc===null?r:acc.filter(x=>r.includes(x));},null)||[];
 // Only tasks these takes can actually satisfy: a mono recording in a bimanual
 // task would be counted yet permanently incomplete, so the fleet refuses it, and
 // offering it here would be a button that only ever alerts. A task with no
 // signature yet is always fine — filing seeds it.
 const fits=TASKS.filter(t=>t.name&&(t.device_signature||[]).every(s=>roles.includes(s)));
 // Keep the operator's chosen target across re-renders (every tick re-renders),
 // falling back only when it stops being a valid destination.
 if(!fits.some(t=>t.name===triageTask))triageTask=fits.length?fits[0].name:'';
 const taskOpts=fits.map(t=>
   `<option value="${esc(t.name)}" ${t.name===triageTask?'selected':''}>${esc(t.name)}</option>`).join('');
 const devOpts=UNASSIGNED.map(x=>
   `<option value="${x.device_id}" ${x.device_id===g.device_id?'selected':''}>${esc(x.device)}</option>`).join('');
 const allOn=g.episodes.length>0&&picked.length===g.episodes.length;
 const dest=taskOpts
  ?`<span class="triage-label">Select the destination task:</span>
    <select class="uassign-sel" onchange="triageTask=this.value;renderTriageBody()">${taskOpts}</select>
    <button class="fill-task" id="triage-move" ${picked.length?'':'disabled'} onclick="fileUnassigned()">${esc(moveLabel(picked.length,triageTask))}</button>`
  :`<span class="unassigned-detail">No task matches ${roles.length?'the selected recordings’ devices ('+roles.map(esc).join(', ')+')':'these recordings'} — create one to file them.</span>`;
 const rows=g.episodes.map(ep=>{
  // The device is chosen at the top, so repeating it on every line says nothing.
  // Peers DO earn a mention: a take recorded with another grabette is a different
  // animal from a solo one, and that is exactly what you can't tell otherwise.
  const peers=Object.entries(ep.members||{})
    .filter(([,m])=>m&&m.device_id&&m.device_id!==g.device_id)
    .map(([role,m])=>`<span class="pill hand" title="${role}">${role[0].toUpperCase()}</span> ${esc(m.name)}`);
  const withWho=peers.length?` · <span class="muted">with</span> ${peers.join(' ')}`:'';
  // Spelled-out length rather than a bare mm:ss: the whole point of showing it is
  // telling a real take from a misfire, and "0:00" said nothing about a 0.8s press.
  const novid=ep.has_video?'':' · <span class="muted">no video</span>';
  return `<div class="hist-row uep" data-eid="${ep.episode_id}"
    onmousedown="epDragStart('inbox','${g.device_id}','${ep.episode_id}',event)"
    onmouseenter="epDragOver('inbox','${g.device_id}','${ep.episode_id}')">
    <input type="checkbox" ${sel.has(ep.episode_id)?'checked':''}>
    <span class="muted">${fmtTs(ep.started_at)}</span> · ${fmtLen(ep.duration_seconds)}${withWho}${novid}</div>`;
 }).join('');
 // The toolbar sticks below the fleet header, so the device, the select-all and the
 // destination stay reachable however far down the list you are — the list itself
 // is shown in full rather than inside its own little scroll box.
 body.innerHTML=`<div class="triage-toolbar">
    <div class="triage-row">
     <span class="triage-label">Select a device:</span>
     <select class="uassign-sel" onchange="triageDev=this.value;renderTriageBody()">${devOpts}</select>
     <span class="triage-count">${g.total} unassigned episode${g.total>1?'s':''}${g.total>g.episodes.length?` (${g.episodes.length} shown)`:''}</span>
     <label class="uall"><input type="checkbox" id="triage-all" ${allOn?'checked':''} onchange="uToggleAll('${g.device_id}',this.checked)"> Select all</label>
    </div>
    <div class="triage-row">
     ${dest}
    </div>
    <div class="triage-row">
     <button class="del-confirm" id="triage-del" ${picked.length?'':'disabled'} onclick="deleteUnassigned()">${delLabel(picked.length)}</button>
    </div>
   </div>
   <div class="unassigned-eps">${rows}</div>`;}
// Plain text (not HTML): the template escapes it, uSyncToolbar assigns it to
// textContent, and a task name can contain anything.
function moveLabel(n,task){
 const dest=task?`“${task}”`:'the destination task';
 return n?`Move ${n} episode${n>1?'s':''} to ${dest}`:`Move episodes to ${dest}`;}
function delLabel(n){return n?`Delete ${n} episode${n>1?'s':''}`:'Delete episodes';}
// Spelled out so the number is unambiguous — and so a sub-second misfire reads as
// "0.8 s" instead of the "0:00" that mm:ss produced for anything under a second.
function fmtLen(sec){
 const s=Number(sec)||0;
 if(s<60)return (s<10?s.toFixed(1):Math.round(s))+' s';
 return Math.floor(s/60)+' min '+String(Math.round(s%60)).padStart(2,'0')+' s';}
// Drag to sweep a run of takes: mousedown decides the direction (select or
// deselect, from the row you started on) and every row entered follows it. Rows are
// patched in place while dragging — a full re-render per row would be 50 rebuilds
// of the list — and the whole page re-renders once on mouseup.
//
// Two lists select takes this way — the unassigned inbox and a task's episodes —
// so the mechanics live here once and each list says which it is. They differ only
// in what a selection is keyed by (a device for the inbox, a task for the other),
// which row class carries it, and who owns the re-render. Only one of the two is
// ever on screen (triage takes the page over, see applyLayout), so their row
// selectors can't collide.
const EP_LISTS={
 inbox:{store:()=>uSel,row:'.uep',sync:k=>uSyncToolbar(k),render:()=>renderTriageBody()},
 manage:{store:()=>epSel,row:'.tep',sync:k=>epSyncToolbar(k),render:()=>renderManageBody()},
 // The advanced-selection panel lives IN the task list, so that is what redraws
 // when the sweep ends — no page of its own.
 dataset:{store:()=>dsEpSel,row:'.dsep',sync:k=>dsSyncToolbar(k),render:()=>renderTaskList()}};
let epDrag=null;
function epApplyPick(list,key,eid,want){
 const L=EP_LISTS[list],store=L.store();
 const s=store[key]||(store[key]=new Set());
 if(want)s.add(eid);else s.delete(eid);
 const row=document.querySelector(`${L.row}[data-eid="${eid}"] input`);
 if(row)row.checked=want;
 L.sync(key);}
function epDragStart(list,key,eid,ev){
 if(ev&&ev.button)return;            // left button only
 if(ev)ev.preventDefault();          // no text selection while sweeping
 const cur=EP_LISTS[list].store()[key];
 const want=!(cur&&cur.has(eid));
 epDrag={list,key,want};
 epApplyPick(list,key,eid,want);}
function epDragOver(list,key,eid){
 if(epDrag&&epDrag.list===list&&epDrag.key===key)epApplyPick(list,key,eid,epDrag.want);}
addEventListener('mouseup',()=>{if(epDrag){const l=epDrag.list;epDrag=null;EP_LISTS[l].render();}});
function uSyncToolbar(did){
 const g=UNASSIGNED.find(x=>x.device_id===did);if(!g)return;
 const n=(uSel[did]||new Set()).size;
 const mv=$('triage-move'),dl=$('triage-del'),all=$('triage-all');
 if(mv){mv.textContent=moveLabel(n,triageTask);mv.disabled=!n;}
 if(dl){dl.textContent=delLabel(n);dl.disabled=!n;}
 if(all)all.checked=n>0&&n===g.episodes.length;}
// Selection is per device and survives the poll's re-render (see uSel).
function uToggle(did,eid){
 const s=uSel[did]||(uSel[did]=new Set());
 if(s.has(eid))s.delete(eid);else s.add(eid);
 renderTriageBody();}
function uToggleAll(did,on){
 const g=UNASSIGNED.find(x=>x.device_id===did);if(!g)return;
 uSel[did]=on?new Set(g.episodes.map(e=>e.episode_id)):new Set();
 renderTriageBody();}
async function deleteUnassigned(){
 const g=triageGroup();if(!g)return;
 const ids=g.episodes.map(e=>e.episode_id).filter(e=>(uSel[g.device_id]||new Set()).has(e));
 if(!ids.length)return;
 if(!confirm(`Permanently delete ${ids.length} recording(s) on ${g.device}? The files go too.`))return;
 const r=await fetch('/api/fleet/episodes/delete',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({device_ids:[g.device_id],episode_ids:ids})});
 if(!r.ok){const e=await r.json().catch(()=>({}));alert(errText(e)||'Could not delete these recordings');}
 else uSel[g.device_id]=new Set();
 await refresh();}
async function fileUnassigned(){
 const g=triageGroup();if(!g||!triageTask)return;
 const ids=g.episodes.map(e=>e.episode_id).filter(e=>(uSel[g.device_id]||new Set()).has(e));
 if(!ids.length)return;
 if(!confirm(`File ${ids.length} recording(s) from ${g.device} under “${triageTask}”?`))return;
 const r=await fetch('/api/fleet/episodes/assign',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({task_name:triageTask,device_ids:[g.device_id],episode_ids:ids})});
 if(!r.ok){const e=await r.json().catch(()=>({}));alert(errText(e)||'Could not file these recordings');}
 else uSel[g.device_id]=new Set();  // filed: drop the selection, stay on the page
 await refresh();}
// Which of the three views owns the page. Triage is NOT a panel inside the record
// view: filing past recordings is a different activity from recording, so it takes
// the page over — mode tabs included — exactly as a running session does. One
// function decides it, so the session and triage renderers can't fight over the
// same elements (they would otherwise each set tasks-card/page-tabs themselves).
function syncStickyOffset(){
 const h=$('top-sticky');if(!h)return;
 document.documentElement.style.setProperty('--sticky-top',Math.round(h.getBoundingClientRect().height)+'px');}
addEventListener('resize',syncStickyOffset);
function applyLayout(){
 syncStickyOffset();
 const sess=SESSIONS.some(s=>s.status==='open');
 // A running session wins: nothing sends the operator off to triage mid-recording.
 const triage=triageOpen&&!sess&&UNASSIGNED.length>0;
 // Managing one task's episodes takes the page the same way, and yields to both:
 // nothing sends the operator off a running session or a triage they opened.
 const manage=!!manageTaskId&&!sess&&!triage;
 $('sessions-card').style.display=sess?'':'none';
 $('triage-card').style.display=triage?'':'none';
 $('manage-card').style.display=manage?'':'none';
 $('tasks-card').style.display=(sess||triage||manage)?'none':'';
 const tabs=$('tabs-sticky');if(tabs)tabs.style.display=(sess||triage||manage)?'none':'';
 // The entry banner sits ABOVE the tabs, and steps aside once you're on the page
 // it leads to (or while a session owns the screen).
 $('unassigned-bar').style.display=(!triage&&!sess&&!manage&&UNASSIGNED.length)?'block':'none';}
function openTriage(){triageOpen=true;uSel={};renderUnassigned();window.scrollTo(0,0);}
function closeTriage(){triageOpen=false;uSel={};renderUnassigned();}
// The banner is an entry point, not an accordion: one click opens the triage page.
function renderUnassigned(){
 if(!UNASSIGNED.length)triageOpen=false;  // nothing left to file → leave the page
 applyLayout();
 const bar=$('unassigned-bar');
 if(!UNASSIGNED.length){bar.innerHTML='';$('triage-body').innerHTML='';return;}
 const total=UNASSIGNED.reduce((n,g)=>n+(g.total||0),0);
 const dn=UNASSIGNED.length;
 bar.innerHTML=`<div class="unassigned-acc">
   <div class="unassigned-acc-head" onclick="openTriage()" role="button" tabindex="0">
    <span>${inboxIcon()} ${total} recording${total>1?'s':''} not assigned to a task, on ${dn} grabette${dn>1?'s':''}</span>
    <span class="unassigned-caret">Sort them ›</span>
   </div>
  </div>`;
 $('count-unassigned').textContent=total;
 renderTriageBody();}
async function cleanupOrphans(i){
 const g=ORPHANS[i];if(!g)return;
 if(!confirm(`Permanently delete ${g.episode_ids.length} orphaned episode(s) of “${g.task}”? Their pair was already removed, so they can't complete a dataset.`))return;
 const r=await fetch('/api/fleet/orphans/cleanup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode_ids:g.episode_ids})});
 if(!r.ok){alert('Failed to clean up orphaned episodes');return;}
 await refresh();}
// ===== SLAM check =====================================================
// Between two takes: run the last few episodes through the SLAM Space and show,
// per episode, whether tracking held. Nothing is pushed unless an episode is
// flagged — and then the whole tested set goes up so it can be opened in the
// LeRobot visualizer. Mechanically it is a dataset job, so it is polled and
// cancelled through the same /api/fleet/lerobot-dataset endpoints.
const SLAM_MAX_N=5;  // keep in sync with SLAM_CHECK_MAX_N server-side
let slamN=3;         // keep in sync with SLAM_CHECK_DEFAULT_N
let slamJob=null;let slamJobId='';let slamPollTimer=null;let slamAdopted=false;
// One fold, on the section itself, closed by default: between takes the panel is
// about recording, and this is a tool you reach for. Opened by the check you
// start (and by adopting one already running), so an answer never lands out of
// sight; while it IS closed, the header carries a state badge.
let slamPanelOpen=false;
function toggleSlamPanel(){slamPanelOpen=!slamPanelOpen;renderSessionList();}
function setSlamN(v){slamN=Math.max(1,Math.min(SLAM_MAX_N,+v||1));renderSessionList();}
function slamRunning(){return !!(slamJob&&['uploading','raw_ready','processing'].includes(slamJob.status));}
function deviceOnCheck(id){const d=DEVICES.find(x=>x.device_id===id);return !!(d&&d.dataset_check);}
const VERDICT_CLS={GOOD:'v-good',WARN:'v-warn',BAD:'v-bad',FAIL:'v-bad',ERROR:'v-bad'};
function verdictPill(v){
 // Empty = the run hasn't judged this take yet (it streams). Neither a verdict
 // nor a warning, so it gets its own quiet pill instead of borrowing one.
 if(!v)return '<span class="v-pill v-pending" title="not judged yet">…</span>';
 v=String(v).toUpperCase();
 return `<span class="v-pill ${VERDICT_CLS[v]||'v-warn'}">${esc(v)}</span>`;}
function num(v,suffix){return (v===null||v===undefined)?'—':esc(String(v))+(suffix||'');}
// The report table. Empty until the Space names its first verdict, so a run in
// progress shows the status line alone rather than an empty grid.
function slamTableHtml(rows){
 if(!rows||!rows.length)return '';
 const body=rows.map(q=>`<tr>`+
   `<td class="sq-ep" title="${esc(q.episode)}">${esc(q.episode)}</td>`+
   `<td class="sq-v">${verdictPill(q.verdict)}</td>`+
   `<td class="sq-num">${num(q.tracking_pct,'%')}</td>`+
   `<td class="sq-num">${q.n_lost===null||q.n_lost===undefined?'—':esc(String(q.n_lost))+(q.n_frames?'/'+esc(String(q.n_frames)):'')}</td>`+
   `<td class="sq-num">${num(q.n_jumps)}</td></tr>`).join('');
 return `<div class="sq-wrap"><table class="sq-table">`+
  `<colgroup><col class="c-ep"><col class="c-v"><col class="c-n"><col class="c-n"><col class="c-n"></colgroup>`+
  `<thead><tr>`+
  `<th class="sq-ep">Episode</th><th class="sq-v">Verdict</th>`+
  `<th class="sq-num" title="Frames the SLAM tracked">Tracked</th>`+
  `<th class="sq-num" title="Frames flagged is_lost">Lost</th>`+
  `<th class="sq-num" title="Relocalization jumps">Jumps</th></tr></thead><tbody>${body}</tbody></table></div>`;}
function slamJobHtml(){
 if(!slamJob)return '';
 const st=slamJob.status;let cls='',head=esc(slamJob.message||st||'');
 if(st==='error'){cls='err';head='✗ '+head;}
 else if(st==='cancelled'){cls='warn';head='⊘ '+head;}
 // done: the colour is the VERDICT, not the run. flagged null (no report) stays
 // neutral — "we could not tell" must not look like "all clear".
 else if(st==='done'){const f=slamJob.flagged;
  if(f&&f.length){cls='warn';head='⚠ '+head;}else if(f){cls='ok';head='✓ '+head;}}
 else head='… '+head;
 const prog=slamRunning()?`<div class="ds-progress indet"><div class="ds-progress-fill"></div></div>`:'';
 const viz=slamJob.visualizer_url?`<a href="${slamJob.visualizer_url}" target="_blank" rel="noopener">Open in the LeRobot visualizer</a>`:'';
 const ds=slamJob.result_url?`<a href="${slamJob.result_url}" target="_blank" rel="noopener">dataset</a>`:'';
 const link=(viz||ds)?`<div class="slam-link">${[viz,ds].filter(Boolean).join(' · ')}</div>`:'';
 return `<div class="slam-job ${cls}">${head}</div>`+prog
  +slamTableHtml(slamJob.quality)+link;}
// What the CLOSED header says: enough to know whether to open it.
function slamBadgeHtml(){
 if(slamRunning())return '<span class="slam-badge run">running…</span>';
 if(!slamJob||slamJob.status!=='done')return '';
 const f=slamJob.flagged;
 if(f&&f.length)return `<span class="slam-badge warn">${f.length} flagged</span>`;
 return f?'<span class="slam-badge ok">clean</span>':'';}
function slamCheckHtml(s){
 const done=Math.max(0,(s.episode_count||0)-(s.recording?1:0));
 // Grey out counts this session can't cover: asking for four when three exist read
 // as accepted and then quietly checked three. `done` is the session's finished
 // takes, so it's an upper bound — a take a device no longer holds complete is
 // dropped server-side, and the "Uploading N episode(s)" line then reports what
 // was actually planned rather than what was asked for.
 const avail=Math.min(SLAM_MAX_N,done);
 const pick=Math.min(slamN,Math.max(1,avail));  // never leave the active pill disabled
 let opts='';
 for(let i=1;i<=SLAM_MAX_N;i++)
  opts+=`<button class="slam-n${i===pick?' on':''}" ${(slamRunning()||i>avail)?'disabled':''}`
   +(i>avail?` title="only ${done} episode(s) recorded in this session so far"`:'')
   +` onclick="setSlamN(${i})">${i}</button>`;
 // Blocked for exactly the reasons the server refuses: mid-take, nothing recorded
 // yet, or a device already tied up by dataset work (the upload IS that work).
 const busy=Object.values(s.members).map(m=>m.device_id).some(deviceBusyForRec);
 const blocked=slamRunning()||s.recording||s.stopping||!done||busy;
 const why=(s.recording||s.stopping)?'Stop the current episode first'
  :(!done?'Record an episode first'
  :(busy?'A device is busy with dataset work':'Run the last episodes through SLAM'));
 const head=`<div class="sp-section-label slam-head" onclick="toggleSlamPanel()">`
  +`<span class="slam-caret">${slamPanelOpen?'▾':'▸'}</span>Trajectory Check`
  +`${slamPanelOpen?'':slamBadgeHtml()}</div>`;
 if(!slamPanelOpen)return `<div class="sp-slam">${head}</div>`;
 return `<div class="sp-slam">${head}
   <div class="slam-row">
    <span class="slam-lbl">Last</span>
    ${opts}
    <span class="slam-lbl">episode(s)</span>
    <button class="slam-btn" ${blocked?'disabled':''} title="${why}" onclick='startSlamCheck("${s.id}",${pick})'>Run Check</button>
    ${slamRunning()?`<button class="slam-cancel" onclick="cancelSlamCheck()">Cancel</button>`:''}
   </div>
   <div class="slam-hint muted">Runs the last episodes through SLAM and reports whether the trajectory held. Recording pauses while it runs.</div>
   ${slamJobHtml()}
  </div>`;}
async function startSlamCheck(sessionId,count){
 const r=await fetch('/api/fleet/slam-check',{method:'POST',headers:{'Content-Type':'application/json'},
  // The count the pills SHOW as picked (slamN clamped to what the session has), so
  // the request can never ask for more than the panel offers.
  body:JSON.stringify({session_id:sessionId,count:count||slamN})});
 const j=await r.json().catch(()=>({}));
 if(!r.ok){alert(errText(j)||'Could not start the trajectory check');return;}
 slamJobId=j.job_id;
 slamPanelOpen=true;  // a new check's answer must not land inside a closed fold
 slamJob={status:'uploading',check:true,quality:[],flagged:null,
  message:`Uploading ${(j.episodes||[]).length} episode(s)… (this can take several minutes)`};
 renderSessionList();pollSlamCheck(j.job_id);}
function pollSlamCheck(id){
 if(slamPollTimer)clearTimeout(slamPollTimer);
 const step=async()=>{
  try{
   const r=await fetch('/api/fleet/lerobot-dataset/'+id);
   if(r.ok){slamJob=await r.json();renderSessionList();
    if(['done','error','cancelled'].includes(slamJob.status))return;}
  }catch(e){}
  slamPollTimer=setTimeout(step,2500);};
 step();}
async function cancelSlamCheck(){
 if(!slamJobId)return;
 if(!confirm('Cancel the trajectory check on every device involved?'))return;
 const r=await fetch('/api/fleet/lerobot-dataset/'+slamJobId+'/cancel',{method:'POST'});
 if(!r.ok){const j=await r.json().catch(()=>({}));alert(errText(j)||'Failed to cancel the check');return;}
 pollSlamCheck(slamJobId);await refresh();}
// A check may already be running: this page was reloaded, or another operator (or
// the other browser tab) started it. Without this the panel would show an idle
// button while the devices are busy uploading for a check nobody can see.
async function adoptLiveSlamCheck(){
 try{const r=await fetch('/api/fleet/lerobot-datasets');if(!r.ok)return;
  const live=((await r.json()).jobs||[]).find(j=>j.check);
  if(!live||live.id===slamJobId)return;
  slamJobId=live.id;slamJob=live;slamPanelOpen=true;pollSlamCheck(live.id);}catch(e){}}
function recordingBuffersHtml(s){
 const labels={depth:'Depth',oak_left:'OAK left video',oak_right:'OAK right video',wrist:'Wrist video'};
 const cards=Object.entries(s.members).map(([role,m])=>{
  const d=DEVICES.find(d=>d.device_id===m.device_id);
  const report=d?.recording_buffers||{};
  const stoppedHere=(s.episodes||[]).some(ep=>ep.episode_id===report.auto_stop_episode_id&&ep.roles?.[role]===m.device_id);
  const stopWarning=stoppedHere&&report.auto_stop_reason?`<div class="sp-fault-warn">${esc(report.auto_stop_reason)} · ${esc(report.auto_stop_episode_id)}</div>`:'';
  const belongs=(s.episodes||[]).some(ep=>ep.episode_id===report.episode_id&&ep.roles?.[role]===m.device_id);
  const streams=belongs?Object.entries(report.buffers||{}):[];
  const title=`<b>${esc(m.name)} (${esc(role)})</b>${d?.online?'':' · Offline, last reported values'}`;
  if(!streams.length)return `<div>${title}${stopWarning}<p class="muted">No buffer report for this session yet.</p></div>`;
  const rows=streams.map(([name,b])=>`<tr><td>${esc(labels[name]||name)}</td><td>${b.peak_percent.toFixed(1)}%</td><td>${(b.peak_bytes/1048576).toFixed(1)} / ${(b.capacity_bytes/1048576).toFixed(0)} MiB</td></tr>`).join('');
  const warnings=streams.filter(([,b])=>!b.complete||b.rejected_frames||b.write_errors).map(([name,b])=>
   `<div class="sp-fault-warn">${esc(labels[name]||name)}: ${b.rejected_frames} rejected frames, ${b.write_errors} write errors. ${esc(b.error||'')}</div>`).join('');
  return `<div>${title}${stopWarning}<p class="muted">Last saved report: ${esc(report.episode_id)}</p><table><thead><tr><th>Stream</th><th>Peak usage</th><th>Used / capacity</th></tr></thead><tbody>${rows}</tbody></table>${warnings}</div>`;
 }).join('');
 return `<div class="session-panel"><div class="sp-section-label">Recording buffers</div>${cards}<p class="muted">Updated after saving. Measures RAM write queues; low usage does not rule out camera-side frame loss.</p></div>`;
}
function renderSessionList(){
 // A single running session REPLACES the task list: when one is open, show the
 // session card and hide the mode tabs + Tasks card (the Fleet accordion in the
 // sticky bar stays); otherwise restore them and hide the (empty) session card.
 // Who is visible is decided in applyLayout, which arbitrates between this and the
 // triage page — the other view that takes the whole page over.
 const open=SESSIONS.filter(s=>s.status==='open');
 const el=$('session-list');
 applyLayout();
 if(!open.length)return;
 el.className='';
 el.innerHTML=open.map(s=>{
  const devs=Object.entries(s.members).map(([role,m])=>{
   const url=`http://${encodeURIComponent(m.name)}.local:8000`;
   return `<span class="sp-dev"><span class="pill hand" title="${role}">${role[0].toUpperCase()}</span> ${esc(m.name)}${deviceOnline(m.device_id)?batteryPill(deviceBattery(m.device_id)):''}${deviceBusyForRec(m.device_id)?activityBadge(m.device_id):''}`+
    ` <a class="dash-icon" target="_blank" rel="noopener" title="Open ${esc(m.name)} dashboard" href="${url}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></a></span>`;
  }).join('');
  const p=sessionPhase(s);
  const ep=(s.episodes&&s.episodes.length)?s.episodes[s.episodes.length-1]:null;
  const t0=ep&&ep.start_at_utc?Date.parse(ep.start_at_utc):0;
  // Count only FINISHED episodes: the in-progress one (recording) isn't counted yet.
  const done=Math.max(0,(s.episode_count||0)-(s.recording?1:0));
  const M={
   idle:{lbl:'Idle',cls:'idle',btn:'idle',dis:'',title:'Start episode'},
   initializing:{lbl:'Initializing…',cls:'init',btn:'busy',dis:'',title:'Cancel'},
   recording:{lbl:'Recording',cls:'rec',btn:'recording',dis:'',title:'Stop episode'},
   stopping:{lbl:'Stopping…',cls:'stopping',btn:'busy',dis:'disabled',title:'Stopping…'},
  }[p];
  const timer=(p==='recording'&&t0)?`<span class="sp-timer rec-dur" data-start="${t0}">${fmtDur(Date.now()-t0)}</span>`:'';
  const task=esc(s.task_name||'(task deleted)');
  // A member busy with a dataset upload/conversion blocks STARTING a new episode
  // (mirrors the server gate); an in-progress recording is left alone.
  const busyMembers=Object.values(s.members).map(m=>m.device_id).filter(deviceBusyForRec);
  // A member in a hardware fault refuses the start on the device itself, and the
  // server refuses to dispatch one at all — so grey the button here too rather
  // than letting the operator press it and collect a 409. A fault outranks busy:
  // busy clears on its own, a fault does not.
  const faultMembers=Object.values(s.members).map(m=>m.device_id).filter(deviceFault);
  const faultBlocked=faultMembers.length>0;
  const recBlocked=(p==='idle')&&(busyMembers.length>0||faultBlocked);
  const circDis=M.dis||(recBlocked?'disabled':'');
  const onCheck=busyMembers.some(deviceOnCheck);
  const busyWhat=onCheck?'running the trajectory check':'processing a dataset';
  // A fault outranks busy in the title too: busy clears on its own, a fault does not.
  const circTitle=faultBlocked?'A device cannot record — see the fault below'
   :recBlocked?`A device is busy ${busyWhat} — wait for it to finish`:M.title;
  const circle=`<div class="sp-rec"><button class="rec-toggle ${M.btn}${recBlocked?' blocked':''}" ${circDis} title="${circTitle}" onclick='episodeToggle("${s.id}")'><span class="rt-inner"></span></button></div>`;
  const busyNote=(recBlocked&&!faultBlocked)?`<div class="sp-busy-note">${busyMembers.map(deviceName).join(', ')} busy ${busyWhat} — recording paused until it finishes.</div>`:'';
  // Two different fault reports, and they answer different questions:
  //  • faultNote — "this device cannot record", known BEFORE you press;
  //  • startErrNote — "this device did not record the take you just started",
  //    which the fleet only ever learns from the command result, and which used
  //    to be buried in the device's command history where nobody looks.
  const faultWarn=faultBlocked
   ?`<div class="sp-fault-warn">${warnIcon()} ${faultMembers.map(d=>`<b>${esc(deviceName(d))}</b>: ${esc(deviceFault(d))}`).join('<br>')}</div>`:'';
  const startErrs=s.start_errors||[];
  const startErrNote=startErrs.length
   ?`<div class="sp-fault-warn">${warnIcon()} ${startErrs.map(m=>`<b>${esc(m.name)}</b> did not start: ${esc(m.error)}`).join('<br>')}<br>This take is missing ${startErrs.length>1?'these arms':'that arm'} — stop it and redo it.</div>`:'';
  // A stop the fleet dispatched but the device never acknowledged: it may still be
  // recording, so say so and offer to re-send rather than showing a clean Idle.
  const unconf=s.stop_unconfirmed||[];
  const unconfNote=unconf.length?`<div class="sp-stop-warn">${warnIcon()} ${unconf.map(m=>esc(m.name)).join(', ')} never confirmed the stop — ${unconf.length>1?'they may':'it may'} still be recording. <button class="restop-btn" onclick='episodeStop("${s.id}")'>Re-send stop</button></div>`:'';
  const del=`<button class="del-ep" ${(p!=='idle'||!done)?'disabled':''} title="Delete the last episode on all its devices" onclick='deleteLastEpisode("${s.id}")'>Delete last episode</button>`;
  const close=`<button class="close-btn" onclick='closeSession("${s.id}")'>Close session</button>`;
  const epWord=`episode${done>1?'s':''}`;
  return `
   <div class="session-panel var-c">
    <div class="sp-rec-zone">
     <div class="sp-head"><span class="sp-task">${task}</span>${phasePill(s)}</div>
     <div class="sp-timer-wrap">${timer}</div>
     ${circle}
     ${busyNote}
     ${faultWarn}
     ${startErrNote}
     ${unconfNote}
     <div class="sp-rec-del">${soundSetting()}${del}</div>
    </div>
    <div class="sp-manage-zone">
     <div class="sp-section-label">Episodes</div>
     <div class="ep-big"><span class="ep-big-num">${done}</span><span class="ep-big-lbl">${epWord} recorded this session</span></div>
     <div class="sp-meta"><span class="sp-label">Devices</span><span class="sp-devs">${devs}</span></div>
     ${recordingBuffersHtml(s)}
     ${slamCheckHtml(s)}
     ${close}
    </div>
   </div>`;
 }).join('');
 tickRecDur();}
const openRaw=new Set();
function rawToggle(id,open){open?openRaw.add(id):openRaw.delete(id);}
const showState=new Set();
function runState(id){showState.add(id);dispatch(id,'get_state',{});}
function hideState(id){showState.delete(id);refresh();}
function kindOf(d){
 const s=`${d.name||''} ${d.device_id||''} ${(d.capabilities||[]).join(' ')}`.toLowerCase();
 if(s.includes('gripette'))return 'gripette';
 if(s.includes('casquette'))return 'casquette';
 return 'grabette';}
function btn(id,type,args,online){const dis=online?'':'disabled';return `<button class="primary" ${dis} onclick='dispatch("${id}","${type}",${JSON.stringify(args||{})})'>${type}</button>`;}
async function dispatch(device_id,type,args){
 await fetch('/api/fleet/dispatch',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({device_id,type,args})});refresh();}
async function removeDevice(device_id){
 await fetch(`/api/fleet/devices/${device_id}/remove`,{method:'POST'});refresh();}
async function checkLogin(){
 // Never leave #who stuck on "Checking…": if the request fails (the free-tier
 // Space is waking up, a redeploy is in flight, or a transient network error),
 // show a retrying message and let the 3s tick recover instead of throwing.
 try{
  const r=await fetch('/api/fleet/me');
  if(!r.ok)throw new Error('HTTP '+r.status);
  const m=await r.json();loggedIn=m.logged_in;
  if(loggedIn){$('who').innerHTML=`<div class="who-row"><span>Signed in as <b>${m.username}</b></span><a class="btn logout" href="/oauth/huggingface/logout">Logout</a></div>`;}
  else{$('who').innerHTML=`<a class="btn primary" href="/oauth/huggingface/login">Sign in with HuggingFace</a>`;}
 }catch(e){
  loggedIn=false;
  $('who').innerHTML=`<span class="muted">Connecting to the Space… <span style="opacity:.7">(it may be waking up — retrying)</span></span>`;
 }
 $('gated').disabled=!loggedIn;}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function cap(s){return String(s).replace(/(^|_)([a-z])/g,(m,p,c)=>p+c.toUpperCase());}
function kv(k,v){return `<div class="k">${esc(k)}</div><div class="v">${v}</div>`;}
function fmtState(st){
 const rows=[];
 rows.push(kv('Daemon',`${esc(st.state||'?')}${st.backend?` · <span class="muted">${esc(st.backend)}</span>`:''}`));
 const sens=st.sensor||{};const cap=sens.capture||{};
 if(cap.is_capturing){
  const bits=[];
  if(cap.duration_seconds!=null)bits.push(`${Math.round(cap.duration_seconds)}s`);
  if(cap.frame_count!=null)bits.push(`${cap.frame_count} frames`);
  rows.push(kv('Capture',`<span class="pill rec">● Recording</span> ${esc(bits.join(' · '))}`));
  if(cap.session_id)rows.push(kv('Session',esc(cap.session_id)));
 }else if(cap.is_starting){
  rows.push(kv('Capture',`<span class="pill idle">○ Starting…</span>`));
 }else{
  rows.push(kv('Capture',`<span class="pill idle">○ Idle</span>`));
 }
 if(sens.angle){const deg=r=>(Number(r)*180/Math.PI).toFixed(1);
  rows.push(kv('Angle',`prox ${deg(sens.angle.proximal)}° · dist ${deg(sens.angle.distal)}°`));}
 if(st.error)rows.push(kv('Error',`<span class="err">${esc(st.error)}</span>`));
 return `<div class="kv">${rows.join('')}</div>`;}
function formatResult(type,res){
 if(res==null)return '<span class="muted">no result</span>';
 if(res.status==='error')return `<div class="err">⚠ ${esc(res.message||'error')}</div>`;
 if(type==='get_state'&&res.state)return fmtState(res.state);
 if(type==='start_capture')return `<div>▶ Capture started${res.episode_id?` · <span class="muted">${esc(res.episode_id)}</span>`:''}</div>`;
 if(type==='stop_capture'){const r=res.result||{};const bits=[];
  if(r.frame_count!=null)bits.push(`${r.frame_count} frames`);
  if(r.duration_seconds!=null)bits.push(`${Math.round(r.duration_seconds)}s`);
  return `<div>■ Capture stopped${bits.length?` · <span class="muted">${esc(bits.join(' · '))}</span>`:''}</div>`;}
 if(type==='logout')return '<div>Logged out</div>';
 return `<pre>${esc(JSON.stringify(res,null,1))}</pre>`;}
// The /24 an IPv4 sits in — the fallback "which network" answer for a device that
// doesn't report its WiFi SSID (wired, hotspot, or an older device build). Coarse,
// but it still tells two devices on different LANs apart.
function subnetOf(ip){const m=/^(\\d+)\\.(\\d+)\\.(\\d+)\\.\\d+$/.exec(ip||'');return m?`${m[1]}.${m[2]}.${m[3]}.0/24`:'';}
// The line under a device's name. Label every value: unlabelled, the id and the IP
// read as two anonymous strings of digits.
function deviceMetaHtml(d){
 const bits=[`<span class="idlab" title="Device serial — stable id persisted on the device">Serial</span> ${esc(d.device_id)}`];
 if(d.ip)bits.push(`<span class="idlab" title="Device address on its local network">IP</span> ${esc(d.ip)}`);
 const net=d.network||subnetOf(d.ip);
 if(net)bits.push(`<span class="idlab" title="${d.network?'WiFi network the device reported':'Subnet derived from the IP — the device reported no WiFi network (wired, hotspot, or older device build)'}">Network</span> ${esc(net)}`);
 if(d.pending)bits.push(`${d.pending} pending`);
 return `<span class="muted">${bits.join(' · ')}</span>`;}
function rowHtml(d){
 const last=d.history[0];
 const caps=d.capabilities||[];
 const hasLogout=caps.includes('logout');
 const logoutDis=d.online?'':'disabled';
 const logoutBtn=hasLogout
  ?`<button class="logout-icon" title="Logout" ${logoutDis} onclick='dispatch("${d.device_id}","logout",{})'><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg></button>`
  :'';
 const deleteBtn=`<button class="del-icon" title="Remove device" onclick='removeDevice("${d.device_id}")'><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>`;
 const dashUrl=`http://${encodeURIComponent(d.name)}.local:8000`;
 const dashBtn=`<a class="dash-icon" target="_blank" rel="noopener" title="Open device dashboard (${dashUrl})" href="${dashUrl}"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></a>`;
 const handBadge=d.hand?`<span class="pill hand" title="${esc(d.hand)} hand">${d.hand==='left'?'L':d.hand==='right'?'R':esc(d.hand)}</span>`:'';
 // Working for a fleet-started dataset build → offer to cancel it from here, for
 // when the build page was left or reloaded. Cancels the build on ALL its devices.
 const dsWhat=d.dataset_check?'trajectory check':'dataset build';
 const dsCancelBtn=d.dataset_job
  ?`<button class="ds-cancel-dev" title="Cancel this ${dsWhat} on every device involved" onclick='cancelDatasetForDevice("${d.device_id}")'>⊘ Cancel ${d.dataset_check?'check':'build'}</button>`
  :'';
 const showRes=last&&!(last.type==='get_state'&&!showState.has(d.device_id));
 const rtime=last&&last.ts?new Date(last.ts*1000).toLocaleTimeString():'';
 const closeBtn=last&&last.type==='get_state'?`<button class="rclose" title="Close" onclick='hideState("${d.device_id}")'>×</button>`:'';
 const resultHtml=showRes
  ?`<div class="result"><div class="rlabel">${esc(cap(last.type))}${rtime?` · ${esc(rtime)}`:''}${closeBtn}</div>${formatResult(last.type,last.result)}`
   +`<details class="raw-d"${openRaw.has(d.device_id)?' open':''} ontoggle='rawToggle("${d.device_id}",this.open)'><summary>raw</summary><pre>${esc(JSON.stringify(last.result,null,1))}</pre></details></div>`
  :'<div class="result empty muted">No command run yet.</div>';
 return `<tr>
   <td class="c-dot"><span class="dot ${d.online?'on':'off'}" title="${d.online?'online':'offline'}"></span></td>
   <td class="name"><b>${d.name}</b>${handBadge}${d.online?batteryPill(d.battery):''}${d.online?activityBadge(d.device_id):''}${d.online?faultBadge(d.device_id):''}${dsCancelBtn}${faultNote(d)}<br>${deviceMetaHtml(d)}${resultHtml}</td>
   <td class="c-tools"><div class="tools">${dashBtn}${logoutBtn}${deleteBtn}</div></td></tr>`;}
// Chime every 10 recorded episodes (1 beep at 10, 2 at 20…) so a session can be
// counted by ear. Synthesised in-page: no asset to load, none to ship.
let EPISODE_CHIME_EVERY=10;  // `let`: overridable from the console to test
let soundOn=true;try{soundOn=localStorage.getItem('grabette.sound')!=='off';}catch(e){}
let audioCtx=null;
function audio(){
 if(!audioCtx){const AC=window.AudioContext||window.webkitAudioContext;
  if(!AC)return null;try{audioCtx=new AC();}catch(e){return null;}}
 if(audioCtx.state==='suspended')audioCtx.resume().catch(()=>{});
 return audioCtx;}
addEventListener('pointerdown',audio,{once:true});  // browsers gate audio on a gesture
addEventListener('keydown',audio,{once:true});
// The gain ramps aren't decoration: a raw start/stop on an oscillator clicks.
function note(ctx,at,freq,dur){
 const t=ctx.currentTime+at,osc=ctx.createOscillator(),g=ctx.createGain();
 osc.frequency.value=freq;
 g.gain.setValueAtTime(0,t);
 g.gain.linearRampToValueAtTime(.15,t+.015);
 g.gain.setValueAtTime(.15,t+dur-.05);
 g.gain.linearRampToValueAtTime(0,t+dur);
 osc.connect(g).connect(ctx.destination);
 osc.start(t);osc.stop(t+dur+.02);}
// Off a round 50, the plain 600Hz blip alone says "another ten" — 1 to 4 of them,
// never mixed with the bigger cues (60 is one blip, 270 is two). Landing exactly on
// a multiple of 50 is what earns the milestone sounds: a rising couplet per 100 then
// a long low note for a leftover 50, so 150 is "hundred, fifty" and 200 is
// "hundred, hundred". The three differ in shape, not just pitch, which is what
// keeps them apart at a glance.
function chime(tens){
 if(!soundOn)return;
 const ctx=audio();if(!ctx)return;
 const rest=tens%5;
 if(rest){for(let i=0;i<rest;i++)note(ctx,i*.22,600,.11);return;}
 let at=0;
 for(let i=Math.floor(tens/10);i>0;i--){note(ctx,at,500,.1);note(ctx,at+.12,750,.14);at+=.38;}
 if(tens%10>=5)note(ctx,at,350,.3);}
function recordingCue(kind){
 if(!soundOn)return;
 const ctx=audio();if(!ctx)return;
 const tones=kind==='start'?[500,800]:kind==='stop'?[800,450]:[250,250,250];
 tones.forEach((freq,i)=>note(ctx,i*.22,freq,.16));
}
const recordingSounds=new Map();
function checkRecordingSounds(){
 for(const s of SESSIONS.filter(s=>s.status==='open')){
  const latest=(s.episodes||[]).at(-1)?.episode_id;
  const reports=Object.values(s.members).map(m=>DEVICES.find(d=>d.device_id===m.device_id))
   .filter(d=>d?.online).map(d=>d.recording_buffers||{});
  let recording=reports.some(r=>r.capture_episode_id===latest&&r.is_capturing&&!r.is_stopping);
  const alarm=reports.some(r=>r.auto_stop_episode_id===latest&&r.auto_stop_reason)?latest:null;
  const previous=recordingSounds.get(s.id);
  // Losing a heartbeat is not confirmation that recording stopped.
  if(previous?.recording&&!recording&&reports.length<Object.keys(s.members).length)recording=true;
  // Seed silently on reload. Never replay old alarms or repeat a cue on polling.
  if(previous){
   if(alarm&&alarm!==previous.alarm)recordingCue('full');
   else if(recording&&!previous.recording)recordingCue('start');
   else if(!recording&&previous.recording)recordingCue('stop');
  }
  recordingSounds.set(s.id,{recording,alarm});
 }
 for(const id of recordingSounds.keys())if(!SESSIONS.some(s=>s.id===id&&s.status==='open'))recordingSounds.delete(id);
}
// A labelled On/Off switch in the recording zone's bottom row, facing the delete
// button. The knob sits on the state you ARE in — unlike a mute button, there is
// nothing to read backwards. The tooltip says what the sound IS, not what the
// click does, so it stays true in both positions.
function bellIcon(){return '<svg class="bell-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>';}
function soundSetting(){
 return `<span class="sp-audio" title="Rising tones: started. Falling tones: stopped. Three low tones: buffer nearly full, automatic stop. Also chimes every ${EPISODE_CHIME_EVERY} episodes.">`
  +`<span class="sp-audio-lbl">${bellIcon()}Audio signal</span>`
  +`<button class="sw" type="button" role="switch" aria-label="Audio signal"`
  +` aria-checked="${soundOn?'true':'false'}" onclick="toggleSound(this)">`
  +`<span class="sw-knob"></span>`
  +`<span class="sw-opt sw-opt-on">On</span><span class="sw-opt sw-opt-off">Off</span>`
  +`</button></span>`;}
function toggleSound(el){
 soundOn=!soundOn;
 // Move the knob here rather than waiting for the panel's 1s re-render, which
 // would make the switch feel stuck (and would drop focus mid-click).
 if(el)el.setAttribute('aria-checked',soundOn?'true':'false');
 try{localStorage.setItem('grabette.sound',soundOn?'on':'off');}catch(e){}
 if(soundOn)chime(1);}
// Finished episodes only: the in-progress one isn't counted (matches the card).
function epTens(s){return Math.floor(Math.max(0,(s.episode_count||0)-(s.recording?1:0))/EPISODE_CHIME_EVERY);}
// session id -> highest tens chimed. Seeded on first sight so a reload mid-session
// doesn't replay, and monotonic because two pollers write SESSIONS: a stale read
// must never re-chime. Pruning costs one size check unless a session really went.
const epChimed=new Map();
function checkEpisodeChime(){
 for(const s of SESSIONS){
  if(s.status!=='open')continue;
  const tens=epTens(s),seen=epChimed.get(s.id);
  if(seen===undefined){epChimed.set(s.id,tens);continue;}
  if(tens>seen){epChimed.set(s.id,tens);if(tens)chime(tens);}}
 if(epChimed.size>SESSIONS.length)
  for(const id of epChimed.keys())if(!SESSIONS.some(s=>s.id===id))epChimed.delete(id);}
async function refresh(){
 if(!loggedIn)return;
 const [dr,tr,sr,or_,ur]=await Promise.all([fetch('/api/fleet/devices'),fetch('/api/fleet/tasks'),fetch('/api/fleet/sessions'),fetch('/api/fleet/orphans'),fetch('/api/fleet/unassigned')]);
 if(dr.ok){DEVICES=(await dr.json()).devices;}
 if(tr.ok){TASKS=(await tr.json()).tasks;}
 if(sr.ok){SESSIONS=(await sr.json()).sessions;}
 // Once per page, and only with a session open: adopt a check already running.
 if(!slamAdopted&&SESSIONS.some(s=>s.status==='open')){slamAdopted=true;adoptLiveSlamCheck();}
 if(or_.ok){const o=await or_.json();ORPHANS=o.groups;SPLITS=o.split||[];}
 if(ur.ok){UNASSIGNED=(await ur.json()).devices;}
 // Fleet tables (Devices must be applied before the launcher/roles render).
 const kinds={grabette:[],gripette:[],casquette:[]};
 for(const d of DEVICES)kinds[kindOf(d)].push(d);
 for(const k of KINDS){
  const list=kinds[k];
  $('tb-'+k).innerHTML=list.map(rowHtml).join('');
  $('count-'+k).textContent=list.length;
  $('empty-'+k).style.display=list.length?'none':'block';}
 renderFleetRecap();
 // Tasks (with inline launcher) + the running session.
 renderTaskList();
 renderSessionList();
 renderOrphans();
 renderUnassigned();
 renderManage();}
async function tick(){
 try{await checkLogin();}catch(e){}
 try{await refresh();}catch(e){}
}
tick();setInterval(tick,3000);
// 1s UI tick. While a session is open, poll session state at 1s (not the 3s
// full refresh) so phase changes show promptly: initializing→recording (at T0)
// and, crucially, stopping→idle the moment the fleet marks the devices' stop
// complete — instead of lagging up to 3s. Devices/tasks stay on the 3s refresh.
async function uiTick(){
 if(SESSIONS.some(s=>s.status==='open')){
  try{const r=await fetch('/api/fleet/sessions');if(r.ok)SESSIONS=(await r.json()).sessions;}catch(e){}
  renderSessionList();
 }
 checkEpisodeChime();  // 1s, not the 3s refresh, so the chime lands on the stop
 checkRecordingSounds();
 tickRecDur();
}
setInterval(uiTick,1000);
</script></body></html>"""
