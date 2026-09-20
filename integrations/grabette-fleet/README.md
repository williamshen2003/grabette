# Local Grabette fleet

Standalone fleet dashboard with per-device recording-buffer telemetry on the
session page. Based on [pollen-robotics/grabette-fleet](https://huggingface.co/spaces/pollen-robotics/grabette-fleet/tree/65d84500844fef4f3fd3591e5389cfa5be3e0c93),
with the session telemetry and local HTTP launcher added in this fork.

## Run on your Mac

From the repository root:

```bash
cd integrations/grabette-fleet
uv run --no-project --with-requirements requirements.txt uvicorn local_app:app --host 0.0.0.0 --port 7860 --no-proxy-headers --timeout-graceful-shutdown 5
```

The Mac must be logged into Hugging Face with the account that owns the devices.
If needed, run this before starting:

```bash
uv run --no-project --with-requirements requirements.txt hf auth login
```

Open [localhost:7860](http://localhost:7860/) and click **Sign in**. Local mode
uses the Mac's logged-in Hugging Face identity. Operator and login routes are
restricted to loopback; only the authenticated `/api/devices/` endpoints are
reachable from other computers. Keep `--no-proxy-headers` enabled and run
directly, without a reverse proxy.

## Connect the Grabettes

Both devices need this fork's device heartbeat support (commit `db08457` or
later). Set `GRABETTE_RELAY_URL=http://<MAC_LAN_IP>:7860` in their service
environment and restart `grabette` while idle. The current deployment uses
`/etc/systemd/system/grabette.service.d/local-fleet.conf` on each Pi:

```ini
[Service]
Environment="GRABETTE_RELAY_URL=http://<MAC_LAN_IP>:7860"
```

After editing that file, run `sudo systemctl daemon-reload` and
`sudo systemctl restart grabette`. Use the Mac's current LAN IP, not `localhost`,
on the Pis. Keep the Mac awake and on the same network while using this fleet.
To return to the hosted fleet, remove this override and reload/restart again.

Launch a session to see **Recording buffers** under its device list. Reports
update after each recording finishes saving and show peak usage, capacity,
rejected frames, and write errors. Only reports from that session are displayed.
No report is shown before the first completed recording. Low usage does not
rule out camera-side frame loss. Latest reports reset when a device restarts;
episode metadata retains the recorded measurements.

With **Sound** on, rising tones indicate recording started, falling tones
indicate recording stopped, and three low tones warn of a buffer auto-stop.
Click the page once to enable browser audio. Cues follow device heartbeats,
so they may lag capture by a few seconds. Reloading does not replay old cues.
Each device automatically stops at 95% queue usage even if the browser is closed;
the session panel displays the reason. This headroom reduces overflow risk but
does not guarantee zero frame loss if the writer or system stalls suddenly.

Fleet sessions and groups are in memory and do not transfer from the hosted
fleet or survive a local server restart. Tasks and recordings remain on the Pis.
This local dashboard does not update the hosted Hugging Face Space.

## Tests

```bash
uv run --no-project --with-requirements requirements.txt --with pytest python -m pytest tests -q
```

The rendering check also requires Node.js. OAuth initialization uses the Mac's
Hugging Face login. No device recording is started by these tests.
