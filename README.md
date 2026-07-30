# Dimensioner Pi fleet deployment runbook

A dimensioner device is a Raspberry Pi 4 running a depth camera (Deptrum Aurora 930), this
directory's Python app (a camera-agnostic FastAPI capture API), and Caddy (HTTPS reverse proxy).
It self-registers with the WMS backend, which auto-creates its public DNS hostname and keeps that
hostname's DNS record in sync with the Pi's LAN IP forever after (see "How the network side
works," below).

This document covers two things: building a reusable base image once, and the exact steps to turn
that image into a working device at a specific warehouse. No new tooling is required — every
script referenced here already exists in this directory.

## Fastest path: one command

For a fresh Pi (SD card flashed with Raspberry Pi Imager, username `unie`, SSH enabled, on the
target warehouse's network -- see step 1 below), `provisioning/setup-golden-image.sh` collapses
everything else in Sections 1-2 into a single command:
```bash
./setup-golden-image.sh <warehouse-code> <zone-code> <provisioning-secret>
# e.g. ./setup-golden-image.sh wh-007 ed3 dps_<secret from the dashboard's Add Device flow>
```
Run as `unie` (not via `sudo` — the script escalates internally only where it actually needs to).
Safe to re-run if it fails partway (every step is idempotent). Reads the rest of this document
for exactly what it's doing under the hood, in case a step needs manual troubleshooting.

## Section 1 — Build the golden image

Do this once, and again only when the base software genuinely needs to change (a dependency bump,
a new camera model, etc). Most new deployments should skip straight to Section 2 using an
already-built image.

1. Flash a fresh Raspberry Pi OS (64-bit, Bookworm) onto an SD card with [Raspberry Pi
   Imager](https://www.raspberrypi.com/software/), using its **OS customization** screen (the gear
   icon shown before writing) to set:
   - Hostname: anything generic (e.g. `dimensioner-golden`) — never read by the app.
   - Username/password: use `unie`. Every systemd unit and script in this directory hardcodes
     `/home/unie/...` paths and `User=unie` — deviating here means hand-editing every unit
     file per device, which defeats the point of a reusable image.
   - Enable SSH.
   - **Do not** set WiFi here. Each physical site's WiFi is set at *flash* time per device
     (Section 2, step 4), not baked into the golden image — the golden image should work at any
     site.
2. Boot it once over Ethernet (not WiFi), so setup isn't tied to any one site's network.
3. Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
   (installs to `~/micromamba`), then create the `ros2` environment from
   [RoboStack](https://robostack.github.io/)'s `robostack-humble` channel (provides `rclpy`,
   `cv_bridge`, `sensor_msgs`, `tf2_msgs`, `image_transport` — these are NOT pip packages):
   ```bash
   micromamba create -n ros2 -c robostack-humble -c conda-forge \
     python=3.12 \
     ros-humble-ros-base ros-humble-cv-bridge ros-humble-image-transport ros-humble-angles \
     colcon-common-extensions \
     fastapi uvicorn python-multipart pillow
   ```
   `fastapi`/`uvicorn`/`pillow` are pulled from conda-forge here rather than via a separate `pip
   install -r requirements.txt` step — confirmed both approaches produce a working env, but doing
   it in one `micromamba create` avoids a second dependency-resolution pass. `numpy` comes in
   transitively via the ROS packages; `requirements.txt` in this directory documents the pip-only
   equivalent for reference. **Known gotcha**: if you ever add a bare `boost=X` or an unrelated
   pin to this environment, `ros-humble-cv-bridge` and `boost`/`python=3.12` can become mutually
   unsatisfiable (confirmed hit on the original golden Pi — `micromamba` reports a wall of
   `Could not solve for environment specs` boost/python version cross-talk). Fix: drop the extra
   pin and let `robostack-humble`/`conda-forge` resolve boost's version themselves; don't pin
   `boost` directly.
4. Build the vendor's `deptrum-ros-driver-aurora930` ROS2 package into `~/dimensioner_ws`. The
   vendor SDK (`deptrum-stream-aurora900-linux-aarch64`, a proprietary precompiled `.so` + headers,
   ~40MB) and the ROS2 driver source are Deptrum-licensed — obtained directly from Deptrum
   (support contact/portal, not a public URL), never committed to this public repo. Once you have
   both from Deptrum:
   ```bash
   mkdir -p ~/dimensioner_ws/src
   cd ~/dimensioner_ws/src
   # Extract/clone Deptrum's deptrum-ros-driver package here, then:
   sed -i 's/deptrum-ros-driver\b/deptrum-ros-driver-aurora930/g' deptrum-ros-driver-aurora930/package.xml
   # Place the vendor SDK (deptrum-stream-aurora900-linux-aarch64-vX.X.X-...) under
   # deptrum-ros-driver-aurora930/ext/ -- the package's CMakeLists.txt expects it there.
   cd ~/dimensioner_ws
   micromamba run -n ros2 colcon build --cmake-args -DSTREAM_SDK_TYPE=AURORA930 -DCMAKE_POLICY_VERSION_MINIMUM=3.5
   ```
   `STREAM_SDK_TYPE=AURORA930` is the load-bearing flag — the same source tree also builds
   STELLAR400/STELLAR420/NEBULA variants depending on this value; get it wrong and the driver
   builds against the wrong camera's SDK. `CMAKE_POLICY_VERSION_MINIMUM=3.5` is required on
   today's `robostack-humble`/`conda-forge` toolchain (CMake 4.x) — the vendor's `CMakeLists.txt`
   declares `cmake_minimum_required(VERSION 3.5)`, which CMake 4 refuses to configure at all
   without this override (confirmed live on a fresh build; the original golden Pi predates this
   and was built against an older CMake that didn't need it). Also run the vendor SDK's udev
   setup once per golden image (grants non-root USB access to the camera — `idVendor=3251`):
   ```bash
   cd ~/dimensioner_ws/src/deptrum-ros-driver-aurora930/ext/deptrum-stream-aurora900-*/scripts
   sudo cp 99-deptrum-libusb.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```
5. Clone this repo to `/home/unie/dimensioner` (public repo, no credentials needed):
   ```bash
   git clone https://github.com/unielogics/Pi_System.git /home/unie/dimensioner
   ```
6. Install the systemd unit files and enable the always-on services, including the network
   watchdog and the auto-updater (neither needs warehouse identity, unlike the heartbeat timer —
   that one's still deferred to Section 2, since it depends on identity that doesn't exist yet):
   ```bash
   sudo install -m 0644 dimensioner-api.service dimensioner-ros.service dimensioner-network-watchdog.service dimensioner-network-watchdog.timer dimensioner-auto-update.service dimensioner-auto-update.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable dimensioner-api.service dimensioner-ros.service dimensioner-network-watchdog.timer dimensioner-auto-update.timer
   ```
6a. Install the sudoers rule both watchdogs depend on (see "Self-healing," below, for what each
    one restarts/reconnects) — without this, both watchdogs' recovery actions silently fail:
   ```bash
   sudo install -m 0440 provisioning/unie-dimensioner-sudoers /etc/sudoers.d/unie-dimensioner
   ```
7. Download the custom Caddy binary (with the Route53 DNS-01 plugin) and place it at
   `/home/unie/caddy`:
   ```
   https://caddyserver.com/api/download?os=linux&arch=arm64&p=github.com%2Fcaddy-dns%2Froute53
   ```
   `provisioning/provision-pi.sh` expects it at exactly that path.
8. Plug the camera into a **USB 3.0 port** (the Pi 4's blue ports) if at all possible.
   `driver_launcher.py` hardcodes `ir_fps:=5 rgb_fps:=5` and `point_cloud_enable:=false` only
   because the original reference Pi's camera sits on a bandwidth-starved USB 2.0 hub port — at
   full bandwidth on that port the camera disconnects after ~10s. If this golden image's hardware
   properly uses USB 3.0, lift those caps in `driver_launcher.py` and verify with
   `ros2 topic hz /aurora/depth/image_raw` for a few minutes before shipping the image — every
   cloned unit inherits whichever caps are present at clone time.
9. **Before shrinking/exporting the image**, delete any device-identity state so every clone
   generates its own fresh identity on first boot instead of colliding with this golden Pi's:
   ```bash
   rm -f dimensioner/data/device_config.json dimensioner/data/registration_state.json dimensioner/.env
   ```
10. Shut down, remove the SD card, and read it back out to a `.img`/`.img.gz` file (Raspberry Pi
    Imager's "Use custom" reader, or `rpi-clone`/`dd`). Store it durably (e.g. S3) with a
    version/date in the filename.

## Section 2 — Deploy one new Pi to a warehouse

Repeat this for every physical device, at every warehouse.

**On the WMS dashboard, before touching hardware:**

1. Confirm the target Warehouse and Zone already exist (Warehouses → Setup). Self-registration
   404s until both do.
2. Go to Warehouse Settings → **Sensors and Cameras** → **+ Add New Sensor or Camera** →
   **Self-registering Pi (recommended)**. Pick the target zone; if the provisioning secret reads
   "Not generated yet," click **Generate** — once per warehouse, not per device; every device at
   that warehouse reuses the same value. The exact `set-warehouse-identity.sh` command (with the
   warehouse/zone codes and secret already filled in) is shown right below it — copy it, you'll
   paste it into the Pi's SSH session in step 6.
3. The plain "Advanced: enter a network scanner's endpoint URL and token manually" link at the
   bottom of that same chooser is the legacy manual endpoint-URL/token-entry path — don't use it
   for a self-registering Pi. A self-registered device appears in the table automatically; nothing
   needs to be pre-created there.

**On the physical Pi:**

4. Flash the golden `.img` with Raspberry Pi Imager, using the OS-customization screen **again**
   at flash time to set this specific site's WiFi SSID/password — the one thing that legitimately
   varies per device and should never be baked into the golden image.
5. Boot it at the target site; confirm it joined the site's WiFi (`ip addr`, or check the router's
   DHCP client list).
6. SSH in and run the command copied from step 2 (paste it as-is):
   ```bash
   cd /home/unie/dimensioner/provisioning
   sudo ./set-warehouse-identity.sh <warehouse-code-lowercase> <zone-code-lowercase> <secret-from-dashboard>
   ```
   This writes `.env`, restarts the API, installs and enables the heartbeat timer, and fires one
   immediate self-registration call.
7. Check the dashboard's Sensors and Cameras device table — the new device should appear, tagged
   "Self-registered," within a few seconds.
8. Run, with no arguments:
   ```bash
   sudo ./provision-pi.sh
   ```
   No AWS credentials file to prepare — this device fetches its own short-lived (1hr), narrowly
   scoped Route53 credentials from the backend (see "Auto-update" below for the one known gap:
   there's no automatic refresh yet, so a cert renewal after that hour expires needs a re-run of
   this script). This installs and starts Caddy with HTTPS via Route53 DNS-01, at the exact
   hostname the backend already created in step 6 (derived from `.env`, not passed as an
   argument).
9. Wait up to ~60s for cert issuance (`journalctl -u caddy -f` to watch), then verify — **from a
   device physically on this warehouse's network** — by opening:
   ```
   https://dimensioner-<warehouse-code>-<zone-code>.uniewms.com/viewer?token=<the device's own token, shown in the dashboard>
   ```
   This will never load from off-site; see the explanation below before assuming something's
   broken.
10. Physically mount/aim the camera, then use the dashboard's Sensors and Cameras → Calibrate tab
    to run initial height/live calibration.

## How the network side works (read this before debugging a "can't be reached" report)

This device's public hostname (`dimensioner-<wh>-<zone>.uniewms.com`) is real and globally
resolvable, but it always resolves to the Pi's private LAN IP (e.g. `192.168.1.x`), which is only
reachable from a device physically connected to this warehouse's own network. There is no
tunnel or relay — once DNS resolves, the browser connects straight to the Pi. This is why the
link will always fail from outside the building by design, and why a *specific device's* stale
DNS cache — not the domain, not the router — is the first thing to suspect if one browser/device
can't load it while others on the same network can. The backend's heartbeat (every 5 minutes,
`dimensioner-heartbeat.timer`) keeps the DNS record in sync automatically whenever the Pi's LAN IP
changes (e.g. after a DHCP lease renewal); no manual DNS action is ever needed.

## Self-healing

Two independent watchdogs run on every device, each recovering from a different failure mode
without human intervention:

- **Camera/ROS-driver hang** — `api.py`'s in-process `_watchdog_loop`, started automatically when
  `dimensioner-api.service` boots. Polls the camera adapter's live frame age every 2 seconds; if
  frames go stale for 12+ seconds (the driver process is still running per `systemctl`, but has
  silently stopped publishing — the observed real-world failure mode on this hardware, not a
  crash), it restarts `dimensioner-ros.service` via `sudo -n systemctl restart`, with a 30-second
  cooldown between restart attempts.
- **Network loss** — `network_watchdog.py`, run every 2 minutes by
  `dimensioner-network-watchdog.timer`. Pings the default gateway (a real reachability probe, not
  `registration.py`'s `_lan_ip()`, which only detects a fully-missing route, not a dead
  gateway/internet). After 2 consecutive failed checks, forces a reconnect on whichever interface
  (wired or WiFi) carries the default route, via `nmcli`.

Both call `sudo -n ...` as `unie`, which requires the `unie-dimensioner-sudoers` rule
installed in Section 1, step 6a — a fresh golden image built without that step will have both
watchdogs detect the problem but silently fail to act on it (check `journalctl` for
`sudo -n: a password is required` if a device's cameras/network never seem to self-recover).

## Auto-update

`auto_update.py`, run once nightly (03:00 local, ±30min jitter across the fleet) by
`dimensioner-auto-update.timer`, keeps every device on the latest commit of this repo with zero
manual SSH sessions:

1. `git fetch origin` + compares local `HEAD` against `origin/main`. No new commits → exits
   immediately, no-op.
2. New commit(s) → `git pull --ff-only` (never a merge/rebase). If the device's local tree has
   ever been hand-edited during a debugging session, this fails loudly instead of silently
   discarding or merging over that in-progress edit — leave a failed pull for a human to resolve.
3. If `requirements.txt` changed, reinstalls dependencies inside the `ros2` env. If any
   `.service`/`.timer` file changed, reinstalls it into `/etc/systemd/system/` and
   `daemon-reload`s.
4. Restarts `dimensioner-api.service`/`dimensioner-ros.service` unconditionally after any
   successful pull, and logs the before/after SHA + a one-line change summary to `journalctl`.

`unielogics/Pi_System` is a **public** GitHub repo, so no credentials are needed anywhere in this
loop — if it's ever made private, this whole mechanism needs a read-only deploy key added first.
There is currently no rollback mechanism (a bad commit on `main` reaches every device on its next
tick) — an accepted gap at the current fleet size, revisit if the fleet grows.

**On-demand updates**: `POST /update-now` (bearer-auth, same as every other endpoint) fires the
same `dimensioner-auto-update.service` unit immediately instead of waiting for the nightly run —
fire-and-forget, since a successful update restarts `dimensioner-api.service` itself partway
through (running `auto_update.py` in-process would kill the request handler mid-response). A
60-second cooldown returns `{"status": "already_triggered", ...}` on a redundant call instead of
silently no-oping. The dashboard's Sensors and Cameras device table calls this directly from the
browser (same direct-to-Pi pattern the live camera viewer uses), including a bulk "Update Now"
action across multiple selected devices at once.

Each device reports the short commit SHA it's running on with every self-registration/heartbeat
call (`registration.py`'s `_git_sha()`) — visible in the dashboard's Sensors and Cameras device
table next to the Self-registered/Manual badge, flagged "(outdated)" if it doesn't match the
latest commit on `Pi_System`'s `main` branch.

## Route53 credentials

`provision-pi.sh` no longer requires a human to manually copy an AWS Route53 IAM secret key onto
the device. Instead, `registration.py`'s `fetch_route53_credentials()` requests short-lived (1hr),
narrowly-scoped AWS STS credentials from the backend (`POST /dimensioners/route53-credentials`,
gated by the same self-registration auth as `/dimensioners/register`) and writes them into
`/etc/caddy/caddy.env` — Caddy's `acme_dns route53` plugin picks up `AWS_SESSION_TOKEN` alongside
the access/secret key automatically, no `Caddyfile.template` change needed.

**Known gap**: those credentials expire after 1 hour, but Caddy only calls Route53 again at each
cert renewal (~every 60-90 days) — there's no automatic refresh loop today. If a renewal fails
because the token's long expired, `journalctl -u caddy` will show a clear ACME/DNS-01 failure; the
fix is just re-running `sudo ./provision-pi.sh` to refresh `caddy.env`. A small
`caddy-credential-refresh.timer` that does this automatically every ~45 minutes is a reasonable
follow-up, not built yet.

Requires `ROUTE53_DIMENSIONER_ROLE_ARN` to be set on the backend (see
`UnieBackend/docs/ENV_VARIABLES_ROUTE53.md`) — a one-time AWS IAM role setup, not something either
repo's code provisions.

## What's fixed vs. what varies per device

The FastAPI app, the ROS2 driver, and Caddy config are identical across every Pi in the fleet —
none of it branches on which warehouse or camera model it's running on. Only two things vary per
physical unit:

- **Warehouse/zone identity + provisioning secret** — written to `.env` by
  `set-warehouse-identity.sh`, once per device.
- **This site's WiFi + the Pi's own OS login** — set once, at flash time, via Raspberry Pi
  Imager's OS-customization screen; never scripted, since it varies per physical location and per
  install.

Everything else (`device_id`, the auth token, the DNS record) is generated automatically — the
`device_id` from the Pi's own CPU serial number, the auth token minted by the backend on first
registration, the DNS record upserted by the backend the moment that registration call lands.

<!-- auto_update.py verified live on WH-007 ED1, 2026-07-29 -->
