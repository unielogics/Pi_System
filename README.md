# Dimensioner Pi fleet deployment runbook

A dimensioner device is a Raspberry Pi 4 running a depth camera (Deptrum Aurora 930), this
directory's Python app (a camera-agnostic FastAPI capture API), and Caddy (HTTPS reverse proxy).
It self-registers with the WMS backend, which auto-creates its public DNS hostname and keeps that
hostname's DNS record in sync with the Pi's LAN IP forever after (see "How the network side
works," below).

This document covers two things: building a reusable base image once, and the exact steps to turn
that image into a working device at a specific warehouse. No new tooling is required — every
script referenced here already exists in this directory.

## Section 1 — Build the golden image

Do this once, and again only when the base software genuinely needs to change (a dependency bump,
a new camera model, etc). Most new deployments should skip straight to Section 2 using an
already-built image.

1. Flash a fresh Raspberry Pi OS (64-bit, Bookworm) onto an SD card with [Raspberry Pi
   Imager](https://www.raspberrypi.com/software/), using its **OS customization** screen (the gear
   icon shown before writing) to set:
   - Hostname: anything generic (e.g. `dimensioner-golden`) — never read by the app.
   - Username/password: use `franco`. Every systemd unit and script in this directory hardcodes
     `/home/franco/...` paths and `User=franco` — deviating here means hand-editing every unit
     file per device, which defeats the point of a reusable image.
   - Enable SSH.
   - **Do not** set WiFi here. Each physical site's WiFi is set at *flash* time per device
     (Section 2, step 4), not baked into the golden image — the golden image should work at any
     site.
2. Boot it once over Ethernet (not WiFi), so setup isn't tied to any one site's network.
3. Install micromamba, then create a `ros2` conda/mamba environment via
   [RoboStack](https://robostack.github.io/) (provides `rclpy`, `cv_bridge`, `sensor_msgs`,
   `tf2_msgs` — these are NOT pip packages). Inside that environment, `pip install -r
   requirements.txt` (this directory's file) for the remaining Python dependencies
   (`fastapi`, `uvicorn`, `numpy`, `pillow`).
4. Build the vendor's `deptrum-ros-driver-aurora930` ROS2 package into `~/dimensioner_ws`,
   following Deptrum's own SDK build instructions for the Aurora 930. **This step is currently
   vendor-specific tribal knowledge with no written instructions anywhere** — whoever builds the
   next golden image should capture the exact commands they run here and fold them into this
   section, closing that gap permanently.
5. Copy this repo's `dimensioner/` directory to `/home/franco/dimensioner`.
6. Install the systemd unit files and enable the always-on services, including the network
   watchdog (which needs no warehouse identity, unlike the heartbeat timer — that one's still
   deferred to Section 2, since it depends on identity that doesn't exist yet):
   ```bash
   sudo install -m 0644 dimensioner-api.service dimensioner-ros.service dimensioner-network-watchdog.service dimensioner-network-watchdog.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable dimensioner-api.service dimensioner-ros.service dimensioner-network-watchdog.timer
   ```
6a. Install the sudoers rule both watchdogs depend on (see "Self-healing," below, for what each
    one restarts/reconnects) — without this, both watchdogs' recovery actions silently fail:
   ```bash
   sudo install -m 0440 provisioning/franco-dimensioner-sudoers /etc/sudoers.d/franco-dimensioner
   ```
7. Download the custom Caddy binary (with the Route53 DNS-01 plugin) and place it at
   `/home/franco/caddy`:
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
2. Go to Warehouse Settings → **Sensors and Cameras**. If the "Provisioning secret" bar near the
   top reads "Not generated yet," click **Generate** — once per warehouse, not per device; every
   device at that warehouse reuses the same value. Click **Show**, then **Copy**.
3. **Do not** use the "Add device" / "+ Add New Sensor or Camera" modal for a self-registering
   Pi — that form is the legacy manual endpoint-URL/token-entry path this process replaces
   entirely. A self-registered device appears in the table automatically; nothing needs to be
   pre-created there.

**On the physical Pi:**

4. Flash the golden `.img` with Raspberry Pi Imager, using the OS-customization screen **again**
   at flash time to set this specific site's WiFi SSID/password — the one thing that legitimately
   varies per device and should never be baked into the golden image.
5. Boot it at the target site; confirm it joined the site's WiFi (`ip addr`, or check the router's
   DHCP client list).
6. SSH in and run, in order:
   ```bash
   cd /home/franco/dimensioner/provisioning
   sudo ./set-warehouse-identity.sh <warehouse-code-lowercase> <zone-code-lowercase> <secret-from-dashboard>
   ```
   This writes `.env`, restarts the API, installs and enables the heartbeat timer, and fires one
   immediate self-registration call.
7. Check the dashboard's Sensors and Cameras device table — the new device should appear, tagged
   "Self-registered," within a few seconds.
8. Get a temporary credentials file onto the Pi with the shared Route53 IAM user's keys:
   ```
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   ```
   then run:
   ```bash
   sudo ./provision-pi.sh dimensioner-<warehouse-code>-<zone-code>.uniewms.com /path/to/that/creds/file
   ```
   (`provision-pi.sh` deletes the creds file itself after use.) This installs and starts Caddy
   with HTTPS via Route53 DNS-01, at the exact hostname the backend already created in step 6.
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
  gateway/internet). After 2 consecutive failed checks, forces a WiFi reconnect via `nmcli`.

Both call `sudo -n ...` as `franco`, which requires the `franco-dimensioner-sudoers` rule
installed in Section 1, step 6a — a fresh golden image built without that step will have both
watchdogs detect the problem but silently fail to act on it (check `journalctl` for
`sudo -n: a password is required` if a device's cameras/network never seem to self-recover).

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
