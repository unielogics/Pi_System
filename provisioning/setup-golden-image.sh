#!/usr/bin/env bash
# Runs ON a bare-flashed Pi (Raspberry Pi OS/Debian, user `unie`, SSH enabled, on the target
# warehouse's network) to build a fully working dimensioner device in one command, collapsing
# README.md's Section 1 (build the golden image) + Section 2 (deploy to a warehouse) manual
# checklist into a single script. Only step this can't do: flashing the SD card itself (a
# physical action that has to happen before this script can even run over SSH) and plugging the
# camera into a USB 3.0 port (also physical).
#
# Usage (run as `unie`, NOT via sudo -- the script escalates internally, line by line, only for
# the specific systemd/sudoers/install operations that need it, same as
# set-warehouse-identity.sh/provision-pi.sh already do):
#   ./setup-golden-image.sh <warehouse-code> <zone-code> <provisioning-secret>
#   e.g. ./setup-golden-image.sh wh-007 ed3 dps_<secret from the dashboard's Add Device flow>
#
# Safe to re-run: every step below is the same idempotent operation the manual checklist already
# used (micromamba create -n ros2 no-ops if the env exists; git clone is skipped if the directory
# is already a clone; set-warehouse-identity.sh/provision-pi.sh are both already idempotent).
set -euo pipefail

WAREHOUSE_CODE="${1:?Usage: setup-golden-image.sh <warehouse-code> <zone-code> <provisioning-secret>}"
ZONE_CODE="${2:?Usage: setup-golden-image.sh <warehouse-code> <zone-code> <provisioning-secret>}"
PROVISIONING_SECRET="${3:?Usage: setup-golden-image.sh <warehouse-code> <zone-code> <provisioning-secret>}"

DIMENSIONER_HOME="/home/unie/dimensioner"
WORKSPACE="/home/unie/dimensioner_ws"
MICROMAMBA_ROOT="/home/unie/micromamba"

log() { echo ">>> $*"; }

if [ "$(whoami)" != "unie" ]; then
  echo "This script must run as the 'unie' user (every systemd unit/path in this repo hardcodes it)." >&2
  exit 1
fi
if [ "$(uname -m)" != "aarch64" ]; then
  echo "Expected a 64-bit (aarch64) OS -- got $(uname -m)." >&2
  exit 1
fi

# ── Step 1: micromamba + ros2 environment ──────────────────────────────────────────────────────
# Every systemd unit in this repo (dimensioner-api/ros/auto-update.service) and both existing
# provisioning scripts hardcode the binary at /home/unie/micromamba/bin/micromamba -- but the
# official installer places it at ~/.local/bin/micromamba and treats ~/micromamba as only the
# MAMBA_ROOT_PREFIX (environments root), NOT the binary's own location (confirmed live: installing
# fresh leaves no binary at .../micromamba/bin/micromamba at all). Symlink it into the path every
# other file already expects, rather than rewriting every unit file/script to match the installer.
if [ ! -x "${MICROMAMBA_ROOT}/bin/micromamba" ]; then
  if [ ! -x "/home/unie/.local/bin/micromamba" ]; then
    log "Installing micromamba..."
    "${SHELL}" <(curl -L micro.mamba.pm/install.sh) < /dev/null
  fi
  mkdir -p "${MICROMAMBA_ROOT}/bin"
  ln -sf /home/unie/.local/bin/micromamba "${MICROMAMBA_ROOT}/bin/micromamba"
fi
MICROMAMBA="${MICROMAMBA_ROOT}/bin/micromamba"

if ! "${MICROMAMBA}" env list 2>/dev/null | grep -q '/envs/ros2$'; then
  log "Creating the ros2 micromamba environment (this takes a while)..."
  # Deliberately no bare boost=X pin in this first pass -- adding one here has previously made
  # ros-humble-cv-bridge/boost/python=3.12 mutually unsatisfiable (see README.md Section 1 step 3's
  # documented gotcha). Let robostack-humble/conda-forge resolve boost's version themselves.
  "${MICROMAMBA}" create -y -n ros2 -c robostack-humble -c conda-forge \
    python=3.12 \
    ros-humble-ros-base ros-humble-cv-bridge ros-humble-image-transport ros-humble-angles \
    colcon-common-extensions \
    fastapi uvicorn python-multipart pillow
else
  log "ros2 environment already exists -- skipping."
fi

# ros-humble-cv-bridge pulls in Boost's runtime .so transitively, but not its headers/CMake
# config files -- the vendor driver's subscribe_node/CMakeLists.txt does its own
# find_package(Boost REQUIRED COMPONENTS system filesystem), which fails without them (confirmed
# live: "Could NOT find Boost (missing: Boost_INCLUDE_DIR system filesystem)"). Installed as a
# separate, unpinned step so it can't reopen the boost/python=3.12 resolver conflict above --
# confirmed live this resolves cleanly in seconds since it's purely additive, not a re-pin.
if [ -z "$(find "${MICROMAMBA_ROOT}/envs/ros2/lib/cmake" -maxdepth 2 -iname 'BoostConfig.cmake' 2>/dev/null)" ]; then
  log "Installing Boost headers/CMake config (libboost-devel)..."
  "${MICROMAMBA}" install -y -n ros2 -c conda-forge libboost-devel
fi

# ── Step 2: this repo ───────────────────────────────────────────────────────────────────────────
if [ ! -d "${DIMENSIONER_HOME}/.git" ]; then
  log "Cloning Pi_System..."
  git clone https://github.com/unielogics/Pi_System.git "${DIMENSIONER_HOME}"
else
  log "Pi_System already cloned -- skipping (run auto_update.py to update)."
fi
cd "${DIMENSIONER_HOME}"

# ── Step 3: systemd units + sudoers ─────────────────────────────────────────────────────────────
# Must happen BEFORE set-warehouse-identity.sh below -- that script's own first action is
# `systemctl restart dimensioner-api.service`, which fails outright if the unit was never
# installed. dimensioner-ros.service is safe to enable before the vendor driver is built --
# Restart=on-failure/RestartSec=5 just retries every 5s until step 5 below makes it succeed
# (driver_launcher.py deliberately raises a clear "workspace not found" error in the meantime,
# not a crash).
log "Installing systemd units and the sudoers rule..."
sudo install -m 0644 dimensioner-api.service dimensioner-ros.service \
  dimensioner-network-watchdog.service dimensioner-network-watchdog.timer \
  dimensioner-auto-update.service dimensioner-auto-update.timer \
  /etc/systemd/system/
sudo systemctl daemon-reload
# systemd-time-wait-sync.service ships with Raspberry Pi OS but is disabled by default -- a Pi 4
# has no battery-backed RTC, so its clock starts wrong on every boot until NTP catches up, which
# risks a bad TLS/ACME/authenticated-HTTPS outcome if dimensioner-api/caddy/heartbeat/auto-update
# race ahead of it (confirmed via a live check: this was previously disabled on the already-live
# WH-007 ED3 Pi). Every unit that depends on it declares that dependency itself (After=/Wants=);
# this just flips the unit on so those dependencies actually resolve to something running.
sudo systemctl enable systemd-time-wait-sync.service
sudo systemctl enable dimensioner-api.service dimensioner-ros.service \
  dimensioner-network-watchdog.timer dimensioner-auto-update.timer
sudo install -m 0440 provisioning/unie-dimensioner-sudoers /etc/sudoers.d/unie-dimensioner
sudo visudo -c

# ── Step 4: self-register early so later steps can authenticate as this device ─────────────────
# set-warehouse-identity.sh writes .env, restarts the API, installs the heartbeat timer, and
# fires one immediate self-registration call -- run it now (not at the very end) so steps 5-6
# below can use registration.py's own device token instead of needing the raw warehouse secret
# passed around in shell state for the whole rest of this script.
log "Registering with the WMS backend..."
sudo ./provisioning/set-warehouse-identity.sh "${WAREHOUSE_CODE}" "${ZONE_CODE}" "${PROVISIONING_SECRET}"

# ── Step 5: vendor camera driver (licensed -- fetched via the backend's presigned S3 URL) ──────
# Deliberately checks the compiled driver lib dir, NOT install/setup.bash -- colcon writes that
# workspace-level setup script even when the actual package build fails (confirmed live: a run
# that hit a CMake configure error still left install/setup.bash in place, which made this guard
# skip a real re-build on the next attempt instead of retrying).
if [ ! -d "${WORKSPACE}/install/deptrum-ros-driver-aurora930/lib" ] || [ -z "$(ls -A "${WORKSPACE}/install/deptrum-ros-driver-aurora930/lib" 2>/dev/null)" ]; then
  if [ ! -d "${WORKSPACE}/src/deptrum-ros-driver-aurora930" ]; then
    log "Fetching the vendor camera driver tarball..."
    # registration.py reads DIMENSIONER_WAREHOUSE_CODE/ZONE_CODE from os.environ -- a plain
    # `source .env` only sets shell-local variables, it does NOT export them to this child
    # process (same gap already fixed in set-warehouse-identity.sh/provision-pi.sh).
    set -a
    # shellcheck disable=SC1091
    source "${DIMENSIONER_HOME}/.env"
    set +a
    DOWNLOAD_JSON="$("${MICROMAMBA}" run -n ros2 python registration.py --vendor-driver-download-url)"
    if echo "${DOWNLOAD_JSON}" | grep -q '"error"'; then
      echo "Failed to fetch the vendor driver download URL: ${DOWNLOAD_JSON}" >&2
      exit 1
    fi
    DOWNLOAD_URL="$(echo "${DOWNLOAD_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["url"])')"

    mkdir -p "${WORKSPACE}/src"
    curl -sL -o /tmp/deptrum-driver.tar.gz "${DOWNLOAD_URL}"
    tar -xzf /tmp/deptrum-driver.tar.gz -C "${WORKSPACE}/src"
    rm -f /tmp/deptrum-driver.tar.gz
    # The extracted tarball's top-level dir is named deptrum-ros-driver-aurora930-<version> --
    # rename to the plain package name every downstream step (this script, README.md, CMake) expects.
    EXTRACTED_DIR="$(find "${WORKSPACE}/src" -maxdepth 1 -type d -name 'deptrum-ros-driver-aurora930*' | head -1)"
    if [ -n "${EXTRACTED_DIR}" ] && [ "${EXTRACTED_DIR}" != "${WORKSPACE}/src/deptrum-ros-driver-aurora930" ]; then
      mv "${EXTRACTED_DIR}" "${WORKSPACE}/src/deptrum-ros-driver-aurora930"
    fi
    sed -i 's/deptrum-ros-driver\b/deptrum-ros-driver-aurora930/g' "${WORKSPACE}/src/deptrum-ros-driver-aurora930/package.xml"
  else
    log "Vendor driver source already extracted -- skipping download."
  fi

  # Clear any stale CMakeCache from a previous failed configure attempt (e.g. missing
  # -DCMAKE_POLICY_VERSION_MINIMUM the first time) -- CMake can otherwise reuse a cached
  # configuration that never actually succeeded, silently masking a real re-build.
  rm -rf "${WORKSPACE}/build" "${WORKSPACE}/install" "${WORKSPACE}/log"

  log "Building the vendor driver (STREAM_SDK_TYPE=AURORA930)..."
  cd "${WORKSPACE}"
  # -DCMAKE_POLICY_VERSION_MINIMUM=3.5: the vendor's CMakeLists.txt files declare
  # cmake_minimum_required(VERSION 3.5), but CMake 4.x (what today's robostack-humble/conda-forge
  # channel installs, confirmed live -- the original golden Pi was built against an older CMake)
  # dropped support for policies below 3.5/3.10 entirely and refuses to configure at all without
  # this override. Confirmed via the exact fix CMake's own error message suggests.
  "${MICROMAMBA}" run -n ros2 colcon build --cmake-args -DSTREAM_SDK_TYPE=AURORA930 -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  cd "${DIMENSIONER_HOME}"

  if [ ! -d "${WORKSPACE}/install/deptrum-ros-driver-aurora930/lib" ] || [ -z "$(ls -A "${WORKSPACE}/install/deptrum-ros-driver-aurora930/lib" 2>/dev/null)" ]; then
    echo "colcon build did not produce the expected driver library -- check ${WORKSPACE}/log/latest_build/ for the real error." >&2
    exit 1
  fi

  log "Installing camera udev rules (non-root USB access)..."
  UDEV_SCRIPTS_DIR="$(find "${WORKSPACE}/src/deptrum-ros-driver-aurora930/ext" -maxdepth 1 -type d -name 'deptrum-stream-aurora900-*')/scripts"
  sudo cp "${UDEV_SCRIPTS_DIR}/99-deptrum-libusb.rules" /etc/udev/rules.d/
  sudo udevadm control --reload-rules
  sudo udevadm trigger
else
  log "Vendor driver already built -- skipping."
fi

# ── Step 6: Caddy binary ────────────────────────────────────────────────────────────────────────
if [ ! -f /home/unie/caddy ]; then
  log "Downloading the Caddy binary (with the Route53 DNS-01 plugin)..."
  curl -sL -o /home/unie/caddy \
    "https://caddyserver.com/api/download?os=linux&arch=arm64&p=github.com%2Fcaddy-dns%2Froute53"
  chmod +x /home/unie/caddy
else
  log "Caddy binary already present -- skipping."
fi

# ── Step 7: restart the capture API/ROS driver now that the real driver is built ──────────────
sudo systemctl restart dimensioner-ros.service dimensioner-api.service
sleep 5

# ── Step 8: HTTPS via Caddy + Route53 DNS-01 (fetches its own short-lived creds automatically) ─
log "Provisioning HTTPS (Caddy + Route53 DNS-01)..."
sudo ./provisioning/provision-pi.sh

log "Done. Check the dashboard's Sensors and Cameras page -- this device should appear tagged"
log "\"Self-registered\" within a few seconds. Physically mount/aim the camera (USB 3.0 blue port"
log "if at all possible), then use Sensors and Cameras -> Calibrate to run initial calibration."
