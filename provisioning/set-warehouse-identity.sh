#!/usr/bin/env bash
# Runs ON the dimensioner Pi. Bakes in this device's warehouse+zone identity and provisioning
# secret, then triggers a one-time self-registration (which also installs the heartbeat timer)
# so the device shows up in the WMS dashboard and gets its own Route53 A record automatically --
# no more manually running create-dns-record.sh (retired) or typing endpointUrl/token into the
# dashboard by hand.
#
# Usage: sudo ./set-warehouse-identity.sh <warehouse-code> <zone-code> <provisioning-secret>
#   e.g. sudo ./set-warehouse-identity.sh wh007 ed1 dps_<secret from the dashboard>
#
# Get the provisioning secret from the WMS dashboard: Settings > Sensors and Cameras, near the
# zone/warehouse pickers -- generate one if the warehouse doesn't have one yet.
#
# If this device_config.json was cloned from another Pi's SD card, delete
# data/device_config.json BEFORE running this script so a fresh device_id gets generated --
# otherwise two physical devices will fight over the same backend row.
set -euo pipefail

WAREHOUSE_CODE="${1:?Usage: set-warehouse-identity.sh <warehouse-code> <zone-code> <provisioning-secret>}"
ZONE_CODE="${2:?Usage: set-warehouse-identity.sh <warehouse-code> <zone-code> <provisioning-secret>}"
PROVISIONING_SECRET="${3:?Usage: set-warehouse-identity.sh <warehouse-code> <zone-code> <provisioning-secret>}"

DIMENSIONER_HOME="/home/unie/dimensioner"
ENV_FILE="${DIMENSIONER_HOME}/.env"

touch "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

# Strip any previous identity lines, then append the current ones -- idempotent re-run if a
# device ever needs to be re-pointed at a different warehouse/zone.
sed -i '/^DIMENSIONER_WAREHOUSE_CODE=/d;/^DIMENSIONER_ZONE_CODE=/d;/^DIMENSIONER_PROVISIONING_SECRET=/d' "${ENV_FILE}"
{
  echo "DIMENSIONER_WAREHOUSE_CODE=${WAREHOUSE_CODE}"
  echo "DIMENSIONER_ZONE_CODE=${ZONE_CODE}"
  echo "DIMENSIONER_PROVISIONING_SECRET=${PROVISIONING_SECRET}"
} >> "${ENV_FILE}"

echo "Wrote warehouse identity to ${ENV_FILE}."

# One-time restart so the running API process picks up the new .env -- not part of the
# heartbeat loop, which never restarts the service.
systemctl restart dimensioner-api.service

install -m 0644 "$(dirname "$0")/../dimensioner-heartbeat.service" /etc/systemd/system/dimensioner-heartbeat.service
install -m 0644 "$(dirname "$0")/../dimensioner-heartbeat.timer" /etc/systemd/system/dimensioner-heartbeat.timer
systemctl daemon-reload
systemctl enable --now dimensioner-heartbeat.timer

echo "Registering with the WMS backend now..."
cd "${DIMENSIONER_HOME}"
/home/unie/micromamba/bin/micromamba run -n ros2 python registration.py

echo "Done. Heartbeat timer installed (systemctl status dimensioner-heartbeat.timer)."
echo "Check the WMS dashboard's Sensors and Cameras page -- this device should now appear tagged \"Self-registered\"."
