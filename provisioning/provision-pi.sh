#!/usr/bin/env bash
# Runs ON the dimensioner Pi. Installs Caddy as a systemd service, wires up Route53 DNS-01 with
# short-lived credentials fetched from the WMS backend, and reverse-proxies HTTPS -> the local
# capture API on :8090.
#
# No AWS credentials file to prepare anymore -- the backend mints short-lived (1hr), narrowly
# scoped Route53 credentials for this device's own FQDN (see registration.py's
# fetch_route53_credentials() / dimensioner-dns.service.ts's mintTemporaryRoute53Credentials) and
# hands them over the same authenticated channel used for self-registration/heartbeat. Requires
# set-warehouse-identity.sh to have already run (so .env has warehouse/zone identity + this
# device's own token) -- run order relative to that script matters now, unlike the old flow.
#
# DNS A-record creation is separately automatic: the backend upserts it on self-registration (see
# set-warehouse-identity.sh / registration.py). Caddy's DNS-01 cert challenge writes its own TXT
# record independently of that A record.
#
# Usage (on the Pi): sudo ./provision-pi.sh
set -euo pipefail

DIMENSIONER_HOME="/home/unie/dimensioner"
CADDY_BIN="/usr/local/bin/caddy"
CADDY_ETC="/etc/caddy"

if [ ! -f "${DIMENSIONER_HOME}/.env" ]; then
  echo "No .env found -- run set-warehouse-identity.sh first (needs warehouse/zone identity and this device's own token)." >&2
  exit 1
fi
# set -a/+a so the sourced vars are actually exported to child processes (registration.py below
# reads them via os.environ) -- a plain `source` alone leaves them as shell-local variables only,
# confirmed live: registration.py --route53-credentials silently saw them as unset without this.
set -a
# shellcheck disable=SC1091
source "${DIMENSIONER_HOME}/.env"
set +a
if [ -z "${DIMENSIONER_WAREHOUSE_CODE:-}" ] || [ -z "${DIMENSIONER_ZONE_CODE:-}" ]; then
  echo "DIMENSIONER_WAREHOUSE_CODE/DIMENSIONER_ZONE_CODE not set in .env -- run set-warehouse-identity.sh first." >&2
  exit 1
fi

DNS_DOMAIN="${DIMENSIONER_DNS_DOMAIN:-uniewms.com}"
sanitize() { echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9'; }
FQDN="dimensioner-$(sanitize "${DIMENSIONER_WAREHOUSE_CODE}")-$(sanitize "${DIMENSIONER_ZONE_CODE}").${DNS_DOMAIN}"

if [ ! -x "${DIMENSIONER_HOME}/caddy" ] && [ ! -f "/home/unie/caddy" ]; then
  echo "Expected the custom Caddy binary (with the route53 DNS plugin) at /home/unie/caddy." >&2
  echo "Download it from https://caddyserver.com/api/download?os=linux&arch=arm64&p=github.com%2Fcaddy-dns%2Froute53" >&2
  exit 1
fi

install -m 0755 /home/unie/caddy "${CADDY_BIN}"
mkdir -p "${CADDY_ETC}"

sed "s/{{DIMENSIONER_HOSTNAME}}/${FQDN}/" "$(dirname "$0")/Caddyfile.template" > "${CADDY_ETC}/Caddyfile"

echo "Fetching short-lived Route53 credentials for ${FQDN}..."
CREDS_JSON="$(cd "${DIMENSIONER_HOME}" && /home/unie/micromamba/bin/micromamba run -n ros2 python registration.py --route53-credentials)"
if echo "${CREDS_JSON}" | grep -q '"error"'; then
  echo "Failed to fetch Route53 credentials: ${CREDS_JSON}" >&2
  exit 1
fi

ACCESS_KEY_ID="$(echo "${CREDS_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["accessKeyId"])')"
SECRET_ACCESS_KEY="$(echo "${CREDS_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["secretAccessKey"])')"
SESSION_TOKEN="$(echo "${CREDS_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sessionToken"])')"
EXPIRATION="$(echo "${CREDS_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["expiration"])')"

{
  echo "AWS_ACCESS_KEY_ID=${ACCESS_KEY_ID}"
  echo "AWS_SECRET_ACCESS_KEY=${SECRET_ACCESS_KEY}"
  echo "AWS_SESSION_TOKEN=${SESSION_TOKEN}"
} > "${CADDY_ETC}/caddy.env"
chmod 0600 "${CADDY_ETC}/caddy.env"
chown root:root "${CADDY_ETC}/caddy.env"

echo "Credentials expire at ${EXPIRATION} -- only needed for the initial DNS-01 challenge and"
echo "future cert renewals (~every 60-90 days), not continuously. If a renewal fails after this"
echo "token has expired, just re-run this script to refresh caddy.env (see dimensioner/README.md's"
echo "Auto-update section for the known gap here -- no automatic refresh exists yet)."

cat > /etc/systemd/system/caddy.service <<'EOF'
[Unit]
Description=Caddy (HTTPS reverse proxy for the dimensioner capture API)
# systemd-time-wait-sync.service: a Pi 4 has no battery-backed RTC, so after being powered off
# for shipping (built/tested on one network, deployed at a different warehouse later) its clock
# starts wrong on every boot until NTP catches up -- and this is the ACME/DNS-01 certificate
# issuance step, which is exactly the piece most sensitive to a bad clock. Confirmed live this
# was previously ungated (this unit only waited on network.target).
After=network.target systemd-time-wait-sync.service
Wants=systemd-time-wait-sync.service

[Service]
Type=simple
User=root
EnvironmentFile=/etc/caddy/caddy.env
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable caddy.service
systemctl restart caddy.service

echo "Caddy installed and started for ${FQDN}. Check: systemctl status caddy.service"
echo "Cert issuance can take up to a minute (Route53 DNS-01 propagation) -- watch: journalctl -u caddy -f"
