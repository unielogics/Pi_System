#!/usr/bin/env bash
# Runs ON the dimensioner Pi. Installs Caddy as a systemd service, wires up Route53 DNS-01 with
# the shared IAM credentials, and reverse-proxies HTTPS -> the local capture API on :8090.
#
# DNS is now automatic: the backend upserts this device's A record itself on self-registration
# (see set-warehouse-identity.sh / registration.py), so there's no DNS precondition here anymore.
# Caddy's DNS-01 cert challenge writes its own TXT record independently of that A record, so run
# order between this script and set-warehouse-identity.sh doesn't matter.
#
# Usage (on the Pi): sudo ./provision-pi.sh <fqdn> <env-file-with-aws-creds>
#   e.g. sudo ./provision-pi.sh dimensioner-wh007-ed1.uniewms.com /home/franco/caddy.env.tmp
# The env file must define AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (one per line). Passing
# credentials via a file rather than argv avoids leaking the secret into shell history / `ps`.
set -euo pipefail

FQDN="${1:?Usage: provision-pi.sh <fqdn> <env-file-with-aws-creds>}"
CREDS_FILE="${2:?Usage: provision-pi.sh <fqdn> <env-file-with-aws-creds>}"

if [ ! -f "${CREDS_FILE}" ]; then
  echo "Credentials file not found: ${CREDS_FILE}" >&2
  exit 1
fi

CADDY_BIN="/usr/local/bin/caddy"
CADDY_ETC="/etc/caddy"
DIMENSIONER_HOME="/home/franco/dimensioner"

if [ ! -x "${DIMENSIONER_HOME}/caddy" ] && [ ! -f "/home/franco/caddy" ]; then
  echo "Expected the custom Caddy binary (with the route53 DNS plugin) at /home/franco/caddy." >&2
  echo "Download it from https://caddyserver.com/api/download?os=linux&arch=arm64&p=github.com%2Fcaddy-dns%2Froute53" >&2
  exit 1
fi

install -m 0755 /home/franco/caddy "${CADDY_BIN}"
mkdir -p "${CADDY_ETC}"

sed "s/{{DIMENSIONER_HOSTNAME}}/${FQDN}/" "$(dirname "$0")/Caddyfile.template" > "${CADDY_ETC}/Caddyfile"

install -m 0600 "${CREDS_FILE}" "${CADDY_ETC}/caddy.env"
rm -f "${CREDS_FILE}"

cat > /etc/systemd/system/caddy.service <<'EOF'
[Unit]
Description=Caddy (HTTPS reverse proxy for the dimensioner capture API)
After=network.target

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
