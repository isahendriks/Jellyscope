#!/usr/bin/env bash
# Sends a one-off test message to the same Slack Incoming Webhook leak_alert.py
# uses (Monitor/webhook_secrets.py, gitignored -- see webhook_secrets_example.py).
# Reads the URL from there instead of hardcoding it here, so there's only one
# place the real secret ever lives.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_FILE="${SCRIPT_DIR}/../Monitor/webhook_secrets.py"

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing ${SECRETS_FILE} -- copy Monitor/webhook_secrets_example.py to" \
       "Monitor/webhook_secrets.py and fill in SLACK_WEBHOOK_URL first." >&2
  exit 1
fi

WEBHOOK_URL="$(sed -nE 's/^SLACK_WEBHOOK_URL[[:space:]]*=[[:space:]]*"(.*)"[[:space:]]*$/\1/p' "${SECRETS_FILE}")"

if [[ -z "${WEBHOOK_URL}" ]]; then
  echo "SLACK_WEBHOOK_URL is empty in ${SECRETS_FILE} -- see webhook_secrets_example.py" \
       "for how to create one." >&2
  exit 1
fi

MESSAGE="${1:-Test message from $(hostname) at $(date '+%F %T %Z')}"

http_status="$(curl -sS -o /tmp/slack_test_response.txt -w "%{http_code}" \
  -X POST -H "Content-type: application/json" \
  --data "{\"text\": $(printf '%s' "${MESSAGE}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}" \
  "${WEBHOOK_URL}")"

if [[ "${http_status}" == "200" ]]; then
  echo "Sent: ${MESSAGE}"
else
  echo "Slack webhook returned HTTP ${http_status}:" >&2
  cat /tmp/slack_test_response.txt >&2
  exit 1
fi
