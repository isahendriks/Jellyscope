"""Template for webhook_secrets.py (gitignored, never committed). Copy this file to
webhook_secrets.py in the same directory and fill in real values.

    cp webhook_secrets_example.py webhook_secrets.py

NOT named secrets.py -- Monitor/ sits on sys.path for every pipeline script (see
config.py), and a local module named exactly `secrets.py` shadows the stdlib
`secrets` module for the whole process. numpy.random's bit_generator imports
`from secrets import randbits` internally for entropy, and gets this file instead
-- which crashed analyse.py (and therefore its live-stream on port 8080) with
"ImportError: cannot import name randbits" the moment secrets.py existed here.
"""

# Slack Incoming Webhook URL for leak_alert.py -- create one at
# https://api.slack.com/apps > (your app) > Incoming Webhooks > Add New
# Webhook to Workspace. Looks like https://hooks.slack.com/services/T.../B.../...
SLACK_WEBHOOK_URL = ""
