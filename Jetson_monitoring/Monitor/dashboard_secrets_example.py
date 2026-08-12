"""Template for dashboard_secrets.py (gitignored, never committed). Copy this file to
dashboard_secrets.py in the same directory and fill in a real password.

    cp dashboard_secrets_example.py dashboard_secrets.py

NOT named secrets.py -- same reason as webhook_secrets_example.py: Monitor/ sits on
sys.path for every pipeline script (see config.py), and a local module named exactly
`secrets.py` shadows the stdlib `secrets` module for the whole process -- which
previously crashed analyse.py (and therefore its live-stream) the moment such a file
existed. analyse.py's password check uses secrets.compare_digest from the real stdlib
module, so this collision matters here specifically, not just as a style preference.
"""

# HTTP Basic Auth password for the live-stream dashboard (frontend_private/) --
# analyse.py checks this on every request before serving anything, including the
# static HTML/CSS/JS files themselves. Any username is accepted; only the password
# is checked.
DASHBOARD_PASSWORD = ""
