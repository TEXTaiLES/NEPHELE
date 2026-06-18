"""WSGI entry point. Production: `gunicorn app.wsgi:app`. Dev: `python -m app.wsgi`."""

from __future__ import annotations

import os

from . import create_app

app = create_app()


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", "5000"))
    # threaded=True so a slow poll (e.g. rglob on the SuGaR outputs tree)
    # can't block other requests on the single Werkzeug worker.
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
