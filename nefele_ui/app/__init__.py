"""Point Picker UI — Flask application package.

The :func:`create_app` factory builds a Flask app from environment-driven
config, registers the route blueprints, optionally enables Directus auth, and
caches the discovered frame list so requests don't re-scan the volume.

When no dataset is configured (env var unset and no coordination file), the
home route redirects to ``/setup`` where the user picks a name and uploads
images. After upload, :func:`rebind_dataset` swaps the active config in-place.
"""

from __future__ import annotations

import logging
import os

from flask import Flask, url_for

from .auth import init_auth
from .config import load_config
from .routes import (
    home_bp,
    hestia_bp,
    picker_bp,
    preview_bp,
    results_bp,
    setup_bp,
    welcome_bp,
)


def rebind_dataset(user_slug: str, name: str):
    """Return a fresh per-user Config after /setup has written the coordination file.

    Only touches that user's ``<IN_MNT>/<user_slug>/.active_dataset``; never
    mutates ``os.environ`` or any process-global state.
    """
    return load_config(user_slug)


def create_app() -> Flask:
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    auth_enabled = os.environ.get("AUTH_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = Flask(__name__, template_folder="templates", static_folder="static")
    # Disable Flask's default 12h static cache + re-read Jinja templates on
    # every request so CSS/JS/HTML edits land on the next browser refresh.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    if auth_enabled:
        # auth.py caches the Directus access_token in the Flask session so we
        # don't hit /auth/refresh on every poll request — but that needs a
        # signed-cookie secret. Refuse to start without one rather than fall
        # back to Flask's "<no secret>" stub which would crash on the first
        # session.write.
        secret = os.environ.get("FLASK_SECRET_KEY", "").strip()
        if not secret:
            raise RuntimeError(
                "AUTH_ENABLED=1 but FLASK_SECRET_KEY is not set — "
                "generate one (e.g. `python -c 'import secrets;print(secrets.token_hex(32))'`) "
                "and add it to .env."
            )
        app.secret_key = secret
        init_auth(app)

    app.register_blueprint(home_bp)
    app.register_blueprint(hestia_bp)
    app.register_blueprint(picker_bp)
    app.register_blueprint(preview_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(setup_bp)
    app.register_blueprint(welcome_bp)

    # Cache-buster: append ?v=<mtime> to every static URL so a stale browser
    # tab fetches fresh JS/CSS the moment the file on disk changes.
    @app.context_processor
    def _inject_versioned_url_for():
        def versioned_url_for(endpoint: str, **values):
            if endpoint == "static":
                filename = values.get("filename")
                if filename:
                    fp = os.path.join(app.static_folder, filename)
                    try:
                        values["v"] = int(os.stat(fp).st_mtime)
                    except OSError:
                        pass
            return url_for(endpoint, **values)
        return {"url_for": versioned_url_for}

    return app
