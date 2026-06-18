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
from .config import Config, load_config
from .routes import (
    home_bp,
    hestia_bp,
    picker_bp,
    preview_bp,
    results_bp,
    setup_bp,
    welcome_bp,
)
from .services.frames import resolve_frames


def _attach_state(app: Flask, cfg: Config) -> None:
    """Store the live config + cached frame list on the app."""
    app.config["APP_CONFIG"] = cfg
    if cfg.is_configured:
        app.config["APP_FRAMES"] = resolve_frames(cfg.input_dir, cfg.index_suffix)
    else:
        app.config["APP_FRAMES"] = []


def rebind_dataset(app: Flask, name: str) -> Config:
    """Re-read config with a new dataset name and refresh the cached frame list.

    Called from /setup after a successful upload. Updates ``os.environ`` so a
    subsequent ``load_config()`` (e.g. after a process restart with the same
    coordination file) yields the same result.
    """
    os.environ["DATASET_NAME"] = name
    cfg = load_config()
    _attach_state(app, cfg)
    return cfg


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("FLASK_DEBUG") == "1" else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    # This is a single-user local tool. Disable Flask's default 12h static
    # cache + re-read Jinja templates on every request so CSS/JS/HTML edits
    # land on the next browser refresh instead of leaving users stuck.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    _attach_state(app, cfg)

    if cfg.auth_enabled:
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
