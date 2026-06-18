"""Landing page shown as the app's entry point.

Renders context-aware action buttons:
  - dataset configured + prompts.json exists → Continue / Results / New dataset
  - dataset configured + no prompts yet      → Continue (to picker) / New dataset
  - nothing configured                       → Get started
"""

from __future__ import annotations

from flask import Blueprint, render_template

from ._helpers import cfg, frames

bp = Blueprint("welcome", __name__)


@bp.get("/welcome")
def welcome():
    c = cfg()
    return render_template(
        "welcome.html",
        is_configured=c.is_configured,
        dataset_name=c.dataset_name,
        has_prompts=c.prompts_json.is_file() if c.is_configured else False,
        has_frames=bool(frames()),
    )
