"""Show SuGaR output files and let the user download them."""

from __future__ import annotations

import io
import zipfile
from dataclasses import asdict
from pathlib import Path

from flask import Blueprint, abort, redirect, render_template, send_file, url_for

from ..services.pipeline import read_status
from ..services.results import (
    friendly_name, list_outputs, list_pgsr_outputs, read_model, read_patched,
)
from ._helpers import cfg, json_err, json_ok

bp = Blueprint("results", __name__)


@bp.get("/pipeline/status")
def pipeline_status():
    """JSON status of the auto-triggered pipeline for the active dataset."""
    c = cfg()
    if not c.is_configured:
        return json_err("No dataset configured", http=400)
    status = read_status(c.indexed_dir, c.dataset_name)
    return json_ok(**status.to_dict())


@bp.get("/results")
def page():
    c = cfg()
    if not c.is_configured:
        return redirect(url_for("welcome.welcome"))
    return render_template(
        "results.html",
        ds=c.dataset_name,
        model=read_model(c.in_mnt, c.dataset_name),
    )


def _results_root(c):
    model = read_model(c.in_mnt, c.dataset_name)
    if model == "pgsr":
        return (c.pgsr_results_root, model)
    if model == "fastpgsr":
        return (c.fastpgsr_results_root, model)
    return (c.sugar_results_root, model)


@bp.get("/results/files")
def files_json():
    """JSON listing for the front-end poll loop."""
    c = cfg()
    if not c.is_configured:
        return json_err("No dataset configured", http=400)
    root, model = _results_root(c)
    if model == "pgsr":
        files = list_pgsr_outputs(root, c.dataset_name)
    else:
        files = list_outputs(root, c.dataset_name)
    return json_ok(
        dataset=c.dataset_name,
        model=model,
        ready=bool(files),
        files=[asdict(f) for f in files],
    )


def _safe_resolve(root: Path, relative: str) -> Path:
    """Resolve ``root/relative`` and guarantee the result stays inside ``root``."""
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        abort(404)
    if not target.is_file():
        abort(404)
    return target


@bp.get("/results/file/<path:relative>")
def download_one(relative: str):
    c = cfg()
    if not c.is_configured:
        return json_err("No dataset configured", http=400)
    root, model = _results_root(c)
    target = _safe_resolve(root, relative)
    data = read_patched(target, c.dataset_name)
    dl_name = target.name if model == "pgsr" else friendly_name(target.name, c.dataset_name)
    return send_file(io.BytesIO(data), as_attachment=True, download_name=dl_name)


@bp.get("/results/zip")
def download_zip():
    c = cfg()
    if not c.is_configured:
        return json_err("No dataset configured", http=400)

    root, model = _results_root(c)
    if model == "pgsr":
        files = list_pgsr_outputs(root, c.dataset_name)
    else:
        files = list_outputs(root, c.dataset_name)
    if not files:
        return json_err("No results to download yet.", http=404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            src = _safe_resolve(root, f.relative)
            arc_name = src.name if model == "pgsr" else f.name
            zf.writestr(arc_name, read_patched(src, c.dataset_name))
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{c.dataset_name}.zip",
    )
