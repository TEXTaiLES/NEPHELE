from __future__ import annotations
import os
import requests
from urllib.parse import quote
from flask import request, redirect, g , make_response


# --- Directus API base ---
DIRECTUS_URL = "https://textailes.athenarc.gr"


# --- Cookie name ---
COOKIE_NAME = "textailes_refresh_token"


LOGIN_URL = "https://textailes.athenarc.gr/archive/user/login"
APP_BASE  = "http://nephele.textailes.athenarc.gr:8093"  


PUBLIC_PATHS = {"/health"}          
PUBLIC_PREFIXES = ("/static/",)     

def redirect_to_login():
    next_path = request.full_path if request.query_string else request.path
    redirect_url = APP_BASE.rstrip("/") + next_path
    return redirect(f"{LOGIN_URL}?redirect_url={quote(redirect_url, safe=':/?=&')}")


def directus_refresh(refresh_token: str):
    try:
        r = requests.post(
            f"{DIRECTUS_URL}/auth/refresh",
            json={"refresh_token": refresh_token},
            timeout=8,
        )
        if r.status_code != 200:
            print("refresh failed:", r.status_code, r.text[:200], flush=True)
            return None
        data = r.json()
        return data.get("data") or data
    except Exception as e:
        print("refresh exception:", e, flush=True)
        return None


def init_auth(app):
    @app.before_request
    def auth_gate():
        path = request.path
        if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return None

        refresh_token = request.cookies.get(COOKIE_NAME)
        if not refresh_token:
            return redirect_to_login()

        tokens = directus_refresh(refresh_token)
        if not tokens:
            return redirect_to_login()

        g.access_token = tokens.get("access_token")

        new_refresh = tokens.get("refresh_token")
        if new_refresh and new_refresh != refresh_token:
            g.new_refresh_token = new_refresh
        else:
            g.new_refresh_token = None

        return None

    @app.after_request
    def apply_refresh_cookie(resp):
        new_refresh = getattr(g, "new_refresh_token", None)
        if not new_refresh:
            return resp

        host = request.host.split(":")[0]
        domain = ".textailes.athenarc.gr" if host.endswith("textailes.athenarc.gr") else None
        is_https = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"

        resp.set_cookie(
            COOKIE_NAME,
            new_refresh,
            httponly=True,
            secure=is_https,
            samesite="Lax",
            path="/",
            domain=domain,
        )
        return resp

