# Multi-user isolation — manual test checklist

Two testers (A and B) each logged in as a different Directus account.
Run steps in parallel unless the step says "A only" or "B only".

---

## Prerequisites

- Two browsers (or one normal + one private window) signed into different Directus accounts.
- The app running with `AUTH_ENABLED=1` and a valid `FLASK_SECRET_KEY`.
- `COMMS_BACKEND` set to `vm_comms` for the HESTIA scan_id tests (sections 3–4); sections 1–2 apply to both backends.
- A small set of test images (≥3 JPEGs) ready for each tester.
- Shell access to the server to inspect the filesystem and HESTIA API during the run.

---

## 1. Session and identity isolation

| # | Step | Expected |
|---|------|----------|
| 1.1 | A logs in; B stays on the login page. Navigate A to `/welcome`. | A sees the welcome screen; B is not redirected. |
| 1.2 | Inspect the Flask session cookie in each browser's DevTools → Application → Cookies. | The cookies are different; neither contains the other's `user_email`. |
| 1.3 | B logs in with a different Directus account. Navigate B to `/welcome`. | B's session shows B's email; A's session is unchanged. |
| 1.4 | On the server: `ls data/in/` | Two subdirectories — one per `user_slug` — are created (or will be after first upload). They do not share a directory. |

---

## 2. Concurrent upload with the same dataset name

Both testers use **exactly the same dataset name** (e.g. `test_scan`).

| # | Step | Expected |
|---|------|----------|
| 2.1 | A and B both navigate to `/setup` simultaneously. | Each sees their own `/setup` page. The `in_mnt` shown in the page footer (if visible) differs between them (`…/<slug_a>/` vs `…/<slug_b>/`). |
| 2.2 | A uploads images; B uploads a **different** set of images — same dataset name, different content. | Both uploads succeed (HTTP 200 with `ok: true`). |
| 2.3 | On the server: `ls data/in/<slug_a>/test_scan/` and `ls data/in/<slug_b>/test_scan/` | Each directory contains only that user's images. Neither directory is shared or cross-pollinated. |
| 2.4 | A navigates to `/home`; B navigates to `/home`. | Each sees only their own dataset name. Neither is redirected to `/setup` because of the other's state. |
| 2.5 | A navigates to `/pick`; B navigates to `/pick`. | The frame list shown to each user contains only their own images. |

---

## 3. HESTIA scan_id prefixing (vm_comms mode only)

| # | Step | Expected |
|---|------|----------|
| 3.1 | After step 2.2, on the server inspect `data/in/<slug_a>/test_scan/.scan_id` and `data/in/<slug_b>/test_scan/.scan_id`. | File for A contains `<slug_a>__test_scan`; file for B contains `<slug_b>__test_scan`. They differ. |
| 3.2 | Query HESTIA: `GET /nefele?scan_id=<slug_a>__test_scan` | Returns only A's jobs. |
| 3.3 | Query HESTIA: `GET /nefele?scan_id=<slug_b>__test_scan` | Returns only B's jobs. |
| 3.4 | Query HESTIA: `GET /nefele?scan_id=test_scan` (unprefixed) | Returns no rows — the unprefixed name was never registered as a scan_id. |
| 3.5 | A creates a second dataset also named `test_scan` (delete the first one locally, repeat upload). | The HESTIA collision check (`scan_exists`) looks up `<slug_a>__test_scan` and rejects the duplicate. B's `<slug_b>__test_scan` is not affected. |
| 3.6 | A posts points and triggers preview (`/save`). A polls `/save/status`. | A's job is created under `<slug_a>__test_scan`. B's `/save/status` poll returns its own job state, unaffected. |

---

## 4. Results isolation

| # | Step | Expected |
|---|------|----------|
| 4.1 | Once A's pipeline completes, A navigates to `/results`. | A sees A's result files only. |
| 4.2 | B navigates to `/results` before B's pipeline has completed. | B sees "no results yet" (or the spinner), not A's files. |
| 4.3 | (vm_comms) Query HESTIA: `GET /reconstructions?scan_id=<slug_a>__test_scan` | Returns A's reconstruction record. |
| 4.4 | (vm_comms) B's `/results/files` response must not contain A's reconstruction. Confirm by checking the `scan_id` used in the request: it should be `<slug_b>__test_scan`. | B's poll returns `ready: false` (or B's own files once done). A's record does not appear. |
| 4.5 | (shared_fs) A downloads `/results/zip`. | The ZIP contains only A's mesh files. Extract and confirm the `.obj` dataset stem matches A's dataset. |
| 4.6 | Try to fetch A's result file directly as B: `GET /results/file/<relative_path_from_A>`. | Returns 404 — B's `out_root` doesn't contain A's path, so the file is not found. |

---

## 5. Preview isolation

| # | Step | Expected |
|---|------|----------|
| 5.1 | A triggers a preview (`/save`). While A's preview is loading, B also triggers a preview for B's own scan. | Both return `ok: true`. A's response contains `job_id` for `<slug_a>__test_scan`; B's for `<slug_b>__test_scan`. |
| 5.2 | A polls `/save/status?job_id=<A's job_id>`. B polls `/save/status?job_id=<B's job_id>`. | Each poll returns status for its own job. Neither returns the other's status. |
| 5.3 | Once A's preview is ready, navigate A to `/preview/<filename>`. | A sees A's preview mask images (under A's `preview_dir`). |
| 5.4 | Try to fetch A's preview URL in B's browser session. | Returns 404 — B's `preview_dir` is under a different path and does not contain A's files. |

---

## 6. Job cancellation scope (vm_comms mode only)

This verifies that `/setup`'s pre-upload cancel only cancels the requesting user's jobs.

| # | Step | Expected |
|---|------|----------|
| 6.1 | A has an active HESTIA job (e.g. status `points_submitted`). | Confirm via `GET /nefele?scan_id=<slug_a>__test_scan`. |
| 6.2 | B starts a new `/setup` upload (which triggers `_cancel_active_jobs`). | B's upload succeeds. |
| 6.3 | Re-query HESTIA for A's job: `GET /nefele/<A's job_id>`. | A's job is **not** cancelled — its status is unchanged. |
| 6.4 | Query HESTIA for B's previous job (if any): `GET /nefele?scan_id=<slug_b>__<old_name>`. | B's old job is cancelled (if it was active). |

---

## 7. Logout and session teardown

| # | Step | Expected |
|---|------|----------|
| 7.1 | A's `textailes_refresh_token` cookie expires or is deleted manually. | A is redirected to the Directus login page on the next request. B's session is unaffected. |
| 7.2 | After A re-logs in, A navigates to `/welcome`. | A's session shows the correct `user_email` again; A's datasets are still present under `data/in/<slug_a>/`. |

---

## Failure modes to watch for

- **Cross-directory reads**: if either tester can see the other's frames at `/frame?i=0`, the `frames()` helper is returning a shared list — check `g.frames` is not being accidentally shared across requests.
- **Stale `g.cfg`**: if a tester sees the wrong dataset name on `/home` after uploading, `g.cfg` may not have been invalidated after `rebind_dataset`.
- **Wrong scan_id in `.scan_id` file**: if the file contains the bare dataset name instead of the prefixed form, the HESTIA job will be created under the unprefixed scan_id and collision checks will fail.
- **`_cancel_active_jobs` over-cancelling**: if A's job disappears during B's upload (step 6), the prefix filter in `_cancel_active_jobs` is not working — verify `job.scan_id.startswith(f"{user_slug}__")` is evaluated before `cancel_job` is called.
