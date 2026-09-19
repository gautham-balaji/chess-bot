"""Phase 0 baseline: exercise the EXISTING Flask endpoints via the test client.

No route, handler, or global is modified. app.py holds module-level game state
(app.py:13-20), so requests are ordered deliberately and /reset is used to
return to a known state. That ordering dependency is itself a recorded finding.

Usage:  python baseline/scripts/measure_api.py baseline/api_results.json
"""
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# (label, method, path, json_body, raw_body, content_type, intent)
REQUESTS = [
    ("reset_initial",      "POST", "/reset",       {},               None, None, "setup"),
    ("index",              "GET",  "/",            None,             None, None, "happy path"),
    ("state",              "GET",  "/state",       None,             None, None, "happy path"),
    ("model_info",         "GET",  "/model_info",  None,             None, None, "happy path"),
    ("move_legal",         "POST", "/move",        {"uci": "e2e4"},  None, None, "happy path"),
    ("engine_move",        "POST", "/engine_move", {},               None, None, "happy path"),
    ("analyse",            "POST", "/analyse",     {},               None, None, "happy path"),
    ("benchmark",          "POST", "/benchmark",   {},               None, None, "happy path (uses Stockfish)"),
    ("game_stats",         "POST", "/game_stats",  {},               None, None, "happy path"),
    ("move_illegal",       "POST", "/move",        {"uci": "e2e5"},  None, None, "invalid input"),
    ("move_bad_uci",       "POST", "/move",        {"uci": "zzzz"},  None, None, "invalid input"),
    ("move_empty_body",    "POST", "/move",        {},               None, None, "invalid input"),
    ("move_non_json",      "POST", "/move",        None, "notjson", "text/plain", "invalid input"),
    ("move_wrong_method",  "GET",  "/move",        None,             None, None, "invalid input"),
    ("forfeit",            "POST", "/forfeit",     {},               None, None, "happy path"),
    ("reset_final",        "POST", "/reset",       {},               None, None, "teardown"),
    ("engine_move_wrong_turn", "POST", "/engine_move", {},           None, None,
     "invalid input: after reset it is White's turn, engine plays Black"),
]


def main():
    out_path = sys.argv[1]

    t0 = time.perf_counter()
    import app as application  # noqa: PLC0415 - timing this import is the point
    import_s = time.perf_counter() - t0
    print(f"import app: {import_s:.2f}s", flush=True)

    client = application.app.test_client()
    rows = []
    for label, method, path, body, raw, ctype, intent in REQUESTS:
        t = time.perf_counter()
        try:
            if method == "GET":
                resp = client.get(path)
            elif raw is not None:
                resp = client.post(path, data=raw, content_type=ctype)
            else:
                resp = client.post(path, json=body)
            elapsed = (time.perf_counter() - t) * 1000.0
            try:
                payload = resp.get_json()
            except Exception:
                payload = None
            rows.append({
                "label": label, "intent": intent, "method": method, "path": path,
                "request_body": body if raw is None else raw,
                "status_code": resp.status_code,
                "content_type": resp.headers.get("Content-Type"),
                "is_json": payload is not None,
                "time_ms": round(elapsed, 2),
                "error_field": (payload or {}).get("error") if isinstance(payload, dict) else None,
                "response_keys": sorted(payload.keys())[:15] if isinstance(payload, dict) else None,
                "response_bytes": len(resp.data),
                "exception": None,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "label": label, "intent": intent, "method": method, "path": path,
                "status_code": None, "time_ms": round((time.perf_counter() - t) * 1000.0, 2),
                "exception": repr(exc),
            })
        r = rows[-1]
        print(f"  {label:24s} {method:4s} {path:14s} -> {r.get('status_code')} "
              f"{r['time_ms']:.0f}ms json={r.get('is_json')}", flush=True)

    payload = {
        "schema_version": 1,
        "generated_by": "baseline/scripts/measure_api.py",
        "method": "flask test_client(); no live server, no network",
        "import_app_seconds": round(import_s, 2),
        "state_model_note": (
            "app.py uses module-level globals for board/history/captures "
            "(app.py:13-20). Requests are therefore order-dependent and share "
            "state; these timings reflect that specific sequence."
        ),
        "requests": rows,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
