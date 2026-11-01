#!/usr/bin/env python3
"""Local, manifest-scoped review server for the Nino style review set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "outputs/dataset/style-review-v1"
DECISIONS_SCHEMA = "nino-style-review-decisions-v1"
MAX_BODY = 1024 * 1024
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inside_root(relpath):
    if not isinstance(relpath, str) or not relpath or "\\" in relpath:
        raise ValueError("Invalid project-relative audio path")
    p = Path(relpath)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts):
        raise ValueError("Audio path must stay inside project root")
    resolved = (ROOT / p).resolve(strict=True)
    if not resolved.is_relative_to(ROOT) or not resolved.is_file() or resolved.suffix.lower() != ".wav":
        raise ValueError("Audio asset must be an existing WAV inside project root")
    return resolved


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class ReviewApp:
    def __init__(self, dataset_dir, store_path=None):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.root_page = self.dataset_dir / "review.html"
        self.manifest_path = self.dataset_dir / "manifest.json"
        self.store_path = Path(store_path).resolve() if store_path else self.dataset_dir / "decisions.json"
        self.lock = threading.RLock()
        self.manifest = load_json(self.manifest_path)
        if self.manifest.get("schema") != "nino-style-review-v1":
            raise ValueError("Unsupported or missing manifest schema")
        self.collection_fp = self.manifest.get("collection_fingerprint")
        self.compatible_old_fps = set(self.manifest.get("compatible_previous_collection_fingerprints", []))
        self.migration_proof = self.manifest.get("label_addition_migration")
        if self.compatible_old_fps:
            if (not isinstance(self.migration_proof, dict) or
                    self.migration_proof.get("from_collection_fingerprint") not in self.compatible_old_fps):
                raise ValueError("Invalid compatible collection migration proof")
        self.parent_fp = self.manifest.get("parent_dataset_fingerprint")
        if not isinstance(self.collection_fp, str) or not isinstance(self.parent_fp, str):
            raise ValueError("Manifest fingerprints are required")
        self.clips = {}
        self.registry = {}
        for c in self.manifest.get("clips", []):
            cid = c.get("clip_id")
            if not isinstance(cid, str) or not ID_RE.fullmatch(cid) or cid in self.clips:
                raise ValueError("Manifest has invalid or duplicate clip_id")
            self.clips[cid] = c
            self.registry[("clip", cid)] = inside_root(c.get("audio_path"))
            if c.get("context_audio_path"):
                self.registry[("context", cid)] = inside_root(c["context_audio_path"])
        if not self.clips:
            raise ValueError("Manifest contains no clips")
        # Source audio is exposed only when the manifest explicitly registers it.
        for s in self.manifest.get("sources", []):
            sid = s.get("source_id")
            path = s.get("context_audio_path", s.get("audio_path"))
            if isinstance(sid, str) and ID_RE.fullmatch(sid) and path:
                self.registry[("context", sid)] = inside_root(path)
        labels = self.manifest.get("label_schema", {})
        self.label_ids = {k: {x["id"] for x in labels.get(k, []) if isinstance(x, dict) and isinstance(x.get("id"), str)}
                          for k in ("attitude", "effort", "delivery")}
        self.caption_parts = {k: {x["id"]: x.get("caption_ja", "") for x in labels.get(k, []) if isinstance(x, dict)}
                              for k in ("attitude", "effort", "delivery")}
        self.clip_hashes = {k: v.get("clip_audio_sha256") for k, v in self.clips.items()}

    def empty_store(self):
        return {"schema": DECISIONS_SCHEMA, "collection_fingerprint": self.collection_fp,
                "parent_dataset_fingerprint": self.parent_fp, "updated_at": None, "decisions": []}

    def read_store(self):
        if not self.store_path.exists():
            return self.empty_store()
        data = load_json(self.store_path)
        if isinstance(data, dict) and data.get("collection_fingerprint") in self.compatible_old_fps:
            with self.lock:
                data = load_json(self.store_path)
                if data.get("collection_fingerprint") in self.compatible_old_fps:
                    old_fp = data["collection_fingerprint"]
                    self.validate_store(data, accepted_fp=old_fp)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    backup = self.store_path.with_name(self.store_path.stem + ".migration-backup-" + stamp + self.store_path.suffix)
                    backup.write_bytes(self.store_path.read_bytes())
                    data["collection_fingerprint"] = self.collection_fp
                    data["migration"] = {"from_collection_fingerprint": old_fp, "to_collection_fingerprint": self.collection_fp,
                                          "migrated_at": now_iso()}
                    atomic_json(self.store_path, data)
        self.validate_store(data)
        return data

    def validate_decision(self, d, preserve_timestamp=False):
        if not isinstance(d, dict):
            raise ValueError("Each decision must be an object")
        cid = d.get("clip_id")
        if cid not in self.clips or d.get("clip_audio_sha256") != self.clip_hashes[cid]:
            raise ValueError("Unknown clip or clip audio SHA-256 mismatch")
        if d.get("status") not in ("draft", "confirmed", "needs_review"):
            raise ValueError("Invalid decision status")
        for key in ("attitude", "effort", "delivery"):
            val = d.get(key)
            if val is not None and val != "" and val not in self.label_ids[key]:
                raise ValueError("Invalid " + key + " label")
        if not isinstance(d.get("heard_audio"), bool) or not isinstance(d.get("anchor", False), bool):
            raise ValueError("heard_audio and anchor must be booleans")
        if not isinstance(d.get("notes", ""), str) or len(d.get("notes", "")) > 4000:
            raise ValueError("notes must be a string of at most 4000 characters")
        confirmed = d["status"] == "confirmed"
        if confirmed and (not d["heard_audio"] or any(d.get(k) not in self.label_ids[k] for k in self.label_ids)):
            raise ValueError("Confirmed decisions require listened audio and all three valid labels")
        if d.get("anchor", False) and not confirmed:
            raise ValueError("Only confirmed decisions may be anchors")
        def usable_timestamp(value):
            if not isinstance(value, str):
                return None
            try:
                datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
                return value
            except ValueError:
                return None

        out = {"clip_id": cid, "clip_audio_sha256": self.clip_hashes[cid], "status": d["status"],
               "attitude": d.get("attitude") or "", "effort": d.get("effort") or "",
               "delivery": d.get("delivery") or "", "heard_audio": d["heard_audio"],
               "notes": d.get("notes", ""), "anchor": d.get("anchor", False),
               "caption_ja": "", "label_basis": "", "confirmed_at": None,
               "updated_at": usable_timestamp(d.get("updated_at")) if preserve_timestamp else None}
        out["updated_at"] = out["updated_at"] or now_iso()
        if confirmed:
            out.update(caption_ja=" ".join(self.caption_parts[k][out[k]] for k in ("attitude", "effort", "delivery") if self.caption_parts[k].get(out[k])).strip(),
                       label_basis="human_listening", confirmed_at=(usable_timestamp(d.get("confirmed_at")) if preserve_timestamp else None) or now_iso())
        return out

    def validate_store(self, data, accepted_fp=None):
        if not isinstance(data, dict) or data.get("schema") != DECISIONS_SCHEMA:
            raise ValueError("Invalid decisions schema")
        fp = data.get("collection_fingerprint")
        if (fp != self.collection_fp and fp not in self.compatible_old_fps and fp != accepted_fp) or data.get("parent_dataset_fingerprint") != self.parent_fp:
            raise ValueError("Decision store fingerprint mismatch")
        ds = data.get("decisions")
        if not isinstance(ds, list):
            raise ValueError("decisions must be an array")
        seen = set()
        for d in ds:
            checked = self.validate_decision(d, preserve_timestamp=True)
            if checked["clip_id"] in seen:
                raise ValueError("Duplicate clip decision")
            seen.add(checked["clip_id"])

    def save_decision(self, payload):
        if not isinstance(payload, dict) or payload.get("collection_fingerprint") not in ({self.collection_fp} | self.compatible_old_fps):
            raise ValueError("Collection fingerprint mismatch")
        candidate = self.validate_decision(payload.get("decision"))
        with self.lock:
            store = self.read_store()
            existing = {d["clip_id"]: d for d in store["decisions"]}
            existing[candidate["clip_id"]] = candidate
            store.update(updated_at=now_iso(), decisions=list(existing.values()))
            atomic_json(self.store_path, store)
            return store

    def import_store(self, incoming):
        self.validate_store(incoming)
        with self.lock:
            current = self.read_store()
            merged = {d["clip_id"]: d for d in current["decisions"]}
            for d in incoming["decisions"]:
                merged[d["clip_id"]] = self.validate_decision(d, preserve_timestamp=True)
            result = self.empty_store()
            if self.migration_proof:
                result["migration"] = {"from_collection_fingerprint": self.migration_proof["from_collection_fingerprint"],
                                        "to_collection_fingerprint": self.collection_fp}
            result.update(updated_at=now_iso(), decisions=list(merged.values()))
            if self.store_path.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                backup = self.store_path.with_name(self.store_path.stem + ".backup-" + stamp + self.store_path.suffix)
                # Preserve bytes exactly before the first successful import mutation.
                backup.write_bytes(self.store_path.read_bytes())
            atomic_json(self.store_path, result)
            return result


class Handler(BaseHTTPRequestHandler):
    server_version = "NinoReview/1"

    @property
    def app(self): return self.server.app

    def send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(data)

    def fail(self, code, msg): self.send_json(code, {"error": msg})

    def origin_ok(self):
        host = self.headers.get("Host", "")
        if not re.fullmatch(r"127\.0\.0\.1(?::\d{1,5})?", host): return False
        origin = self.headers.get("Origin")
        return origin is None or origin == "http://" + host

    def body_json(self):
        rawlen = self.headers.get("Content-Length", "")
        if not rawlen.isdigit() or int(rawlen) > MAX_BODY: raise ValueError("Request body missing or exceeds 1 MiB")
        raw = self.rfile.read(int(rawlen))
        try: return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e: raise ValueError("Malformed UTF-8 JSON") from e

    def do_GET(self):
        if not self.origin_ok(): return self.fail(403, "Host or Origin is not allowed")
        path = urlsplit(self.path).path
        try:
            if path == "/health":
                return self.send_json(200, {"status": "ready", "dataset_id": self.app.manifest.get("dataset_id"),
                    "collection_fingerprint": self.app.collection_fp, "count": len(self.app.clips)})
            if path == "/api/dataset": return self.send_json(200, self.app.manifest)
            if path == "/api/review":
                with self.app.lock: return self.send_json(200, self.app.read_store())
            if path == "/api/export":
                data = json.dumps(self.app.read_store(), ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="nino-style-review-decisions.json"')
                self.send_header("Content-Length", str(len(data))); self.end_headers(); return self.wfile.write(data)
            if path == "/":
                b = self.app.root_page.read_bytes(); self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
                self.end_headers(); return self.wfile.write(b)
            m = re.fullmatch(r"/audio/(clip|context)/([^/]+)\.wav", path)
            if m:
                kind, cid = m.group(1), unquote(m.group(2))
                asset = self.app.registry.get((kind, cid))
                if not asset: return self.fail(404, "Unknown audio asset")
                return self.send_audio(asset)
            return self.fail(404, "Not found")
        except FileNotFoundError: return self.fail(404, "Requested asset is unavailable")
        except (ValueError, json.JSONDecodeError) as e: return self.fail(500, str(e))

    def send_audio(self, path):
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end, status = 0, size - 1, 200
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if not m or (not m.group(1) and not m.group(2)):
                return self.range_error(size)
            if m.group(1):
                start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else size - 1
            else:
                suffix = int(m.group(2)); start = max(0, size - suffix); end = size - 1
            if start >= size or end < start: return self.range_error(size)
            end = min(end, size - 1); status = 206
        length = end - start + 1
        self.send_response(status); self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(length))
        if status == 206: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start); remaining = length
            while remaining:
                chunk = f.read(min(65536, remaining))
                if not chunk: break
                self.wfile.write(chunk); remaining -= len(chunk)

    def range_error(self, size):
        self.send_response(416); self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0"); self.end_headers()

    def do_POST(self):
        if not self.origin_ok(): return self.fail(403, "Host or Origin is not allowed")
        path = urlsplit(self.path).path
        if path not in ("/api/review", "/api/import"): return self.fail(404, "Not found")
        if self.headers.get_content_type() != "application/json": return self.fail(415, "Content-Type must be application/json")
        try:
            payload = self.body_json()
            result = self.app.save_decision(payload) if path == "/api/review" else self.app.import_store(payload)
            return self.send_json(200, result)
        except ValueError as e: return self.fail(400, str(e))
        except (OSError, json.JSONDecodeError) as e: return self.fail(500, "Could not persist review data: " + str(e))

    def log_message(self, fmt, *args): pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    app = ReviewApp(args.dataset_dir, args.store)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.app = app
    print(f"Nino review ready: http://127.0.0.1:{httpd.server_port}/ ({app.manifest.get('dataset_id')}, {len(app.clips)} clips)", flush=True)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: httpd.server_close()


if __name__ == "__main__": main()
