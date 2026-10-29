"""Append-only audit trail (SEBI IA Reg. 19 record-keeping; AI decisions must be traceable).

Each line is JSON carrying the app version, rule-set version and the hash of the previous record.
What the hash chain DETECTS: editing, deleting, inserting or reordering any record in the middle of
the file, when checked with verify_chain().
What it does NOT detect: someone who rewrites the whole file and recomputes every hash, or who
truncates the most recent records. It is tamper-EVIDENT, not tamper-proof; production use would add
write-restricted storage, periodic off-site anchoring of the latest hash, and access control.
Client names are stored only as a SHA-256 prefix (pseudonymous, not anonymous); PII patterns are masked before writing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import APP_VERSION, DATA_CUTOFF, RULESET_VERSION
from .guardrails import mask_pii

LOG_PATH = Path(__file__).resolve().parent.parent / "outputs" / "audit_log.jsonl"


def _last_hash(path: Path) -> str:
    if not path.exists():
        return "GENESIS"
    last = ""
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last)["hash"] if last else "GENESIS"


def log(event: str, client_ref: str, payload: dict, rules_fired: list[str] | None = None,
        path: Path | None = None) -> dict:
    path = path or LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, default=str, ensure_ascii=False)
    body, _ = mask_pii(body)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "client_ref": hashlib.sha256(client_ref.encode()).hexdigest()[:12],
        "rules_fired": rules_fired or [],
        "app_version": APP_VERSION,
        "ruleset": RULESET_VERSION,
        "data_cutoff": DATA_CUTOFF,
        "payload": json.loads(body),
        "prev": _last_hash(path),
    }
    rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def verify_chain(path: Path | None = None) -> tuple[bool, int]:
    path = path or LOG_PATH
    if not path.exists():
        return True, 0
    prev, n = "GENESIS", 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            h = rec.pop("hash")
            if rec["prev"] != prev:
                return False, n
            if hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != h:
                return False, n
            prev, n = h, n + 1
    return True, n


def read(path: Path | None = None, limit: int = 200) -> list[dict]:
    path = path or LOG_PATH
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows[-limit:]
