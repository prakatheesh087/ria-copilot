"""Append-only audit trail (SEBI IA Reg. 19 record-keeping; AI decisions must be traceable).

Each line is JSON with a hash chain, so any later edit to a past record is detectable.
PII is masked before writing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

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
