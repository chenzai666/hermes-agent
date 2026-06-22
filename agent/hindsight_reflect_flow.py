from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agent.memory_layers import ingest_recall_excerpt, uplift_persona, uplift_scenario
from hermes_constants import get_hermes_home


REPORTS_DIRNAME = "hindsight-reports"
STATE_FILENAME = "state.json"


@dataclass(frozen=True)
class ReflectBackflowResult:
    ok: bool
    changed: bool
    skipped: bool
    reason: str
    bank_id: str
    query: str
    destination: str
    output_path: str
    latest_path: str
    memory_path: str
    content_hash: str
    timestamp: str
    state_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "changed": self.changed,
            "skipped": self.skipped,
            "reason": self.reason,
            "bank_id": self.bank_id,
            "query": self.query,
            "destination": self.destination,
            "output_path": self.output_path,
            "latest_path": self.latest_path,
            "memory_path": self.memory_path,
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
            "state_path": self.state_path,
        }


def get_reports_dir(output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return get_hermes_home() / REPORTS_DIRNAME


def get_state_path(output_dir: str | Path | None = None) -> Path:
    return get_reports_dir(output_dir) / STATE_FILENAME


def load_state(output_dir: str | Path | None = None) -> dict[str, Any]:
    path = get_state_path(output_dir)
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}
    return {}


def save_state(state: Mapping[str, Any], output_dir: str | Path | None = None) -> Path:
    reports_dir = get_reports_dir(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / STATE_FILENAME
    path.write_text(json.dumps(dict(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def content_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").strip().encode("utf-8")).hexdigest()


def record_reflect_run(
    *,
    bank_id: str,
    query: str,
    rendered: str,
    output_path: str | Path,
    latest_path: str | Path,
    output_dir: str | Path | None = None,
    usage: Mapping[str, Any] | None = None,
) -> Path:
    existing = load_state(output_dir)
    now = datetime.now(timezone.utc).isoformat()
    state = dict(existing)
    state["last_reflect"] = {
        "timestamp": now,
        "bank_id": bank_id,
        "query": query,
        "output_path": str(output_path),
        "latest_path": str(latest_path),
        "content_hash": content_hash(rendered),
        "usage": dict(usage or {}),
    }
    return save_state(state, output_dir)


def _write_backflow_destination(
    *,
    rendered: str,
    destination: str,
    bank_id: str,
    query: str,
    output_path: str | Path,
) -> Path | None:
    metadata = {
        "source": "hindsight-reflect",
        "source_ref": str(output_path),
        "query": query,
        "tags": ["hindsight", "reflect", destination],
        "metadata": {"bank_id": bank_id, "query": query, "source_ref": str(output_path)},
    }
    normalized = str(destination or "atom").strip().lower()
    if normalized == "persona":
        return uplift_persona(rendered, metadata=metadata)
    if normalized == "scenario":
        return uplift_scenario(rendered, metadata=metadata)
    return ingest_recall_excerpt(
        rendered,
        query=query,
        source_ref=str(output_path),
        metadata=metadata,
    )


def backflow_reflect_report(
    *,
    bank_id: str,
    query: str,
    rendered: str,
    output_path: str | Path,
    latest_path: str | Path,
    output_dir: str | Path | None = None,
    destination: str = "atom",
    dry_run: bool = False,
    force: bool = False,
) -> ReflectBackflowResult:
    reports_dir = get_reports_dir(output_dir)
    state_path = reports_dir / STATE_FILENAME
    now = datetime.now(timezone.utc).isoformat()
    normalized_destination = str(destination or "atom").strip().lower()
    digest = content_hash(rendered)
    state = load_state(output_dir)
    last_backflow = state.get("last_backflow") if isinstance(state.get("last_backflow"), Mapping) else {}
    duplicate = (
        last_backflow.get("content_hash") == digest
        and last_backflow.get("destination") == normalized_destination
        and not bool(last_backflow.get("dry_run"))
    )
    if duplicate and not force:
        result = ReflectBackflowResult(
            ok=True,
            changed=False,
            skipped=True,
            reason="duplicate_hash",
            bank_id=bank_id,
            query=query,
            destination=normalized_destination,
            output_path=str(output_path),
            latest_path=str(latest_path),
            memory_path=str(last_backflow.get("memory_path") or ""),
            content_hash=digest,
            timestamp=now,
            state_path=str(state_path),
        )
        state = dict(state)
        state["last_backflow"] = {
            **dict(last_backflow),
            "timestamp": now,
            "bank_id": bank_id,
            "query": query,
            "output_path": str(output_path),
            "latest_path": str(latest_path),
            "destination": normalized_destination,
            "content_hash": digest,
            "reason": result.reason,
            "skipped": True,
            "changed": False,
            "dry_run": False,
            "force": force,
        }
        save_state(state, output_dir)
        return result

    if dry_run:
        result = ReflectBackflowResult(
            ok=True,
            changed=False,
            skipped=True,
            reason="dry_run",
            bank_id=bank_id,
            query=query,
            destination=normalized_destination,
            output_path=str(output_path),
            latest_path=str(latest_path),
            memory_path="",
            content_hash=digest,
            timestamp=now,
            state_path=str(state_path),
        )
        state = dict(state)
        state["last_backflow"] = {
            "timestamp": now,
            "bank_id": bank_id,
            "query": query,
            "output_path": str(output_path),
            "latest_path": str(latest_path),
            "destination": normalized_destination,
            "content_hash": digest,
            "memory_path": "",
            "reason": result.reason,
            "skipped": True,
            "changed": False,
            "dry_run": True,
            "force": force,
        }
        save_state(state, output_dir)
        return result

    memory_path = _write_backflow_destination(
        rendered=rendered,
        destination=normalized_destination,
        bank_id=bank_id,
        query=query,
        output_path=output_path,
    )
    result = ReflectBackflowResult(
        ok=memory_path is not None,
        changed=memory_path is not None,
        skipped=False,
        reason="backflow_written" if memory_path is not None else "write_failed",
        bank_id=bank_id,
        query=query,
        destination=normalized_destination,
        output_path=str(output_path),
        latest_path=str(latest_path),
        memory_path=str(memory_path or ""),
        content_hash=digest,
        timestamp=now,
        state_path=str(state_path),
    )
    state = dict(state)
    state["last_backflow"] = {
        "timestamp": now,
        "bank_id": bank_id,
        "query": query,
        "output_path": str(output_path),
        "latest_path": str(latest_path),
        "destination": normalized_destination,
        "content_hash": digest,
        "memory_path": str(memory_path or ""),
        "reason": result.reason,
        "skipped": False,
        "changed": bool(memory_path),
        "dry_run": False,
        "force": force,
    }
    save_state(state, output_dir)
    return result
