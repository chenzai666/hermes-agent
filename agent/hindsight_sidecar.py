"""Minimal Hindsight sidecar client for Hermes/OpenClaw shared learning.

This module is intentionally SIDE-CAR only:
- it does NOT replace the primary memory.provider
- it does NOT inject recall into prompts by default
- it focuses on high-value retain + optional reflect/report workflows

Goal: let Hermes keep using memtensor/builtin memory while mirroring a small,
carefully filtered subset of durable events into Hindsight for future shared
learning with OpenClaw.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

MEMORY_LAYER_ORDER = ("persona", "scenario", "atom")

from hermes_cli.config import cfg_get, read_raw_config

logger = logging.getLogger(__name__)


@dataclass
class HindsightSidecarConfig:
    enabled: bool = False
    api_url: str = "http://localhost:8888"
    api_key: str = ""
    timeout: int = 20
    retain_on_memory_write: bool = True
    retain_on_session_end: bool = True
    retain_on_delegation: bool = True
    enable_recall: bool = False
    enable_reflect: bool = True
    shared_bank_prefix: str = "shared"
    hermes_bank_template: str = "hermes:user:{user_id}"
    shared_user_bank_template: str = "shared:user:{user_id}"
    shared_ops_bank: str = "shared:ops"


def _cfg(path: str, default: Any) -> Any:
    try:
        value = cfg_get(path, default)
        if value is not None:
            return value
    except Exception:
        pass
    try:
        raw = read_raw_config()
        cur = raw
        for part in path.split('.'):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return default if cur is None else cur
    except Exception:
        return default


def load_sidecar_config() -> HindsightSidecarConfig:
    return HindsightSidecarConfig(
        enabled=bool(_cfg("hindsight_sidecar.enabled", False)),
        api_url=str(_cfg("hindsight_sidecar.api_url", "http://localhost:8888")).rstrip("/"),
        api_key=str(_cfg("hindsight_sidecar.api_key", "")),
        timeout=int(_cfg("hindsight_sidecar.timeout", 20) or 20),
        retain_on_memory_write=bool(_cfg("hindsight_sidecar.retain_on_memory_write", True)),
        retain_on_session_end=bool(_cfg("hindsight_sidecar.retain_on_session_end", True)),
        retain_on_delegation=bool(_cfg("hindsight_sidecar.retain_on_delegation", True)),
        enable_recall=bool(_cfg("hindsight_sidecar.enable_recall", False)),
        enable_reflect=bool(_cfg("hindsight_sidecar.enable_reflect", True)),
        shared_bank_prefix=str(_cfg("hindsight_sidecar.shared_bank_prefix", "shared")),
        hermes_bank_template=str(_cfg("hindsight_sidecar.hermes_bank_template", "hermes:user:{user_id}")),
        shared_user_bank_template=str(_cfg("hindsight_sidecar.shared_user_bank_template", "shared:user:{user_id}")),
        shared_ops_bank=str(_cfg("hindsight_sidecar.shared_ops_bank", "shared:ops")),
    )


def _sanitize(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


_SESSION_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-fA-F]{6,}$")
_CRON_SESSION_ID_RE = re.compile(r"^cron_[0-9a-fA-F]+_\d{8}_\d{6}$")


def _normalize_user_id(value: str, fallback: str = "agent-main") -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    lowered = raw.lower()
    if lowered in {"unknown", "none", "null", "anonymous"}:
        return fallback
    if raw == "agent:main":
        return "agent-main"
    # Hermes session/task ids are provenance, not stable user identities.
    # Never render them into shared user banks like shared:user:20260605_...
    if _SESSION_ID_RE.match(raw) or _CRON_SESSION_ID_RE.match(raw):
        return fallback
    sanitized = _sanitize(raw)
    if not sanitized or sanitized.lower() == "unknown":
        return fallback
    if _SESSION_ID_RE.match(sanitized) or _CRON_SESSION_ID_RE.match(sanitized):
        return fallback
    return sanitized


def _render(template: str, **kwargs: str) -> str:
    data = {
        k: (_normalize_user_id(v) if k == "user_id" else _sanitize(v))
        for k, v in kwargs.items()
    }
    try:
        rendered = template.format(**data)
    except Exception:
        rendered = template
    rendered = re.sub(r"-+", "-", rendered).strip("-")
    return rendered or "hermes"


def _normalize_memory_type(value: str) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in MEMORY_LAYER_ORDER else "atom"


def _guess_memory_type(content: str, *, target: str = "memory", context: str = "", tags: Optional[Iterable[str]] = None) -> str:
    haystack = "\n".join([
        str(content or ""),
        str(context or ""),
        " ".join(str(t) for t in (tags or []) if str(t).strip()),
        str(target or ""),
    ]).lower()
    if target == "user" or re.search(r"\b(prefer|always|never|default|habit|persona)\b|偏好|习惯|默认|记住我|我喜欢|我不喜欢", haystack):
        return "persona"
    if re.search(r"\b(summary|workflow|playbook|runbook|procedure|steps|resolved|fix|scenario)\b|复盘|经验|步骤|排查|修复|总结|方案", haystack):
        return "scenario"
    return "atom"


def _layer_section_title(layer: str) -> str:
    return {
        "persona": "### Persona memory",
        "scenario": "### Scenario memory",
        "atom": "### Atom memory",
    }.get(layer, "### Atom memory")


class HindsightSidecar:
    def __init__(self, config: Optional[HindsightSidecarConfig] = None):
        self.config = config or load_sidecar_config()

    @property
    def enabled(self) -> bool:
        raw = self.config.enabled
        if isinstance(raw, str):
            raw = raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw and self.config.api_url)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "api_url": self.config.api_url,
            "timeout": self.config.timeout,
            "retain_on_memory_write": self.config.retain_on_memory_write,
            "retain_on_session_end": self.config.retain_on_session_end,
            "retain_on_delegation": self.config.retain_on_delegation,
            "enable_recall": self.config.enable_recall,
            "enable_reflect": self.config.enable_reflect,
            "shared_bank_prefix": self.config.shared_bank_prefix,
            "hermes_bank_template": self.config.hermes_bank_template,
            "shared_user_bank_template": self.config.shared_user_bank_template,
            "shared_ops_bank": self.config.shared_ops_bank,
        }

    def healthcheck(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "sidecar disabled", **self.status()}
        try:
            url = f"{self.config.api_url}/version"
            req = urllib.request.Request(url, headers=self._headers(), method="GET")
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            if isinstance(data, dict):
                return {"ok": True, **self.status(), "version": data.get("version") or data.get("api_version") or "unknown", "raw": data}
            return {"ok": True, **self.status(), "raw": raw}
        except Exception as e:
            return {"ok": False, **self.status(), "error": str(e)}

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.api_url}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw) if raw else {"ok": True}
        except Exception:
            return {"raw": raw}

    def retain(
        self,
        *,
        bank_id: str,
        content: str,
        context: str = "",
        tags: Optional[Iterable[str]] = None,
        memory_type: str = "atom",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled or not content.strip():
            return {"skipped": True}
        normalized_type = _normalize_memory_type(memory_type)
        rendered_content = content
        if not re.search(r"^Memory-Type:\s*(persona|scenario|atom)\b", rendered_content, re.IGNORECASE):
            rendered_content = f"Memory-Type: {normalized_type}\n{rendered_content.strip()}"
        item: Dict[str, Any] = {
            "content": rendered_content,
            "document_id": f"hermes-{_sanitize(bank_id)}-{normalized_type}",
            "metadata": {
                "memory_type": normalized_type,
                **{k: v for k, v in (metadata or {}).items() if v is not None},
            },
        }
        if context:
            item["context"] = context
        if tags:
            item["tags"] = [str(t) for t in tags if str(t).strip()]
        payload = {
            "items": [item],
            "async": False,
        }
        path = f"/v1/default/banks/{bank_id}/memories"
        return self._post_json(path, payload)

    def reflect(self, *, bank_id: str, query: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"skipped": True}
        path = f"/v1/default/banks/{bank_id}/reflect"
        return self._post_json(path, {"query": query})

    def bank_for_user(self, user_id: str) -> str:
        return _render(self.config.hermes_bank_template, user_id=user_id or "unknown")

    def shared_user_bank(self, user_id: str) -> str:
        return _render(self.config.shared_user_bank_template, user_id=user_id or "unknown")

    def recall(self, *, bank_id: str, query: str, top_k: int = 6) -> Dict[str, Any]:
        if not self.enabled:
            return {"success": False, "reason": "disabled", "results": []}
        q = str(query or "").strip()
        if not q:
            return {"success": False, "reason": "empty_query", "results": []}
        # Hindsight API 0.6.x does not accept a top_k request field; it logs
        # "Unknown parameters ignored: [top_k]" when clients send it. Keep the
        # Python argument for local post-filtering/backwards compatibility, but
        # only send supported API fields over the wire.
        raw = self._post_json(
            f"/v1/default/banks/{bank_id}/memories/recall",
            {"query": q},
        )
        results = []
        for item in raw.get("results", raw.get("data", {}).get("results", [])) or []:
            memory_type = _normalize_memory_type(
                (item.get("metadata") or {}).get("memory_type")
                or item.get("memory_type")
                or (re.search(r"^Memory-Type:\s*(persona|scenario|atom)\b", str(item.get("text") or item.get("content") or ""), re.IGNORECASE) or [None, "atom"])[1]
            )
            results.append({**item, "memory_type": memory_type})
        results.sort(key=lambda item: MEMORY_LAYER_ORDER.index(item.get("memory_type", "atom")))
        return {"success": True, "results": results, "raw": raw}

    def recall_many(self, *, bank_ids: Iterable[str], query: str, top_k: int = 6) -> Dict[str, Any]:
        unique_bank_ids: List[str] = []
        seen_banks = set()
        for bank_id in bank_ids or []:
            normalized = str(bank_id or "").strip()
            if not normalized or normalized in seen_banks:
                continue
            seen_banks.add(normalized)
            unique_bank_ids.append(normalized)
        if not unique_bank_ids:
            return {"success": False, "reason": "empty_bank_ids", "results": []}

        combined: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        seen_entries = set()
        for bank_id in unique_bank_ids:
            try:
                result = self.recall(bank_id=bank_id, query=query, top_k=top_k)
            except Exception as exc:
                errors.append({"bank_id": bank_id, "error": str(exc)})
                continue
            if not result.get("success"):
                errors.append({"bank_id": bank_id, "error": str(result.get("reason") or "recall_failed")})
                continue
            for item in result.get("results", []):
                text = str(item.get("text") or item.get("content") or "").strip()
                metadata_blob = json.dumps(item.get("metadata") or {}, ensure_ascii=False, sort_keys=True)
                dedupe_key = (
                    item.get("memory_type", "atom"),
                    text,
                    metadata_blob,
                )
                if dedupe_key in seen_entries:
                    continue
                seen_entries.add(dedupe_key)
                combined.append({**item, "bank_id": bank_id})

        combined.sort(
            key=lambda item: (
                MEMORY_LAYER_ORDER.index(item.get("memory_type", "atom")),
                1 if str(item.get("bank_id") or "") == self.config.shared_ops_bank else 0,
                str(item.get("bank_id") or ""),
                str(item.get("text") or item.get("content") or ""),
            )
        )
        # Keep ops memories as a fallback layer instead of letting them dominate.
        non_ops = [item for item in combined if str(item.get("bank_id") or "") != self.config.shared_ops_bank]
        ops = [item for item in combined if str(item.get("bank_id") or "") == self.config.shared_ops_bank]
        if non_ops:
            combined = non_ops[:6] + ops[:2]
        else:
            combined = ops[:4]
        return {
            "success": bool(combined),
            "results": combined,
            "banks": unique_bank_ids,
            "errors": errors,
        }

    def format_layered_recall(self, results: List[Dict[str, Any]]) -> str:
        grouped: Dict[str, List[Dict[str, Any]]] = {layer: [] for layer in MEMORY_LAYER_ORDER}
        for item in results or []:
            grouped[_normalize_memory_type(item.get("memory_type", "atom"))].append(item)
        sections: List[str] = []
        for layer in MEMORY_LAYER_ORDER:
            items = grouped[layer]
            if not items:
                continue
            body = []
            for idx, item in enumerate(items, 1):
                text = str(item.get("text") or item.get("content") or "").strip()
                bank_id = str(item.get("bank_id") or "").strip()
                prefix = f"[{bank_id}] " if bank_id else ""
                body.append(f"{idx}. {prefix}{text[:500]}")
            sections.append(f"{_layer_section_title(layer)}\n" + "\n\n".join(body))
        if not sections:
            return ""
        return "## Layered Hindsight recall\n\n" + "\n\n".join(sections)


def should_retain_memory_write(action: str, target: str, content: str) -> bool:
    if not content or not content.strip():
        return False
    if action not in {"add", "replace"}:
        return False
    if target not in {"user", "memory"}:
        return False
    return True


def summarize_session_for_retain(messages: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown"))
        content = str(msg.get("content", "") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        parts.append(f"[{role}] {content}")
        if sum(len(p) for p in parts) > max_chars:
            break
    text = "\n".join(parts)
    return text[:max_chars]
