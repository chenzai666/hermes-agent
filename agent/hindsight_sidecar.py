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

import hashlib
import json
import logging
import re
import time
import urllib.request
from dataclasses import dataclass, field
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
    near_dup_enabled: bool = True
    near_dup_window_days: int = 30
    near_dup_jaccard: float = 0.78
    near_dup_recall_limit: int = 8


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
        near_dup_enabled=bool(_cfg("hindsight_sidecar.near_dup_enabled", True)),
        near_dup_window_days=int(_cfg("hindsight_sidecar.near_dup_window_days", 30) or 30),
        near_dup_jaccard=float(_cfg("hindsight_sidecar.near_dup_jaccard", 0.78) or 0.78),
        near_dup_recall_limit=int(_cfg("hindsight_sidecar.near_dup_recall_limit", 8) or 8),
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
        self._recent_fp_cache: Dict[str, float] = {}
        self._recent_text_cache: Dict[str, List[Dict[str, Any]]] = {}

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
            "near_dup_enabled": self.config.near_dup_enabled,
            "near_dup_window_days": self.config.near_dup_window_days,
            "near_dup_jaccard": self.config.near_dup_jaccard,
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

    @staticmethod
    def _normalize_for_fingerprint(text: str) -> str:
        s = str(text or "").lower()
        s = re.sub(r"source:\s*openclaw[^\n]*", " ", s, flags=re.I)
        s = re.sub(r"owner:\s*[^\n]*|session:\s*[^\n]*|task id:\s*[^\n]*|turn id:\s*[^\n]*|chunk id:\s*[^\n]*|role:\s*[^\n]*", " ", s, flags=re.I)
        s = re.sub(r"\|?\s*when:\s*[^|]+", " ", s, flags=re.I)
        s = re.sub(r"\|?\s*involving:\s*[^|]+", " ", s, flags=re.I)
        s = re.sub(r"\bpid\s*[:=]?\s*\d+\b", "pid", s, flags=re.I)
        s = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "date", s)
        s = re.sub(r"\b\d+(?:\.\d+)?\s*(?:gb|mb|kb|%)\b", "n", s, flags=re.I)
        s = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "ip", s)
        s = re.sub(r"https?://\S+", "url", s, flags=re.I)
        if re.search(r"gateway", s) and re.search(r"内存|rss|swap|物理空闲", s):
            s = "topic gateway memory pressure " + s
        if re.search(r"gateway", s) and re.search(r"重启|restart", s) and re.search(r"确认|中断", s):
            s = "topic gateway restart confirm " + s
        s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", s)
        s = re.sub(r"([\u4e00-\u9fff])", r" \1 ", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def _token_set(cls, text: str) -> set[str]:
        raw = [t for t in cls._normalize_for_fingerprint(text).split(" ") if t]
        stop = {"的", "了", "和", "与", "及", "并", "在", "是", "为", "仍", "高", "仅", "约", "需", "但", "可", "n", "0", "23"}
        out: set[str] = set()
        for i, t in enumerate(raw):
            if not t or t in stop:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]", t):
                out.add(t)
                if i + 1 < len(raw) and re.fullmatch(r"[\u4e00-\u9fff]", raw[i + 1]):
                    out.add(t + raw[i + 1])
            elif len(t) >= 2:
                out.add(t)
        for t in raw:
            if t.startswith("topic") or t in {"gateway", "memory", "pressure"}:
                out.add(t)
        return out

    @classmethod
    def _fingerprint(cls, text: str) -> str:
        tokens = sorted(cls._token_set(text))
        if not tokens:
            return ""
        return hashlib.sha1(" ".join(tokens).encode("utf-8")).hexdigest()[:24]

    @classmethod
    def _jaccard(cls, a: str, b: str) -> float:
        sa = cls._token_set(a)
        sb = cls._token_set(b)
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    def _cache_get(self, bank_id: str, fp: str) -> bool:
        key = f"{bank_id}:{fp}"
        ts = self._recent_fp_cache.get(key)
        if not ts:
            return False
        if time.time() - ts > 6 * 3600:
            self._recent_fp_cache.pop(key, None)
            return False
        return True

    def _cache_put(self, bank_id: str, fp: str) -> None:
        if fp:
            self._recent_fp_cache[f"{bank_id}:{fp}"] = time.time()

    def _remember_text(self, bank_id: str, content: str) -> None:
        body = str(content or "").strip()
        if not body:
            return
        now = time.time()
        arr = [x for x in self._recent_text_cache.get(bank_id, []) if now - float(x.get("ts", 0)) <= 6 * 3600]
        arr.insert(0, {"text": body, "ts": now})
        self._recent_text_cache[bank_id] = arr[:200]
        self._cache_put(bank_id, self._fingerprint(body))

    def _local_text_duplicate(self, bank_id: str, content: str) -> Optional[Dict[str, Any]]:
        body = str(content or "").strip()
        if not body:
            return None
        now = time.time()
        arr = [x for x in self._recent_text_cache.get(bank_id, []) if now - float(x.get("ts", 0)) <= 6 * 3600]
        self._recent_text_cache[bank_id] = arr
        for item in arr:
            text = str(item.get("text") or "")
            if self._fingerprint(text) == self._fingerprint(body):
                return {"duplicate": True, "reason": "local_text_fp", "fingerprint": self._fingerprint(body), "matched": text[:120]}
            sim = self._jaccard(body, text)
            if sim >= float(self.config.near_dup_jaccard):
                return {
                    "duplicate": True,
                    "reason": "local_text_jaccard",
                    "fingerprint": self._fingerprint(body),
                    "similarity": round(sim, 3),
                    "matched": text[:120],
                }
        return None

    def _is_near_duplicate(self, bank_id: str, content: str) -> Dict[str, Any]:
        if not self.config.near_dup_enabled:
            return {"duplicate": False}
        body = str(content or "").strip()
        if not body:
            return {"duplicate": False}
        fp = self._fingerprint(body)
        if not fp:
            return {"duplicate": False}
        if self._cache_get(bank_id, fp):
            return {"duplicate": True, "reason": "local_fp_cache", "fingerprint": fp}
        local = self._local_text_duplicate(bank_id, body)
        if local and local.get("duplicate"):
            self._cache_put(bank_id, fp)
            return local
        query = self._normalize_for_fingerprint(body)[:240] or body[:240]
        try:
            raw = self._post_json(f"/v1/default/banks/{bank_id}/memories/recall", {"query": query})
        except Exception as exc:
            logger.warning("[hindsight-sidecar] near-dup recall failed bank=%s: %s", bank_id, exc)
            return {"duplicate": False, "reason": "checker_error"}
        items = raw.get("results") or (raw.get("data") or {}).get("results") or []
        window_s = max(1, int(self.config.near_dup_window_days)) * 86400
        now = time.time()
        checked = 0
        for item in items:
            if checked >= int(self.config.near_dup_recall_limit):
                break
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("content") or item.get("summary") or "").strip()
            if not text:
                continue
            checked += 1
            ts_raw = item.get("event_date") or item.get("created_at") or item.get("mentioned_at")
            if ts_raw:
                try:
                    if isinstance(ts_raw, (int, float)):
                        ts = float(ts_raw)
                        if ts > 1e12:
                            ts /= 1000.0
                    else:
                        from datetime import datetime
                        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
                    if now - ts > window_s:
                        continue
                except Exception:
                    pass
            existing_fp = self._fingerprint(text)
            if existing_fp and existing_fp == fp:
                self._cache_put(bank_id, fp)
                return {"duplicate": True, "reason": "fingerprint_match", "fingerprint": fp, "matched": text[:120]}
            sim = self._jaccard(body, text)
            if sim >= float(self.config.near_dup_jaccard):
                self._cache_put(bank_id, fp)
                return {
                    "duplicate": True,
                    "reason": "jaccard_match",
                    "fingerprint": fp,
                    "similarity": round(sim, 3),
                    "matched": text[:120],
                }
        return {"duplicate": False, "fingerprint": fp, "checked": checked}

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
        # Runtime diagnostic noise should not become durable shared memory.
        if re.search(
            r"Gateway\s*内存|RSS\s*[:=]?\s*\d|Swap\s*已用|物理空闲|Auto-recall|memory_search|LimitNOFILE|response_store",
            content,
            re.I,
        ):
            return {"skipped": True, "reason": "runtime_noise"}
        dup = self._is_near_duplicate(bank_id, content)
        if dup.get("duplicate"):
            logger.info(
                "[hindsight-sidecar] near-dup suppressed bank=%s reason=%s sim=%s fp=%s",
                bank_id,
                dup.get("reason"),
                dup.get("similarity"),
                dup.get("fingerprint"),
            )
            return {"skipped": True, "reason": "near_duplicate", "detail": dup}
        # Remember before remote write so bursty near-dups collapse even if recall lags.
        self._remember_text(bank_id, content)
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
        result = self._post_json(path, payload)
        self._remember_text(bank_id, content)
        return result

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
