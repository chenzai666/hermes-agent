"""Minimal structured memory-layer helpers.

This module provides a tiny schema and loader for profile-scoped memory
layers while keeping the final assembled recall text compatible with the
existing plain-text injection flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from typing import Any, Iterable, List, Mapping, Sequence

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
_SAFE_STEM_RE = re.compile(r"[^a-z0-9._-]+")
_WHITESPACE_RE = re.compile(r"\s+")
_PERSONA_FACT_RE = re.compile(r"^\s*(?:[-*]\s+)?(?P<key>[^:：\n]{2,80})\s*[:：]\s*(?P<value>.+?)\s*$")
_SESSION_HINT_RE = re.compile(
    r"\b(prefer|preference|always|never|don't|do not|remember|note|rule|must|avoid|likes?|dislikes?)\b|偏好|规则|总是|不要|记住|喜欢|不喜欢",
    re.IGNORECASE,
)
_RECALL_HINT_RE = re.compile(
    r"recall\s+summary|memory\s+summary|recalled\s+memory|retrieved\s+memory|summary:|摘要|回忆|记忆总结",
    re.IGNORECASE,
)
_SCENARIO_HINT_RE = re.compile(
    r"\b(project|workflow|context|current|currently|working on|goal|constraint|environment|setup|task)\b|项目|场景|上下文|正在|目标|约束|环境",
    re.IGNORECASE,
)
_SOURCE_PRIORITY = {
    "curated-uplift": 50,
    "session": 40,
    "builtin-memory-write": 35,
    "recall": 25,
    "": 0,
}


@dataclass(frozen=True)
class LayerEntry:
    """Structured representation of one memory layer section."""

    key: str
    label: str
    content: str
    source: str = ""
    id: str = ""
    type: str = "memory-layer"
    source_ref: str = ""
    derived_from: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    structure: Mapping[str, Any] = field(default_factory=dict)
    relations: Sequence[Any] = field(default_factory=tuple)

    def render(self) -> str:
        """Render this layer using the legacy plain-text section format."""
        parts: List[str] = []
        body = self.content.strip()
        if body:
            parts.append(body)

        structure_text = _render_structure(self.structure)
        if structure_text:
            parts.append("Structure:\n" + structure_text)

        relations_text = _render_relations(self.relations)
        if relations_text:
            parts.append("Relations:\n" + relations_text)

        return f"[{self.label}]\n" + "\n\n".join(parts).strip()


def get_memory_layers_dir(*, base_dir: Path | None = None) -> Path:
    """Return the profile-scoped memory-layers directory."""
    return base_dir if base_dir is not None else (get_hermes_home() / "memory_layers")


def build_layered_memory_context(provider_recall: str = "", *, base_dir: Path | None = None) -> str:
    """Compose lightweight layered memory context plus optional provider recall."""
    root = get_memory_layers_dir(base_dir=base_dir)
    sections: list[str] = []

    persona = _read_layer_file(root / "persona.md")
    if persona:
        sections.append(f"[Persona]\n{persona}")

    scenario = _read_layer_file(root / "scenario.md")
    if scenario:
        sections.append(f"[Scenario]\n{scenario}")

    atom_parts: list[str] = []
    atom_single = _read_layer_file(root / "atom.md")
    if atom_single:
        atom_parts.append(atom_single)
    atoms_dir = root / "atoms"
    if atoms_dir.exists() and atoms_dir.is_dir():
        for atom_path in sorted(atoms_dir.glob("*.md")):
            atom_body, _, _ = _parse_layer_file(atom_path)
            if atom_body:
                atom_parts.append(f"{atom_path.stem}: {atom_body}")
    if atom_parts:
        sections.append("[Atoms]\n" + "\n\n".join(atom_parts))

    provider_recall = (provider_recall or "").strip()
    if provider_recall:
        if provider_recall.startswith("[Provider Recall]") or provider_recall.startswith("[Hindsight Sidecar Recall"):
            sections.append(provider_recall)
        else:
            sections.append(f"[Provider Recall]\n{provider_recall}")

    return "\n\n".join(section for section in sections if section.strip())


def _read_layer_file(path: Path) -> str:
    """Read a lightweight memory-layer file if it exists."""
    try:
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.debug("Failed reading memory layer file %s: %s", path, e)
        return ""


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse optional YAML-style frontmatter from a markdown file.

    Returns ``(frontmatter_dict, body_text)``. Files without frontmatter remain
    fully backwards compatible and are returned unchanged as body text.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text

    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break
    if end_index is None:
        return {}, text

    frontmatter_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).strip()
    loaded = _load_yaml_like_mapping(frontmatter_text)
    if not loaded:
        return {}, text
    return loaded, body


def _load_yaml_like_mapping(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        pass

    return _minimal_yaml_mapping(text)


def _minimal_yaml_mapping(text: str) -> dict[str, Any]:
    """Very small YAML subset parser used as a dependency-free fallback.

    Supported:
    - ``key: value`` scalars
    - nested mappings via indentation
    - lists using ``- item`` or ``- key: value``

    It is intentionally tiny, but sufficient for the structured memory-layer
    frontmatter used by tests and local curated files.
    """

    def parse_scalar(raw: str) -> Any:
        value = raw.strip()
        if value == "":
            return ""
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value

    def split_key_value(raw: str) -> tuple[str, str] | None:
        if ":" not in raw:
            return None
        key, value = raw.split(":", 1)
        key = key.strip()
        if not key:
            return None
        return key, value.strip()

    entries: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        entries.append((indent, raw_line.strip()))

    def parse_block(index: int, current_indent: int) -> tuple[Any, int]:
        container: Any = None

        while index < len(entries):
            indent, line = entries[index]
            if indent < current_indent:
                break
            if indent > current_indent:
                index += 1
                continue

            if line.startswith("- "):
                if container is None:
                    container = []
                if not isinstance(container, list):
                    break

                item_text = line[2:].strip()
                index += 1
                next_indent = entries[index][0] if index < len(entries) else -1
                maybe_pair = split_key_value(item_text)

                if maybe_pair:
                    item_key, item_value = maybe_pair
                    item: dict[str, Any] = {}
                    if item_value:
                        item[item_key] = parse_scalar(item_value)
                    elif next_indent > current_indent:
                        nested, index = parse_block(index, next_indent)
                        item[item_key] = nested
                    else:
                        item[item_key] = {}
                    if index < len(entries):
                        lookahead_indent = entries[index][0]
                        if lookahead_indent > current_indent:
                            nested_tail, index = parse_block(index, lookahead_indent)
                            if isinstance(item[item_key], dict) and isinstance(nested_tail, dict):
                                item[item_key].update(nested_tail)
                            elif isinstance(nested_tail, dict):
                                item.update(nested_tail)
                    container.append(item)
                elif next_indent > current_indent:
                    nested, index = parse_block(index, next_indent)
                    container.append(nested)
                else:
                    container.append(parse_scalar(item_text))
                continue

            maybe_pair = split_key_value(line)
            if maybe_pair is None:
                index += 1
                continue

            if container is None:
                container = {}
            if not isinstance(container, dict):
                break

            key, value = maybe_pair
            index += 1
            next_indent = entries[index][0] if index < len(entries) else -1
            if value:
                container[key] = parse_scalar(value)
            elif next_indent > current_indent:
                nested, index = parse_block(index, next_indent)
                container[key] = nested
            else:
                container[key] = {}

        if container is None:
            container = {}
        return container, index

    parsed, _ = parse_block(0, entries[0][0] if entries else 0)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _render_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render_structure(data: Mapping[str, Any] | None, *, indent: int = 0) -> str:
    if not data:
        return ""
    lines: List[str] = []
    pad = "  " * indent
    for key, value in data.items():
        if isinstance(value, Mapping):
            nested = _render_structure(value, indent=indent + 1)
            if nested:
                lines.append(f"{pad}- {key}:")
                lines.append(nested)
            else:
                lines.append(f"{pad}- {key}: {{}}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            lines.append(f"{pad}- {key}:")
            for item in value:
                lines.extend(_render_sequence_item(item, indent=indent + 1))
        else:
            lines.append(f"{pad}- {key}: {_render_scalar(value)}")
    return "\n".join(lines)


def _render_sequence_item(value: Any, *, indent: int = 0) -> List[str]:
    pad = "  " * indent
    if isinstance(value, Mapping):
        nested = _render_structure(value, indent=indent + 1)
        if not nested:
            return [f"{pad}- {{}}"]
        first_key = next(iter(value.keys()), None)
        if first_key is not None and len(value) == 1 and not isinstance(value[first_key], (Mapping, list, tuple)):
            return [f"{pad}- {first_key}: {_render_scalar(value[first_key])}"]
        return [f"{pad}-"] + [nested]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        lines = [f"{pad}-"]
        for item in value:
            lines.extend(_render_sequence_item(item, indent=indent + 1))
        return lines
    return [f"{pad}- {_render_scalar(value)}"]


def _render_relations(relations: Sequence[Any] | None) -> str:
    if not relations:
        return ""
    lines: List[str] = []
    for relation in relations:
        if isinstance(relation, Mapping):
            relation_type = relation.get("type") or relation.get("kind") or relation.get("relation")
            target = relation.get("target") or relation.get("to") or relation.get("id") or relation.get("ref")
            note = relation.get("label") or relation.get("summary") or relation.get("note")
            if relation_type and target and note:
                lines.append(f"- {relation_type}: {target} ({note})")
            elif relation_type and target:
                lines.append(f"- {relation_type}: {target}")
            elif note:
                lines.append(f"- {note}")
            else:
                pairs = ", ".join(f"{k}={_render_scalar(v)}" for k, v in relation.items())
                if pairs:
                    lines.append(f"- {pairs}")
        else:
            lines.append(f"- {_render_scalar(relation)}")
    return "\n".join(lines)


def _safe_stem(value: str) -> str:
    normalized = _SAFE_STEM_RE.sub("-", value.strip().lower()).strip("-._")
    return normalized or "memory"


def _normalized_excerpt_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip().lower())


def canonicalize_atom_text(text: str) -> str:
    """Return a tiny canonical form for atom dedupe and conflict detection."""
    body = (text or "").strip().lower()
    if not body:
        return ""
    body = re.sub(r"^[\-*>\d.\)\s]+", "", body, flags=re.MULTILINE)
    body = body.replace("：", ":")
    body = re.sub(r"\s*:\s*", ": ", body)
    return _WHITESPACE_RE.sub(" ", body)


def _timestamp_from_record(record: Mapping[str, Any]) -> str:
    created_at = str(record.get("created_at") or "").strip()
    if created_at:
        return created_at
    path = record.get("path")
    if isinstance(path, Path):
        return path.stem.split("-", 1)[0]
    return ""


def _source_priority(source: str) -> int:
    return _SOURCE_PRIORITY.get((source or "").strip().lower(), 10)


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, int]:
    return (
        _timestamp_from_record(record),
        _source_priority(str(record.get("source") or "")),
    )


def _unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return tuple(items)


def _merge_record_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for record in sorted(records, key=_record_sort_key):
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            merged.update(dict(metadata))
    return merged


def dedupe_atoms(atom_records: Sequence[Mapping[str, Any]]) -> List[dict[str, Any]]:
    """Merge duplicate atoms by canonical text, preferring stronger/newer sources."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    ordered_keys: list[str] = []
    for record in atom_records:
        canonical = canonicalize_atom_text(str(record.get("body") or ""))
        if not canonical:
            continue
        if canonical not in grouped:
            grouped[canonical] = []
            ordered_keys.append(canonical)
        grouped[canonical].append(record)

    merged_records: list[dict[str, Any]] = []
    for canonical in ordered_keys:
        group = grouped[canonical]
        best = max(group, key=_record_sort_key)
        merged: dict[str, Any] = dict(best)
        merged["canonical_body"] = canonical
        merged["aliases"] = tuple(
            dict.fromkeys(str(item.get("body") or "").strip() for item in group if str(item.get("body") or "").strip())
        )
        merged["tags"] = _unique_strings(
            tag
            for item in group
            for tag in _normalize_string_sequence(item.get("tags"))
        )
        merged["derived_from"] = _unique_strings(
            ref
            for item in group
            for ref in _normalize_string_sequence(item.get("derived_from"))
        )
        merged["metadata"] = _merge_record_metadata(group)
        if not merged.get("source_ref"):
            merged["source_ref"] = next((str(item.get("source_ref") or "") for item in reversed(group) if item.get("source_ref")), "")
        if not merged.get("query"):
            merged["query"] = next((str(item.get("query") or "") for item in reversed(group) if item.get("query")), "")
        merged_records.append(merged)
    return merged_records


def _iter_existing_atom_bodies(*, base_dir: Path | None = None) -> Iterable[str]:
    root = get_memory_layers_dir(base_dir=base_dir)
    atom_single_path = root / "atom.md"
    atom_single, _, _ = _parse_layer_file(atom_single_path)
    if atom_single:
        yield atom_single

    atoms_dir = root / "atoms"
    if not atoms_dir.exists() or not atoms_dir.is_dir():
        return

    for atom_path in sorted(atoms_dir.glob("*.md")):
        atom_body, _, _ = _parse_layer_file(atom_path)
        if atom_body:
            yield atom_body


def _is_duplicate_atom_content(content: str, *, base_dir: Path | None = None) -> bool:
    normalized = canonicalize_atom_text(content)
    if not normalized:
        return True
    for existing in _iter_existing_atom_bodies(base_dir=base_dir):
        if canonicalize_atom_text(existing) == normalized:
            return True
    return False


def _looks_like_meaningful_excerpt(content: str, *, min_chars: int, max_chars: int) -> bool:
    body = (content or "").strip()
    if len(body) < min_chars or len(body) > max_chars:
        return False
    if len(body.split()) < 6 and len(body) < (min_chars + 20):
        return False
    return True


def should_ingest_session_excerpt(
    content: str,
    *,
    source_type: str = "",
    base_dir: Path | None = None,
) -> bool:
    """Return True for short, preference/rule-like session snippets worth atomizing."""
    if not _looks_like_meaningful_excerpt(content, min_chars=60, max_chars=1200):
        return False
    lowered_source = (source_type or "").strip().lower()
    if "recall" in lowered_source:
        return False
    if not _SESSION_HINT_RE.search(content or ""):
        return False
    if _is_duplicate_atom_content(content, base_dir=base_dir):
        return False
    return True


def should_ingest_recall_excerpt(
    content: str,
    *,
    source_type: str = "",
    base_dir: Path | None = None,
) -> bool:
    """Return True for concise recall-summary snippets worth preserving as atoms."""
    if not _looks_like_meaningful_excerpt(content, min_chars=80, max_chars=1800):
        return False
    haystack = f"{source_type}\n{content}"
    if not _RECALL_HINT_RE.search(haystack):
        return False
    if _is_duplicate_atom_content(content, base_dir=base_dir):
        return False
    return True


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "\n", '"', "'", "[", "]", "{", "}"]):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _dump_yaml_like(value: Any, *, indent: int = 0) -> List[str]:
    pad = "  " * indent
    if isinstance(value, Mapping):
        lines: List[str] = []
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, Mapping):
                lines.append(f"{pad}{key_text}:")
                nested = _dump_yaml_like(item, indent=indent + 1)
                lines.extend(nested)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                lines.append(f"{pad}{key_text}:")
                nested = _dump_yaml_like(list(item), indent=indent + 1)
                lines.extend(nested or [f"{'  ' * (indent + 1)}-"])
            else:
                lines.append(f"{pad}{key_text}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        lines = []
        for item in value:
            if isinstance(item, Mapping):
                lines.append(f"{pad}-")
                lines.extend(_dump_yaml_like(item, indent=indent + 1))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                lines.append(f"{pad}-")
                lines.extend(_dump_yaml_like(list(item), indent=indent + 1))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    return [f"{pad}{_yaml_scalar(value)}"]


def _compose_frontmatter(data: Mapping[str, Any]) -> str:
    lines = _dump_yaml_like(data)
    if not lines:
        return ""
    return "---\n" + "\n".join(lines) + "\n---\n"


def write_memory_atom(
    *,
    content: str,
    target: str,
    action: str = "add",
    base_dir: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist one built-in memory write as an atom markdown file."""
    body = (content or "").strip()
    if not body:
        return None

    root = get_memory_layers_dir(base_dir=base_dir)
    atoms_dir = root / "atoms"
    atoms_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(metadata or {})
    canonical_body = canonicalize_atom_text(body)
    if not canonical_body:
        return None

    existing_records = dedupe_atoms(_load_atom_records(base_dir=base_dir))
    for record in existing_records:
        if str(record.get("canonical_body") or "") != canonical_body:
            continue
        existing_path = record.get("path")
        if not isinstance(existing_path, Path):
            return None
        existing_frontmatter, existing_body = _parse_frontmatter(_read_layer_file(existing_path))
        merged_tags = _unique_strings([
            *_normalize_string_sequence(existing_frontmatter.get("tags")),
            "memory",
            str(target).strip(),
            str(action).strip(),
            *_normalize_string_sequence(meta.get("tags")),
        ])
        merged_frontmatter = dict(existing_frontmatter)
        merged_frontmatter.update(
            {
                "target": existing_frontmatter.get("target") or target,
                "action": existing_frontmatter.get("action") or action,
                "source": existing_frontmatter.get("source") or meta.get("source") or "builtin-memory-write",
                "tags": merged_tags,
            }
        )
        for optional_key in ("session_id", "source_ref", "query", "write_origin", "user_id"):
            if not merged_frontmatter.get(optional_key) and meta.get(optional_key):
                merged_frontmatter[optional_key] = str(meta[optional_key])
        existing_meta = existing_frontmatter.get("metadata")
        merged_payload = dict(existing_meta) if isinstance(existing_meta, Mapping) else {}
        if isinstance(meta.get("metadata"), Mapping):
            merged_payload.update(dict(meta["metadata"]))
        if merged_payload:
            merged_frontmatter["metadata"] = merged_payload
        existing_path.write_text(
            _compose_frontmatter(merged_frontmatter) + ((existing_body or body).strip()) + "\n",
            encoding="utf-8",
        )
        return existing_path

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    stem_hint = meta.get("atom_name") or meta.get("slug") or body.splitlines()[0][:48]
    safe_stem = _safe_stem(str(stem_hint))
    atom_path = atoms_dir / f"{timestamp}-{safe_stem}.md"

    frontmatter: dict[str, Any] = {
        "id": f"atom-{timestamp}-{safe_stem}",
        "type": "memory-atom",
        "target": target,
        "action": action,
        "source": meta.get("source") or "builtin-memory-write",
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "tags": tuple(
            dict.fromkeys(
                [
                    "memory",
                    str(target).strip(),
                    str(action).strip(),
                    *_normalize_string_sequence(meta.get("tags")),
                ]
            )
        ),
    }
    if meta.get("session_id"):
        frontmatter["session_id"] = str(meta["session_id"])
    if meta.get("source_ref"):
        frontmatter["source_ref"] = str(meta["source_ref"])
    if meta.get("query"):
        frontmatter["query"] = str(meta["query"])
    if meta.get("write_origin"):
        frontmatter["write_origin"] = str(meta["write_origin"])
    if meta.get("user_id"):
        frontmatter["user_id"] = str(meta["user_id"])
    if meta.get("metadata") and isinstance(meta.get("metadata"), Mapping):
        frontmatter["metadata"] = dict(meta["metadata"])

    atom_path.write_text(_compose_frontmatter(frontmatter) + body.rstrip() + "\n", encoding="utf-8")
    return atom_path


def ingest_session_excerpt(
    content: str,
    *,
    session_id: str,
    source_ref: str = "",
    base_dir: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist a session-derived excerpt into the atoms layer."""
    if not (content or "").strip():
        return None
    meta = dict(metadata or {})
    meta.setdefault("source", "session")
    meta.setdefault("session_id", session_id)
    if source_ref:
        meta.setdefault("source_ref", source_ref)

    tags = ["session", "excerpt", *_normalize_string_sequence(meta.get("tags"))]
    meta["tags"] = tuple(dict.fromkeys(tags))
    payload = dict(meta.get("metadata") or {}) if isinstance(meta.get("metadata") or {}, Mapping) else {}
    payload.setdefault("session_id", session_id)
    if source_ref:
        payload.setdefault("source_ref", source_ref)
    meta["metadata"] = payload
    meta.setdefault("atom_name", meta.get("slug") or f"session-{session_id or 'excerpt'}")

    return write_memory_atom(
        content=content,
        target="session",
        action="ingest",
        base_dir=base_dir,
        metadata=meta,
    )



def ingest_recall_excerpt(
    content: str,
    *,
    query: str,
    session_id: str = "",
    source_ref: str = "",
    base_dir: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist a recall-derived excerpt into the atoms layer."""
    if not (content or "").strip():
        return None
    meta = dict(metadata or {})
    meta.setdefault("source", "recall")
    meta.setdefault("query", query)
    if session_id:
        meta.setdefault("session_id", session_id)
    if source_ref:
        meta.setdefault("source_ref", source_ref)

    tags = ["recall", "excerpt", *_normalize_string_sequence(meta.get("tags"))]
    meta["tags"] = tuple(dict.fromkeys(tags))
    payload = dict(meta.get("metadata") or {}) if isinstance(meta.get("metadata") or {}, Mapping) else {}
    payload.setdefault("query", query)
    if session_id:
        payload.setdefault("session_id", session_id)
    if source_ref:
        payload.setdefault("source_ref", source_ref)
    meta["metadata"] = payload
    meta.setdefault("atom_name", meta.get("slug") or f"recall-{query[:48] or 'excerpt'}")

    return write_memory_atom(
        content=content,
        target="recall",
        action="ingest",
        base_dir=base_dir,
        metadata=meta,
    )



def uplift_layer_file(
    layer_name: str,
    content: str,
    *,
    base_dir: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Lightweight helper to persist curated persona/scenario layers."""
    body = (content or "").strip()
    if not body:
        return None

    normalized = layer_name.strip().lower()
    if normalized not in {"persona", "scenario"}:
        raise ValueError(f"Unsupported layer uplift target: {layer_name}")

    meta = dict(metadata or {})
    root = get_memory_layers_dir(base_dir=base_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{normalized}.md"

    frontmatter: dict[str, Any] = {
        "id": str(meta.get("id") or normalized),
        "type": normalized,
        "source": str(meta.get("source") or "curated-uplift"),
    }
    if meta.get("source_ref"):
        frontmatter["source_ref"] = str(meta["source_ref"])
    if meta.get("derived_from"):
        frontmatter["derived_from"] = _normalize_string_sequence(meta.get("derived_from"))
    if meta.get("tags"):
        frontmatter["tags"] = _normalize_string_sequence(meta.get("tags"))
    if meta.get("metadata") and isinstance(meta.get("metadata"), Mapping):
        frontmatter["metadata"] = dict(meta["metadata"])

    path.write_text(_compose_frontmatter(frontmatter) + body + "\n", encoding="utf-8")
    return path


def uplift_persona(
    content: str,
    *,
    base_dir: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist persona.md with minimal structured frontmatter."""
    return uplift_layer_file("persona", content, base_dir=base_dir, metadata=metadata)


def uplift_scenario(
    content: str,
    *,
    base_dir: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist scenario.md with minimal structured frontmatter."""
    return uplift_layer_file("scenario", content, base_dir=base_dir, metadata=metadata)



def promote_atom_to_layer(
    atom_path: str | Path,
    *,
    destination: str,
    base_dir: Path | None = None,
) -> Path:
    """Promote one atom markdown file into persona/scenario and remove it."""
    source_path = Path(atom_path)
    normalized = destination.strip().lower()
    if normalized not in {"persona", "scenario"}:
        raise ValueError(f"Unsupported promotion destination: {destination}")

    body, _, _ = _parse_layer_file(source_path)
    if not body:
        raise ValueError(f"Atom file is empty or unreadable: {source_path}")

    layers_root = (
        get_memory_layers_dir(base_dir=base_dir)
        if base_dir is not None
        else source_path.parent.parent
    )
    destination_path = layers_root / f"{normalized}.md"
    existing = _read_layer_file(destination_path)
    combined = f"{existing}\n\n{body}".strip() if existing else body.strip()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(combined.rstrip() + "\n", encoding="utf-8")
    source_path.unlink()
    return destination_path


def _split_candidate_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ")):
            line = line[2:].strip()
        if line:
            lines.append(line)
    return lines


def _classify_rule_candidate(line: str) -> str | None:
    text = (line or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if any(
        phrase in lowered
        for phrase in (
            "is complete",
            "was complete",
            "are complete",
            "completed",
            "done",
            "finished",
            "current task:",
        )
    ):
        return None
    if any(
        phrase in lowered
        for phrase in (
            "run ",
            "before every commit",
            "must ",
            "should ",
            "need to ",
            "remember to ",
            "todo",
        )
    ) and "prefer" not in lowered and "preference" not in lowered:
        return None

    if any(
        phrase in lowered
        for phrase in (
            "prefers ",
            "preference",
            "likes ",
            "dislikes ",
            "usually prefers",
            "tends to prefer",
        )
    ):
        return "persona"

    if _SCENARIO_HINT_RE.search(text):
        return "scenario"
    return None


def extract_rule_candidates(atom_path: str | Path) -> list[dict[str, Any]]:
    """Extract tiny durable-rule candidates from one atom file."""
    path = Path(atom_path)
    raw_text = _read_layer_file(path)
    if not raw_text:
        return []

    frontmatter, body = _parse_frontmatter(raw_text)
    body_text = body if frontmatter else raw_text
    candidates: list[dict[str, Any]] = []
    for line in _split_candidate_lines(body_text):
        layer = _classify_rule_candidate(line)
        if not layer:
            continue
        candidates.append(
            {
                "path": path,
                "layer": layer,
                "content": line,
                "source": str(frontmatter.get("source") or ""),
                "session_id": str(frontmatter.get("session_id") or ""),
                "query": str(frontmatter.get("query") or ""),
            }
        )
    return candidates


def _append_unique_layer_content(
    *,
    layer: str,
    content: str,
    base_dir: Path | None = None,
) -> Path:
    root = get_memory_layers_dir(base_dir=base_dir)
    destination_path = root / f"{layer}.md"
    existing_raw = _read_layer_file(destination_path)
    existing_frontmatter, existing_body = _parse_frontmatter(existing_raw) if existing_raw else ({}, "")
    existing_lines = _split_candidate_lines(existing_body if existing_frontmatter else existing_raw)
    existing_canonical = {canonicalize_atom_text(line) for line in existing_lines}

    new_lines: list[str] = []
    for line in _split_candidate_lines(content):
        canonical = canonicalize_atom_text(line)
        if canonical and canonical not in existing_canonical:
            existing_canonical.add(canonical)
            new_lines.append(line)

    combined_lines = [*existing_lines, *new_lines]
    combined_body = "\n".join(combined_lines).strip()

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if existing_frontmatter:
        destination_path.write_text(
            _compose_frontmatter(existing_frontmatter) + combined_body + "\n",
            encoding="utf-8",
        )
    elif existing_raw:
        destination_path.write_text(combined_body + "\n", encoding="utf-8")
    elif layer == "persona":
        uplift_persona(combined_body, base_dir=base_dir, metadata={"source": "curated-uplift"})
    else:
        uplift_scenario(combined_body, base_dir=base_dir, metadata={"source": "curated-uplift"})
    return destination_path


def auto_promote_rules(*, base_dir: Path | None = None) -> list[Path]:
    """Promote corroborated durable rule candidates into higher memory layers."""
    root = get_memory_layers_dir(base_dir=base_dir)
    atom_records = _load_atom_records(base_dir=base_dir)
    grouped: dict[tuple[str, str], list[Path]] = {}

    for record in atom_records:
        atom_path = record.get("path")
        if not isinstance(atom_path, Path):
            continue
        seen_for_atom: set[tuple[str, str]] = set()
        for candidate in extract_rule_candidates(atom_path):
            layer = str(candidate.get("layer") or "").strip().lower()
            content = str(candidate.get("content") or "").strip()
            key = (layer, canonicalize_atom_text(content))
            if layer not in {"persona", "scenario"} or not key[1] or key in seen_for_atom:
                continue
            grouped.setdefault(key, []).append(atom_path)
            seen_for_atom.add(key)

    promoted_paths: list[Path] = []
    for (layer, canonical_content), paths in grouped.items():
        unique_paths = list(dict.fromkeys(paths))
        if len(unique_paths) < 2:
            continue
        content = None
        for atom_path in unique_paths:
            for candidate in extract_rule_candidates(atom_path):
                if canonicalize_atom_text(str(candidate.get("content") or "")) == canonical_content:
                    content = str(candidate.get("content") or "").strip()
                    break
            if content:
                break
        if not content:
            continue

        destination_path = _append_unique_layer_content(layer=layer, content=content, base_dir=root)
        if destination_path not in promoted_paths:
            promoted_paths.append(destination_path)
        for atom_path in unique_paths:
            try:
                atom_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("Failed removing promoted atom %s: %s", atom_path, e)

    return promoted_paths


def _load_atom_records(*, base_dir: Path | None = None) -> List[dict[str, Any]]:
    root = get_memory_layers_dir(base_dir=base_dir)
    records: List[dict[str, Any]] = []
    atoms_dir = root / "atoms"
    if not atoms_dir.exists() or not atoms_dir.is_dir():
        return records

    for atom_path in sorted(atoms_dir.glob("*.md")):
        body, _, _ = _parse_layer_file(atom_path)
        if not body:
            continue
        raw_text = _read_layer_file(atom_path)
        frontmatter, _ = _parse_frontmatter(raw_text) if raw_text else ({}, "")
        records.append(
            {
                "path": atom_path,
                "body": body,
                "source": str(frontmatter.get("source") or ""),
                "target": str(frontmatter.get("target") or ""),
                "action": str(frontmatter.get("action") or ""),
                "source_ref": str(frontmatter.get("source_ref") or ""),
                "query": str(frontmatter.get("query") or ""),
                "write_origin": str(frontmatter.get("write_origin") or ""),
                "session_id": str(frontmatter.get("session_id") or ""),
                "created_at": str(frontmatter.get("created_at") or ""),
                "tags": _normalize_string_sequence(frontmatter.get("tags")),
                "derived_from": _normalize_string_sequence(frontmatter.get("derived_from")),
                "metadata": dict(frontmatter.get("metadata") or {}) if isinstance(frontmatter.get("metadata"), Mapping) else {},
            }
        )
    return records


def should_promote_atoms_to_scenario(
    atom_records: Sequence[Mapping[str, Any]],
    *,
    base_dir: Path | None = None,
) -> bool:
    """Heuristic for uplifting enough contextual atoms into scenario.md."""
    root = get_memory_layers_dir(base_dir=base_dir)
    if _read_layer_file(root / "scenario.md"):
        return False
    relevant = [r for r in atom_records if _SCENARIO_HINT_RE.search(str(r.get("body") or ""))]
    if not relevant:
        return False
    total_len = sum(len(str(r.get("body") or "").strip()) for r in relevant)
    return len(relevant) >= 2 or total_len >= 220


def should_promote_to_persona(
    atom_records: Sequence[Mapping[str, Any]],
    *,
    base_dir: Path | None = None,
) -> bool:
    """Heuristic for uplifting stable preference/rule atoms into persona.md."""
    root = get_memory_layers_dir(base_dir=base_dir)
    if _read_layer_file(root / "persona.md"):
        return False
    relevant = [r for r in atom_records if _SESSION_HINT_RE.search(str(r.get("body") or ""))]
    if not relevant:
        return False
    total_len = sum(len(str(r.get("body") or "").strip()) for r in relevant)
    return len(relevant) >= 2 or total_len >= 180


def _scenario_cluster_key(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    query = str(record.get("query") or metadata.get("query") or "").strip().lower()
    source_ref = str(record.get("source_ref") or metadata.get("source_ref") or "").strip().lower()
    tags = [
        tag.strip().lower()
        for tag in _normalize_string_sequence(record.get("tags"))
        if tag.strip().lower() not in {"memory", "ingest", "session", "recall", "excerpt", "add"}
    ]
    tag_key = ",".join(sorted(dict.fromkeys(tags))[:4])
    return "|".join(part for part in (query, tag_key, source_ref) if part) or canonicalize_atom_text(str(record.get("body") or ""))[:120]


def cluster_atoms_to_scenarios(atom_records: Sequence[Mapping[str, Any]]) -> List[dict[str, Any]]:
    """Cluster scenario-like atoms using query/tag/source_ref as a tiny stable key."""
    clusters: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for record in atom_records:
        body = str(record.get("body") or "")
        if not _SCENARIO_HINT_RE.search(body):
            continue
        key = _scenario_cluster_key(record)
        if key not in clusters:
            clusters[key] = []
            order.append(key)
        clusters[key].append(record)

    clustered: list[dict[str, Any]] = []
    for key in order:
        group = clusters[key]
        best = max(group, key=_record_sort_key)
        metadata = best.get("metadata") if isinstance(best.get("metadata"), Mapping) else {}
        clustered.append(
            {
                "key": key,
                "query": str(best.get("query") or metadata.get("query") or ""),
                "source_ref": str(best.get("source_ref") or metadata.get("source_ref") or ""),
                "tags": _unique_strings(tag for item in group for tag in _normalize_string_sequence(item.get("tags"))),
                "items": [str(item.get("body") or "").strip() for item in group if str(item.get("body") or "").strip()],
                "metadata": _merge_record_metadata(group),
            }
        )
    return clustered


def _parse_persona_fact(line: str) -> tuple[str, str] | None:
    match = _PERSONA_FACT_RE.match(line or "")
    if not match:
        return None
    key = canonicalize_atom_text(match.group("key"))
    value = (match.group("value") or "").strip()
    if not key or not value:
        return None
    return key, value


def merge_persona_facts(
    base_persona_text: str,
    atom_records: Sequence[Mapping[str, Any]],
) -> str:
    """Merge persona-like facts and resolve conflicts by source priority, then recency."""
    ordered_keys: list[str] = []
    fact_map: dict[str, dict[str, Any]] = {}

    for line in (base_persona_text or "").splitlines():
        parsed = _parse_persona_fact(line)
        if not parsed:
            continue
        key, value = parsed
        if key not in fact_map:
            ordered_keys.append(key)
        fact_map[key] = {"value": value, "source": "curated-uplift", "created_at": ""}

    for record in atom_records:
        body = str(record.get("body") or "")
        parsed = _parse_persona_fact(body)
        if not parsed:
            continue
        key, value = parsed
        candidate = {
            "value": value,
            "source": str(record.get("source") or ""),
            "created_at": _timestamp_from_record(record),
        }
        existing = fact_map.get(key)
        if existing is None:
            ordered_keys.append(key)
            fact_map[key] = candidate
            continue
        if (candidate["created_at"], _source_priority(candidate["source"])) >= (
            str(existing.get("created_at") or ""),
            _source_priority(existing["source"]),
        ):
            fact_map[key] = candidate

    if not fact_map:
        return (base_persona_text or "").strip()

    lines = [f"- {key}: {fact_map[key]['value']}" for key in ordered_keys if key in fact_map]
    return "\n".join(lines).strip()


def maybe_rewrite_persona(
    base_persona_text: str,
    atom_records: Sequence[Mapping[str, Any]],
) -> str:
    """Return a conflict-resolved persona body when atom facts add signal."""
    merged = merge_persona_facts(base_persona_text, atom_records)
    return merged or (base_persona_text or "").strip()


def maybe_promote_layers(*, base_dir: Path | None = None) -> dict[str, Any]:
    """Auto-promote existing atom files into persona/scenario using tiny heuristics."""
    atom_records = _load_atom_records(base_dir=base_dir)
    promoted: dict[str, Any] = {"persona": [], "scenario": []}

    if should_promote_to_persona(atom_records, base_dir=base_dir):
        persona_candidates = [
            record
            for record in atom_records
            if _SESSION_HINT_RE.search(str(record.get("body") or ""))
        ]
        if persona_candidates:
            canonical_groups: dict[str, list[Mapping[str, Any]]] = {}
            for record in persona_candidates:
                canonical = canonicalize_atom_text(str(record.get("body") or ""))
                if not canonical:
                    continue
                canonical_groups.setdefault(canonical, []).append(record)

            chosen_records: list[Mapping[str, Any]] = []
            if canonical_groups:
                best_key = max(
                    canonical_groups,
                    key=lambda key: (
                        len(canonical_groups[key]),
                        max(_record_sort_key(item) for item in canonical_groups[key]),
                    ),
                )
                chosen_records = sorted(canonical_groups[best_key], key=_record_sort_key)

            for record in chosen_records:
                try:
                    destination = promote_atom_to_layer(record["path"], destination="persona", base_dir=base_dir)
                    promoted["persona"].append(str(destination))
                except Exception as e:
                    logger.debug("Failed promoting atom %s to persona: %s", record.get("path"), e)

    atom_records = _load_atom_records(base_dir=base_dir)
    if should_promote_atoms_to_scenario(atom_records, base_dir=base_dir):
        scenario_candidates = [
            record
            for record in atom_records
            if _SCENARIO_HINT_RE.search(str(record.get("body") or ""))
        ]
        clusters = cluster_atoms_to_scenarios(scenario_candidates)
        if clusters:
            best_cluster = max(
                clusters,
                key=lambda cluster: (
                    len(cluster.get("items", [])),
                    sum(len(item) for item in cluster.get("items", [])),
                ),
            )
            cluster_items = {
                canonicalize_atom_text(item)
                for item in best_cluster.get("items", [])
                if canonicalize_atom_text(item)
            }
            for record in scenario_candidates:
                body = str(record.get("body") or "")
                if canonicalize_atom_text(body) not in cluster_items:
                    continue
                try:
                    destination = promote_atom_to_layer(record["path"], destination="scenario", base_dir=base_dir)
                    promoted["scenario"].append(str(destination))
                except Exception as e:
                    logger.debug("Failed promoting atom %s to scenario: %s", record.get("path"), e)

    return promoted


def _parse_layer_file(path: Path) -> tuple[str, dict[str, Any], tuple[Any, ...]]:
    raw_text = _read_layer_file(path)
    if not raw_text:
        return "", {}, ()

    frontmatter, body = _parse_frontmatter(raw_text)
    if not frontmatter:
        return raw_text, {}, ()

    structure = frontmatter.get("structure")
    if not isinstance(structure, Mapping):
        structure = frontmatter.get("structured_fields")
    if not isinstance(structure, Mapping):
        structure = {}

    relations = frontmatter.get("relations")
    if not isinstance(relations, Sequence) or isinstance(relations, (str, bytes, bytearray)):
        relations = frontmatter.get("relationship")
    if isinstance(relations, Sequence) and not isinstance(relations, (str, bytes, bytearray)):
        normalized_relations: tuple[Any, ...] = tuple(relations)
    elif relations:
        normalized_relations = (relations,)
    else:
        normalized_relations = ()

    return body.strip(), dict(structure), normalized_relations


def _entry_from_file(
    *,
    key: str,
    label: str,
    content: str,
    path: Path,
    structure: Mapping[str, Any],
    relations: Sequence[Any],
    fallback_type: str = "memory-layer",
    fallback_id: str | None = None,
    source_override: str | None = None,
    metadata_override: Mapping[str, Any] | None = None,
    tags_override: Sequence[str] | None = None,
    derived_from_override: Sequence[str] | None = None,
) -> LayerEntry:
    raw_text = _read_layer_file(path)
    frontmatter, _ = _parse_frontmatter(raw_text) if raw_text else ({}, "")
    meta = frontmatter.get("metadata")
    if not isinstance(meta, Mapping):
        meta = {}
    if metadata_override:
        merged_meta = dict(meta)
        merged_meta.update(dict(metadata_override))
    else:
        merged_meta = dict(meta)

    tags = tags_override if tags_override is not None else _normalize_string_sequence(frontmatter.get("tags"))
    derived_from = (
        tuple(derived_from_override)
        if derived_from_override is not None
        else _normalize_string_sequence(frontmatter.get("derived_from"))
    )

    return LayerEntry(
        key=key,
        label=label,
        content=content,
        source=source_override or str(frontmatter.get("source") or str(path)),
        id=str(frontmatter.get("id") or fallback_id or path.stem),
        type=str(frontmatter.get("type") or fallback_type),
        source_ref=str(frontmatter.get("source_ref") or ""),
        derived_from=tuple(derived_from),
        tags=tuple(tags),
        metadata=merged_meta,
        structure=structure,
        relations=relations,
    )


def load_memory_layers(*, base_dir: Path | None = None) -> List[LayerEntry]:
    """Load structured memory layers from the active profile directory.

    Supported files under ``$HERMES_HOME/memory_layers``:
      - persona.md
      - scenario.md
      - atom.md
      - atoms/*.md

    Files may optionally include YAML-style frontmatter with ``structure`` and
    ``relations`` fields. Files without frontmatter remain fully supported.
    """
    root = get_memory_layers_dir(base_dir=base_dir)
    entries: List[LayerEntry] = []
    atom_records = dedupe_atoms(_load_atom_records(base_dir=base_dir))

    persona_content, persona_structure, persona_relations = _parse_layer_file(root / "persona.md")
    persona_content = maybe_rewrite_persona(persona_content, atom_records)
    if persona_content:
        entries.append(
            _entry_from_file(
                key="persona",
                label="Persona",
                content=persona_content,
                path=root / "persona.md",
                structure=persona_structure,
                relations=persona_relations,
                fallback_type="persona",
                fallback_id="persona",
            )
        )

    scenario_content, scenario_structure, scenario_relations = _parse_layer_file(root / "scenario.md")
    scenario_clusters = cluster_atoms_to_scenarios(atom_records)
    if scenario_clusters:
        cluster_lines: list[str] = []
        for cluster in scenario_clusters:
            header_bits = []
            if cluster.get("query"):
                header_bits.append(f"query={cluster['query']}")
            if cluster.get("source_ref"):
                header_bits.append(f"source_ref={cluster['source_ref']}")
            cluster_tags = _normalize_string_sequence(cluster.get("tags"))
            if cluster_tags:
                header_bits.append(f"tags={', '.join(cluster_tags)}")
            heading = f"- cluster[{cluster['key']}]"
            if header_bits:
                heading += f" ({', '.join(header_bits)})"
            items = [f"  - {item}" for item in cluster.get("items", [])]
            cluster_lines.append("\n".join([heading, *items]).strip())
        cluster_text = "\n\n".join(line for line in cluster_lines if line.strip())
        scenario_content = f"{scenario_content}\n\n{cluster_text}".strip() if scenario_content else cluster_text
    if scenario_content:
        entries.append(
            _entry_from_file(
                key="scenario",
                label="Scenario",
                content=scenario_content,
                path=root / "scenario.md",
                structure=scenario_structure,
                relations=scenario_relations,
                fallback_id="scenario",
            )
        )

    atom_sections: List[str] = []
    aggregate_structure: dict[str, Any] = {}
    aggregate_relations: List[Any] = []
    aggregate_tags: List[str] = []
    aggregate_derived_from: List[str] = []
    aggregate_metadata: dict[str, Any] = {}

    atom_single_path = root / "atom.md"
    atom_single, atom_single_structure, atom_single_relations = _parse_layer_file(atom_single_path)
    if atom_single:
        atom_sections.append(atom_single)
        aggregate_structure.update(atom_single_structure)
        aggregate_relations.extend(atom_single_relations)
        atom_single_frontmatter, _ = _parse_frontmatter(_read_layer_file(atom_single_path))
        aggregate_tags.extend(_normalize_string_sequence(atom_single_frontmatter.get("tags")))
        aggregate_derived_from.extend(_normalize_string_sequence(atom_single_frontmatter.get("derived_from")))
        single_metadata = atom_single_frontmatter.get("metadata")
        if isinstance(single_metadata, Mapping):
            aggregate_metadata.update(dict(single_metadata))

    atoms_dir = root / "atoms"
    try:
        if atom_records:
            for record in atom_records:
                atom_path = record.get("path")
                if not isinstance(atom_path, Path):
                    continue
                atom_content = str(record.get("body") or "")
                _, atom_structure, atom_relations = _parse_layer_file(atom_path)
                if atom_content:
                    provenance_parts: list[str] = []
                    for key in ("source", "write_origin", "session_id", "query", "source_ref"):
                        value = record.get(key)
                        if value:
                            provenance_parts.append(f"{key}={value}")
                    if provenance_parts:
                        atom_sections.append(
                            f"- {atom_path.stem} ({', '.join(provenance_parts)}):\n{atom_content}"
                        )
                    else:
                        atom_sections.append(f"- {atom_path.stem}:\n{atom_content}")
                if atom_structure:
                    aggregate_structure[atom_path.stem] = dict(atom_structure)
                if atom_relations:
                    aggregate_relations.append({"source": atom_path.stem, "items": list(atom_relations)})
                aggregate_tags.extend(_normalize_string_sequence(record.get("tags")))
                aggregate_derived_from.extend(_normalize_string_sequence(record.get("derived_from")))
                metadata_block = record.get("metadata")
                if isinstance(metadata_block, Mapping):
                    aggregate_metadata.setdefault(atom_path.stem, dict(metadata_block))
    except Exception as e:
        logger.debug("Failed reading memory layer atoms from %s: %s", atoms_dir, e)

    if atom_sections:
        entries.append(
            LayerEntry(
                key="atoms",
                label="Atoms",
                content="\n\n".join(atom_sections),
                source=str(atoms_dir if atoms_dir.exists() else atom_single_path),
                id="atoms",
                type="memory-atom",
                source_ref="",
                derived_from=tuple(dict.fromkeys(aggregate_derived_from)),
                tags=tuple(dict.fromkeys(aggregate_tags)),
                metadata=aggregate_metadata,
                structure=aggregate_structure,
                relations=tuple(aggregate_relations),
            )
        )

    return entries


def render_memory_layers(entries: Iterable[LayerEntry]) -> str:
    """Render structured layers in the existing text format."""
    rendered = [entry.render() for entry in entries if entry.content and entry.content.strip()]
    return "\n\n".join(rendered)
