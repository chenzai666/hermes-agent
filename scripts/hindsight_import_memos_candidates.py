#!/usr/bin/env python3
"""Import dry-run Memos candidates into isolated Hindsight banks.

Defaults to a 100-item trial per source. This script is intentionally
idempotent-ish at the document_id level: each imported candidate uses a stable
memos:<source>:<id> document_id and update_mode=replace.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DRYRUN_DIR = Path('/root/.hermes/hindsight-import-dryrun')
API = 'http://127.0.0.1:8888'
BANKS = {
    'hermes': 'import:memos:hermes:knowledge-dryrun',
    'openclaw': 'import:memos:openclaw:knowledge-dryrun',
}


def post_json(url: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def put_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='PUT')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def load_candidates(source: str, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    path = DRYRUN_DIR / f'{source}-candidates.jsonl'
    items: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if idx < offset:
                continue
            row = json.loads(line)
            items.append(row)
            if len(items) >= limit:
                break
    return items


def to_iso(created_at: Any) -> str | None:
    try:
        ts = int(created_at)
    except Exception:
        return None
    # Memos stores unix seconds in observed DBs.
    if ts > 10_000_000_000:
        ts = ts // 1000
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def make_memory_item(source: str, row: dict[str, Any]) -> dict[str, Any]:
    cats = row.get('categories') or []
    role = str(row.get('role') or '')
    tags = [
        'source:memos',
        f'source:{source}',
        f'role:{role}',
        'import:dryrun',
        'import:knowledge-candidate',
    ]
    tags.extend(f'category:{c}' for c in cats[:4])
    metadata = {
        'source': source,
        'memos_id': str(row.get('id') or ''),
        'session_key': str(row.get('session_key') or ''),
        'turn_id': str(row.get('turn_id') or ''),
        'seq': str(row.get('seq') or ''),
        'role': role,
        'score': str(row.get('score') or ''),
    }
    return {
        'content': row.get('content') or '',
        'context': f"Memos {source} candidate; role={role}; categories={','.join(cats)}; score={row.get('score')}",
        'document_id': f"memos:{source}:{row.get('id')}",
        'timestamp': to_iso(row.get('created_at')) or 'unset',
        'metadata': metadata,
        'tags': tags,
        'observation_scopes': 'per_tag',
        'update_mode': 'replace',
    }


def create_bank(bank_id: str) -> dict[str, Any]:
    payload = {
        'name': bank_id,
        'retain_mission': (
            'Extract durable knowledge from filtered Memos candidates. Keep stable user preferences, '
            'project facts, architecture decisions, verified troubleshooting lessons, and reusable workflows. '
            'Ignore transient status updates, credentials, one-time tokens, and chatter.'
        ),
        'reflect_mission': (
            'Synthesize this import bank into concise reusable knowledge for future Hermes/OpenClaw/Hindsight operations. '
            'Prefer stable facts and proven procedures over session-by-session logs.'
        ),
        'observations_mission': (
            'Create observations only for stable facts, preferences, project configuration, architecture, and verified operational lessons. '
            'Do not create observations for temporary progress messages or credentials.'
        ),
        'retain_extraction_mode': 'concise',
        'enable_observations': False,
    }
    return put_json(f'{API}/v1/default/banks/{bank_id}', payload)


def import_source(source: str, limit: int, batch_size: int, sleep_s: float, async_mode: bool = False, offset: int = 0) -> dict[str, Any]:
    bank = BANKS[source]
    create_resp = create_bank(bank)
    rows = load_candidates(source, limit, offset)
    out = {
        'source': source,
        'bank_id': bank,
        'requested': limit,
        'offset': offset,
        'loaded': len(rows),
        'create_bank': create_resp,
        'batches': [],
    }
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i+batch_size]
        payload = {'async': async_mode, 'items': [make_memory_item(source, r) for r in chunk]}
        started = time.time()
        try:
            resp = post_json(f'{API}/v1/default/banks/{bank}/memories', payload, timeout=600)
            ok = True
        except urllib.error.HTTPError as e:
            ok = False
            resp = {'error': e.read().decode('utf-8', errors='replace'), 'status': e.code}
        elapsed = round(time.time() - started, 2)
        out['batches'].append({'offset': i, 'count': len(chunk), 'ok': ok, 'elapsed_s': elapsed, 'response': resp})
        print(f'{source} batch offset={i} count={len(chunk)} ok={ok} elapsed={elapsed}s')
        sys.stdout.flush()
        if not ok:
            break
        if sleep_s:
            time.sleep(sleep_s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=10)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--source', choices=['hermes', 'openclaw', 'all'], default='all')
    ap.add_argument('--sleep', type=float, default=0.2)
    ap.add_argument('--async-mode', action='store_true', help='submit retain jobs asynchronously')
    args = ap.parse_args()
    sources = ['hermes', 'openclaw'] if args.source == 'all' else [args.source]
    results = []
    for src in sources:
        results.append(import_source(src, args.limit, args.batch_size, args.sleep, args.async_mode, args.offset))
    out_path = DRYRUN_DIR / f'import-result-{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'RESULT_FILE={out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
