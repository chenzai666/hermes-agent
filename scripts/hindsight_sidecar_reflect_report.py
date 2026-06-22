#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hindsight_sidecar import HindsightSidecar
from agent.hindsight_reflect_flow import backflow_reflect_report, record_reflect_run


def default_query() -> str:
    return (
        "Review recent retained operational and user-facing memories. "
        "Summarize durable lessons, recurring failure modes, stable user preferences, "
        "and candidate skill-worthy procedures. Return concise markdown with sections: "
        "Key Learnings, Recurring Issues, Candidate Skills, Stable Preferences, Open Questions."
    )


def _clean_reflect_text(text: str) -> str:
    body = str(text or '').strip()
    if not body:
        return '_No result text returned._'

    if body.startswith('```'):
        lines = body.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        body = '\n'.join(lines).strip()

    body = re.sub(r'^<think>.*?</think>\s*', '', body, flags=re.DOTALL).strip()

    for marker in (
        '## Final Answer',
        '# Final Answer',
        'Final Answer:',
        'Final answer:',
        'Answer:',
    ):
        idx = body.find(marker)
        if idx != -1:
            candidate = body[idx + len(marker):].strip()
            if candidate:
                body = candidate
                break

    body = re.sub(r'^<think>.*?</think>\s*', '', body, flags=re.DOTALL).strip()
    body = re.sub(r'\n{3,}', '\n\n', body)

    return body.strip() or '_No result text returned._'


def render_markdown(bank_id: str, query: str, result: dict) -> str:
    text = result.get('text') if isinstance(result, dict) else str(result)
    return _clean_reflect_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run Hindsight sidecar reflect and write markdown report')
    parser.add_argument('--bank', default='', help='Explicit bank id. Defaults to sidecar shared_ops_bank')
    parser.add_argument('--query', default='', help='Custom reflect query')
    parser.add_argument('--output-dir', default='/root/.hermes/hindsight-reports', help='Directory to write markdown reports')
    parser.add_argument('--backflow-destination', choices=['atom', 'persona', 'scenario'], default='atom', help='Memory layer destination for reflect backflow')
    parser.add_argument('--dry-run', action='store_true', help='Render/report and record state without writing into memory_layers')
    parser.add_argument('--force', action='store_true', help='Force backflow even if the latest reflected content hash already matches')
    args = parser.parse_args()

    sidecar = HindsightSidecar()
    if not sidecar.enabled:
        print('Hindsight sidecar disabled', file=sys.stderr)
        return 1

    bank_id = args.bank or sidecar.config.shared_ops_bank
    query = args.query or default_query()
    result = sidecar.reflect(bank_id=bank_id, query=query)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    rendered = render_markdown(bank_id, query, result)

    path = out_dir / f'reflect_{bank_id.replace(":", "_")}_{stamp}.md'
    path.write_text(rendered, encoding='utf-8')

    latest_path = out_dir / 'latest.md'
    latest_path.write_text(rendered, encoding='utf-8')

    state_path = record_reflect_run(
        bank_id=bank_id,
        query=query,
        rendered=rendered,
        output_path=path,
        latest_path=latest_path,
        output_dir=out_dir,
        usage=result.get('usage') if isinstance(result, dict) and isinstance(result.get('usage'), dict) else None,
    )
    backflow = backflow_reflect_report(
        bank_id=bank_id,
        query=query,
        rendered=rendered,
        output_path=path,
        latest_path=latest_path,
        output_dir=out_dir,
        destination=args.backflow_destination,
        dry_run=args.dry_run,
        force=args.force,
    )

    payload = {
        'ok': True,
        'bank_id': bank_id,
        'query': query,
        'output_path': str(path),
        'latest_path': str(latest_path),
        'state_path': str(state_path),
        'backflow': backflow.to_dict(),
        'usage': result.get('usage') if isinstance(result, dict) else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
