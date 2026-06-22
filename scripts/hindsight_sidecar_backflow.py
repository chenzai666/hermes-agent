#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hindsight_reflect_flow import backflow_reflect_report, get_reports_dir, load_state


def main() -> int:
    parser = argparse.ArgumentParser(description='Backflow an existing Hindsight reflect report into Hermes memory_layers')
    parser.add_argument('--report', default='', help='Explicit markdown report path. Defaults to hindsight-reports/latest.md')
    parser.add_argument('--bank', default='', help='Override bank id for state/output metadata')
    parser.add_argument('--query', default='', help='Override query for state/output metadata')
    parser.add_argument('--output-dir', default='', help='Reports directory containing state.json/latest.md')
    parser.add_argument('--destination', choices=['atom', 'persona', 'scenario'], default='atom', help='Memory layer destination for backflow')
    parser.add_argument('--dry-run', action='store_true', help='Record intended backflow without writing memory_layers')
    parser.add_argument('--force', action='store_true', help='Force write even if state hash matches latest backflow')
    args = parser.parse_args()

    reports_dir = get_reports_dir(args.output_dir or None)
    state = load_state(reports_dir)
    last_reflect = state.get('last_reflect') if isinstance(state.get('last_reflect'), dict) else {}

    report_path = Path(args.report) if args.report else reports_dir / 'latest.md'
    if not report_path.exists():
        print(json.dumps({'ok': False, 'error': f'report not found: {report_path}'}, ensure_ascii=False, indent=2))
        return 1

    rendered = report_path.read_text(encoding='utf-8').strip()
    bank_id = args.bank or str(last_reflect.get('bank_id') or 'unknown')
    query = args.query or str(last_reflect.get('query') or '')
    latest_path = Path(last_reflect.get('latest_path') or report_path)

    result = backflow_reflect_report(
        bank_id=bank_id,
        query=query,
        rendered=rendered,
        output_path=report_path,
        latest_path=latest_path,
        output_dir=reports_dir,
        destination=args.destination,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(json.dumps({'ok': result.ok, 'backflow': result.to_dict()}, ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
