#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hindsight_sidecar import HindsightSidecar
from agent.hindsight_reflect_flow import get_reports_dir, load_state


def main() -> int:
    sidecar = HindsightSidecar()
    reports_dir = get_reports_dir()
    state = load_state(reports_dir)
    payload = {
        'healthcheck': sidecar.healthcheck(),
        'reports_dir': str(reports_dir),
        'state_path': str(reports_dir / 'state.json'),
        'last_reflect': state.get('last_reflect'),
        'last_backflow': state.get('last_backflow'),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
