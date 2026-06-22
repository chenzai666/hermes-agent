#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hindsight_sidecar import HindsightSidecar


def default_query() -> str:
    return (
        "Review recent retained operational and user-facing memories. "
        "Summarize durable lessons, recurring failure modes, stable user preferences, "
        "and candidate skill-worthy procedures. Return concise markdown with sections: "
        "Key Learnings, Recurring Issues, Candidate Skills, Stable Preferences, Open Questions."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal Hindsight sidecar reflect job")
    parser.add_argument("--bank", default="", help="Explicit bank id. Defaults to sidecar shared_ops_bank")
    parser.add_argument("--query", default="", help="Custom reflect query")
    args = parser.parse_args()

    sidecar = HindsightSidecar()
    if not sidecar.enabled:
        print("Hindsight sidecar disabled")
        return 1

    bank_id = args.bank or sidecar.config.shared_ops_bank
    query = args.query or default_query()
    result = sidecar.reflect(bank_id=bank_id, query=query)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bank_id": bank_id,
        "query": query,
        "result": result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
