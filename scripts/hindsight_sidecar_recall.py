#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hindsight_sidecar import HindsightSidecar


def _default_bank_ids(sidecar: HindsightSidecar, session_id: str) -> list[str]:
    sid = str(session_id or "unknown").strip() or "unknown"
    return [
        sidecar.bank_for_user(sid),
        sidecar.shared_user_bank(sid),
        sidecar.config.shared_ops_bank,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal Hindsight sidecar multi-bank recall")
    parser.add_argument("query", help="Recall query")
    parser.add_argument("--session-id", default="unknown", help="Session/user identifier used to derive default banks")
    parser.add_argument("--bank", dest="banks", action="append", default=None, help="Explicit bank id. Repeatable.")
    parser.add_argument("--top-k", type=int, default=6, help="Per-bank recall limit")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON only")
    args = parser.parse_args()

    sidecar = HindsightSidecar()
    if not sidecar.enabled:
        print("Hindsight sidecar disabled", file=sys.stderr)
        return 1
    if not sidecar.config.enable_recall:
        print("Hindsight sidecar recall disabled", file=sys.stderr)
        return 2

    bank_ids = list(args.banks or _default_bank_ids(sidecar, args.session_id))
    result = sidecar.recall_many(bank_ids=bank_ids, query=args.query, top_k=max(1, int(args.top_k or 1)))

    payload = {
        "ok": bool(result.get("success")),
        "query": args.query,
        "session_id": args.session_id,
        "banks": result.get("banks") or bank_ids,
        "errors": result.get("errors") or [],
        "result_count": len(result.get("results") or []),
        "rendered": sidecar.format_layered_recall(result.get("results") or []),
    }
    if args.raw:
        payload["raw"] = result
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
