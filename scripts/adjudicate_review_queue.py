#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_epistemics.eval.adjudicate_outputs import adjudicate_policy_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-adjudicate the hard-output manual review queue.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key-file", default="/tmp/rl_epistemics_openai_key")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input", default="outputs/blog/manual_review_queue.csv")
    parser.add_argument("--output", default="outputs/blog/manual_review_queue_adjudicated_gpt55.csv")
    parser.add_argument("--metrics", default="outputs/blog/manual_review_queue_adjudicated_gpt55_metrics.json")
    parser.add_argument("--judged-output", default="outputs/blog/manual_review_queue_llm_judged_gpt55.csv")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    key_path = Path(args.api_key_file).expanduser()
    if not key_path.exists() or key_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"API key file is missing or empty: {key_path}. "
            "Create it with: read -s -p 'OpenAI API key: ' OPENAI_KEY; printf '\\n'; "
            "printf '%s' \"$OPENAI_KEY\" > /tmp/rl_epistemics_openai_key; "
            "chmod 600 /tmp/rl_epistemics_openai_key; unset OPENAI_KEY"
        )

    result = adjudicate_policy_outputs(
        {
            "policy_outputs": args.input,
            "output_csv": args.output,
            "output_metrics": args.metrics,
            "write_judged_rows_to": args.judged_output,
            "llm_judge": {
                "enabled": True,
                "provider": args.provider,
                "model": args.model,
                "api_key_file": str(key_path),
                "limit": args.limit,
                "reasoning_effort": args.reasoning_effort,
                "max_tokens": args.max_tokens,
            },
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
