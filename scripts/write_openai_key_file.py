#!/usr/bin/env python
from __future__ import annotations

import getpass
from pathlib import Path


def main() -> None:
    target = Path("/tmp/rl_epistemics_openai_key")
    key = getpass.getpass("OpenAI API key: ").strip()
    if not key:
        raise SystemExit("No key entered; not writing empty key file.")
    target.write_text(key, encoding="utf-8")
    target.chmod(0o600)
    print(f"key file ready: {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
