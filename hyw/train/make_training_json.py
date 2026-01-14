# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_manifest", type=str, required=True, help="manifest_raw_{split}.json from sythgenerator")
    p.add_argument("--latent_root", type=str, required=True, help="Root containing per-sample latent.pt outputs")
    p.add_argument("--out_json", type=str, required=True, help="Output json_path for HY-WorldPlay training")
    p.add_argument("--max_samples", type=int, default=0, help="0 means no limit")
    args = p.parse_args()

    raw_manifest = Path(args.raw_manifest).expanduser().resolve()
    latent_root = Path(args.latent_root).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    items = json.loads(raw_manifest.read_text(encoding="utf-8"))
    if args.max_samples and args.max_samples > 0:
        items = items[: args.max_samples]

    train_items: list[dict] = []
    for i, it in enumerate(items):
        sample_dir = latent_root / f"sample_{i:05d}"
        latent_path = sample_dir / "latent.pt"
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent.pt: {latent_path}")

        train_items.append(
            {
                "latent_path": str(latent_path),
                "pose_path": str(Path(it["pose_path"]).expanduser().resolve()),
                "action_path": str(Path(it["action_path"]).expanduser().resolve()),
            }
        )

    out_json.write_text(json.dumps(train_items, indent=2), encoding="utf-8")
    print(f"[OK] wrote {len(train_items)} items -> {out_json}")


if __name__ == "__main__":
    main()


