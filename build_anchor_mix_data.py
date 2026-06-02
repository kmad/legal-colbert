"""Build continuation mixes from existing triplet datasets.

P5 improved MLEB with document-side LEDGAR breadth but slightly regressed ACORD.
This builder keeps the P5 dataset intact and adds only ACORD triplets from P2a
as an anchor, avoiding duplicated CUAD triplets from the P2a dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data-dir", default="data_p3_ledgar_docside")
    parser.add_argument("--anchor-data-dir", default="data_p2a_acord_cuad")
    parser.add_argument("--output-dir", default="data_p6_ledgar_acord_anchor")
    parser.add_argument("--anchor-source-prefix", default="acord")
    parser.add_argument("--anchor-repeat", type=int, default=2)
    args = parser.parse_args()

    if args.anchor_repeat < 0:
        raise ValueError("--anchor-repeat must be >= 0")

    base = load_from_disk(str(Path(args.base_data_dir) / "train"))
    anchor_source = load_from_disk(str(Path(args.anchor_data_dir) / "train"))
    if "source" not in anchor_source.column_names:
        raise ValueError(f"{args.anchor_data_dir}/train has no source column")

    anchor = anchor_source.filter(
        lambda row: str(row["source"]).startswith(args.anchor_source_prefix),
        desc=f"Filtering {args.anchor_source_prefix} anchors",
    )
    pieces = [base]
    pieces.extend(anchor for _ in range(args.anchor_repeat))
    mixed: Dataset = concatenate_datasets(pieces)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(out / "train"))

    manifest = {
        "base_data_dir": args.base_data_dir,
        "anchor_data_dir": args.anchor_data_dir,
        "anchor_source_prefix": args.anchor_source_prefix,
        "anchor_repeat": args.anchor_repeat,
        "base_triplets": len(base),
        "anchor_triplets": len(anchor),
        "mixed_triplets": len(mixed),
        "base_columns": base.column_names,
        "anchor_columns": anchor.column_names,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
