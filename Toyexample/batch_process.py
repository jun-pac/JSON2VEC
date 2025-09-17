#!/usr/bin/env python3
"""Batch runner for converting all MinerU JSON files into canonical chunks with VLM enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from mineru_to_canonical import (
    _load_env,
    flatten_mineru,
    inject_vlm_metadata,
    _save_json,
)


def _derive_doc_id(json_path: Path) -> str:
    stem = json_path.stem
    if "_MinerU__" in stem:
        stem = stem.split("_MinerU__", 1)[0]
    return stem


def _format_usage(usage: Dict[str, int]) -> str:
    return "/".join(str(usage.get(key, 0)) for key in ("input_tokens", "output_tokens", "total_tokens"))


def process_all(
    mineru_dir: Path,
    output_dir: Path,
    rate_limit: float = 0.6,
    model: str = "gpt-4o-mini",
    tpm_limit: int = 200_000,
    max_retries: int = 5,
    skip_existing: bool = True,
) -> Tuple[List[str], List[Dict[str, str]]]:
    _load_env()
    mineru_dir = mineru_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files: List[str] = []
    doc_entries: List[Dict[str, str]] = []
    aggregate_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    existing_outputs = {p.name for p in output_dir.glob("*.json")}
    if skip_existing and existing_outputs:
        print(
            f"Detected {len(existing_outputs)} existing canonical JSON files in {output_dir.name}. "
            "Existing outputs will be skipped; delete them or set skip_existing=False to reprocess."
        )

    json_files = sorted(mineru_dir.glob("*.json"))
    for json_path in json_files:
        doc_id = _derive_doc_id(json_path)
        doc_entries.append({"doc_name": doc_id, "doc_id": doc_id})

        out_path = output_dir / f"{doc_id}.json"
        if skip_existing and out_path.name in existing_outputs:
            print(
                f"Skipping {json_path.name} because {out_path.name} already exists."
            )
            continue

        with json_path.open("r", encoding="utf-8") as f:
            mineru_raw = json.load(f)

        objs = flatten_mineru(mineru_raw, doc_id=doc_id)
        objs, usage_summary = inject_vlm_metadata(
            objs,
            rate_limit_s=rate_limit,
            model=model,
            classify=True,
            tpm_limit=tpm_limit,
            max_retries=max_retries,
        )
        _save_json(objs, str(out_path))

        for key in aggregate_usage:
            aggregate_usage[key] += usage_summary.get(key, 0)

        processed_files.append(str(out_path))
        print(
            f"Processed {json_path.name} -> {out_path.name} (doc_id={doc_id}, "
            f"objects={len(objs)}, tokens={_format_usage(usage_summary)})"
        )

    if doc_entries:
        map_path = output_dir.parent / "doc_map.json"
        with map_path.open("w", encoding="utf-8") as f:
            json.dump(doc_entries, f, ensure_ascii=False, indent=2)
        print(
            f"Saved doc mapping for {len(doc_entries)} documents to {map_path.name}. "
            f"Total tokens used: {_format_usage(aggregate_usage)}"
        )
    else:
        print("No MinerU JSON files found to process.")

    return processed_files, doc_entries


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    mineru_dir = base_dir / "mineru_JSON"
    output_dir = base_dir / "canonical_JSON"
    process_all(mineru_dir, output_dir)
