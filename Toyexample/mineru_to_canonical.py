# Let's create a ready-to-run script that:
# - reads a MinerU JSON (handles pdf_info/pages + para_blocks)
# - flattens it into canonical objects (one per layout block)
# - optionally calls OpenAI (GPT-4o/4o-mini) to classify & caption figures/charts lacking captions
# - updates object_type to "chart" when classifier says so
# - writes canonical JSONL (chunks.jsonl) or JSON array if requested
#
# The script exposes CLI flags:
#   python mineru_to_canonical.py --input mineru.json --output chunks.jsonl --doc-id mydoc \
#       --use-vlm --openai-model gpt-4o-mini --format jsonl
#
# ENV:
#   OPENAI_API_KEY (required only when --use-vlm is set)
#
# Dependencies:
#   pip install openai requests
#
# We'll save this to /mnt/data/mineru_to_canonical.py so the user can download it.

import os
import re
import json
import time
import base64
import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable, Optional

try:
    import requests
except ImportError:
    requests = None  # We will guard usage

# ============== Small utilities ==============

def _load_env(env_path: Optional[str] = None) -> None:
    """Populate os.environ from a .env file if present."""
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    else:
        here = Path(__file__).resolve().parent
        candidates.extend([
            Path.cwd() / ".env",
            here / ".env",
            here.parent / ".env",
        ])

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                os.environ[key] = value.strip()
        except Exception:
            pass
        break

def _norm_bbox(b) -> List[float]:
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [float(x) for x in b]
    if isinstance(b, dict):
        # accept {x1,y1,x2,y2} or {left,top,right,bottom}
        x1 = b.get("x1", b.get("left", 0.0))
        y1 = b.get("y1", b.get("top", 0.0))
        x2 = b.get("x2", b.get("right", 1.0))
        y2 = b.get("y2", b.get("bottom", 1.0))
        return [float(x1), float(y1), float(x2), float(y2)]
    return [0.0, 0.0, 1.0, 1.0]

def _bbox_normalize_xyxy(b: List[float], page_w: float, page_h: float) -> List[float]:
    if not page_w or not page_h:
        return b
    x1, y1, x2, y2 = b
    return [x1/page_w, y1/page_h, x2/page_w, y2/page_h]

def _text_hash(s: str) -> str:
    s = (s or "").strip()
    if len(s) > 200:
        s = s[:200]  # don't hash megabytes
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# ============== Content collectors ==============

def _collect_text_from_lines(block: Dict[str, Any]) -> str:
    pieces: List[str] = []
    # some dumps use block["lines"][*]["spans"][*]["content"]
    for line in block.get("lines", []) or []:
        # line-level text (rare but possible)
        if isinstance(line.get("text"), str):
            pieces.append(line["text"])
        for span in line.get("spans", []) or []:
            if span.get("type") == "text" and isinstance(span.get("content"), str):
                pieces.append(span["content"])
    # fallbacks
    if not pieces and isinstance(block.get("text"), str):
        pieces.append(block["text"])
    # normalize whitespace a bit
    txt = " ".join(pieces).replace(" \n", "\n").strip()
    # collapse double spaces
    txt = re.sub(r"\s+", " ", txt)
    return txt

def _extract_table_html(block: Dict[str, Any]) -> Optional[str]:
    # prefer explicit HTML serialization if present
    for line in block.get("lines", []) or []:
        for span in line.get("spans", []) or []:
            if isinstance(span.get("html"), str):
                return span["html"]
    # some dumps embed table html directly under the block
    if isinstance(block.get("html"), str):
        return block["html"]
    return None

def _deep_find_image_path(node: Any) -> Optional[str]:
    # walk arbitrarily nested dict/list to find an image span with image_path
    if isinstance(node, dict):
        if node.get("type") == "image" and isinstance(node.get("image_path"), str):
            return node["image_path"]
        for v in node.values():
            hit = _deep_find_image_path(v)
            if hit: return hit
    elif isinstance(node, list):
        for v in node:
            hit = _deep_find_image_path(v)
            if hit: return hit
    return None

# ============== Typing & mapping ==============

def _map_type(raw: str) -> str:
    t = (raw or "").lower()
    return {
        "title": "heading",
        "heading": "heading",
        "text": "paragraph",
        "paragraph": "paragraph",
        "list": "list_item",
        "list_item": "list_item",
        "table": "table",
        "table_body": "table",
        "image": "figure",
        "image_body": "figure",
        "figure": "figure",
        "chart": "chart",
        "graph": "chart",
        "caption": "caption",
        "formula": "formula",
        "equation": "formula",
        "code": "code",
    }.get(t, "paragraph")  # unknown → paragraph (safe default)

# ============== Block extraction ==============

def _emit_block(block: Dict[str, Any], doc_id: str, page_no: int, page_w: float, page_h: float) -> Optional[Dict[str, Any]]:
    raw_type = block.get("type") or block.get("category") or "paragraph"
    obj_type = _map_type(raw_type)
    bbox = [round(v, 4) for v in _bbox_normalize_xyxy(_norm_bbox(block.get("bbox")), page_w, page_h)]
    attrs: Dict[str, Any] = {}
    extracted_caption = None
    text = ""

    if obj_type in ("heading", "paragraph", "list_item", "caption"):
        text = _collect_text_from_lines(block)
        if obj_type == "heading" and isinstance(block.get("level"), int):
            attrs["heading_level"] = block["level"]

    elif obj_type == "table":
        html = _extract_table_html(block)
        if html:
            text = html
            attrs["table"] = {"format": "html"}
        else:
            text = _collect_text_from_lines(block)
            attrs["table"] = {"format": "text"}
        if isinstance(block.get("caption"), str):
            extracted_caption = block["caption"]

    elif obj_type in ("figure", "chart"):
        img_ref = _deep_find_image_path(block)
        attrs["figure"] = {"image_ref": img_ref}
        if isinstance(block.get("caption"), str):
            extracted_caption = block["caption"]
        text = _collect_text_from_lines(block)  # OCR’d labels may appear here

    elif obj_type == "formula":
        text = _collect_text_from_lines(block)
        if isinstance(block.get("latex"), str):
            attrs["formula"] = {"latex": block["latex"]}

    elif obj_type == "code":
        text = _collect_text_from_lines(block)
        if isinstance(block.get("language"), str):
            attrs["code"] = {"language": block["language"]}

    # linearize for embeddings
    render = text or ""
    if obj_type in ("figure","chart") and extracted_caption:
        render = f"caption: {extracted_caption}\n{text}".strip()
    if obj_type == "table" and extracted_caption:
        render = f"caption: {extracted_caption}\n{text}".strip()
    if obj_type == "heading" and attrs.get("heading_level"):
        render = f"[HEADING L{attrs['heading_level']}] {text}"

    return {
        "doc_id": doc_id,
        "page": page_no,
        "object_type": obj_type,
        "bbox": bbox,
        "text": text,
        "extracted_caption": extracted_caption,
        "generated_desc": None,
        "attrs": attrs,
        "render_for_embedding": render
    }

def _iter_candidate_blocks(page: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield candidate blocks from the page, supporting:
    - page['blocks'], page['elements'], and page['para_blocks']
    - nested 'blocks' under a block (e.g., table_body / image_body)
    """
    seen: set[int] = set()

    def walk(node: Any):
        if isinstance(node, dict):
            # a "block" is any dict that has both bbox and type
            if "bbox" in node and "type" in node:
                oid = id(node)
                if oid not in seen:
                    seen.add(oid)
                    yield node
                # also look into nested body blocks (do not stop here)
                for k in ("blocks", "elements"):
                    if k in node and isinstance(node[k], list):
                        for child in node[k]:
                            yield from walk(child)
            else:
                # generic walk
                for v in node.values():
                    if isinstance(v, (dict, list)):
                        for child in walk(v):
                            yield child
        elif isinstance(node, list):
            for it in node:
                for child in walk(it):
                    yield child

    # preferred top-level containers (order matters)
    for key in ("blocks", "elements", "para_blocks"):
        if isinstance(page.get(key), list):
            for b in page[key]:
                yield from walk(b)

def flatten_mineru(doc_json: Dict[str, Any], doc_id: str, dedup: bool = True) -> List[Dict[str, Any]]:
    # find pages under pdf_info[*] or pages[*]
    pages = []
    if isinstance(doc_json.get("pdf_info"), list):
        pages = doc_json["pdf_info"]
    elif isinstance(doc_json.get("pages"), list):
        pages = doc_json["pages"]
    else:
        raise ValueError("Cannot find pages: expected 'pdf_info' or 'pages' at top level.")

    out: List[Dict[str, Any]] = []
    sigset = set()  # for dedup

    for p in pages:
        page_no = int(p.get("page_idx") or p.get("page_number") or 1)
        page_size = p.get("page_size") or [1, 1]
        pw, ph = (float(page_size[0] or 1), float(page_size[1] or 1))

        local_idx = 0
        for blk in _iter_candidate_blocks(p):
            emitted = _emit_block(blk, doc_id, page_no, pw, ph)
            if not emitted:
                continue

            # simple dedup signature: (page, type, bbox_rounded, text_hash)
            if dedup:
                bx = emitted["bbox"]
                sig = (
                    page_no,
                    emitted["object_type"],
                    tuple(round(v, 4) for v in bx),
                    _text_hash(emitted.get("text",""))
                )
                if sig in sigset:
                    continue
                sigset.add(sig)

            emitted["chunk_id"] = f"{doc_id}_p{page_no}_obj{local_idx}"
            local_idx += 1
            out.append(emitted)

    return out

# ============== Optional: VLM classification + caption via OpenAI ==============

def _guess_mime_from_name(name: str) -> str:
    name = (name or "").lower()
    if name.endswith(".png"): return "image/png"
    if name.endswith(".webp"): return "image/webp"
    if name.endswith(".gif"): return "image/gif"
    return "image/jpeg"

def _download_bytes(url: str, timeout: int = 20) -> Optional[bytes]:
    if not requests:
        raise RuntimeError("requests not installed; run `pip install requests` to enable downloads.")
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        return None
    return None

def _load_image_bytes(image_ref: str) -> Optional[bytes]:
    if not image_ref:
        return None
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        return _download_bytes(image_ref)
    path = Path(image_ref)
    if path.is_file():
        try:
            return path.read_bytes()
        except Exception:
            return None
    return None

def _to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _openai_caption_and_classify(image_bytes: bytes, filename_hint: str = "image.jpg", model: str = "gpt-4o-mini") -> Optional[Dict[str, Any]]:
    """
    Calls OpenAI chat.completions with a base64 data URL.
    Returns dict like: {"class": "chart|figure|diagram|screenshot|photo", "caption": "...", "keywords": [...]}
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed; run `pip install openai`.") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is required when --use-vlm is enabled.")

    client = OpenAI(api_key=api_key)
    mime = _guess_mime_from_name(filename_hint)
    data_url = _to_data_url(image_bytes, mime=mime)

    prompt = (
        "You create retrieval metadata for document figures.\n"
        "1) Classify the image into one of: chart, plot, diagram, screenshot_of_text, photo, other.\n"
        "2) Return a concise factual caption (<=40 words).\n"
        "3) Return 3-8 keywords.\n"
        "If it's a chart/plot, include chart type, axes, and the main trend in the caption if visible.\n"
        "Output JSON ONLY with fields: {\"class\":\"...\",\"caption\":\"...\",\"keywords\":[\"...\"]}."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful vision assistant for document understanding."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=220
        )
    except Exception as e:
        print("OpenAI API error:", e)
        return None

    text = (resp.choices[0].message.content or "").strip()
    # try to parse a JSON block from the model content
    try:
        m = re.search(r'\{.*\}', text, re.S)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    # fallback: treat the whole string as caption
    return {"class": "figure", "caption": text, "keywords": []}

def inject_vlm_metadata(
    objs: List[Dict[str, Any]],
    rate_limit_s: float = 0.6,
    model: str = "gpt-4o-mini",
    classify: bool = True
) -> List[Dict[str, Any]]:
    """
    For each figure/chart block, call OpenAI to classify and caption the visual.
    Updates include:
      - generated_desc
      - attrs.figure/chart keywords
      - object_type -> 'chart' if classification returns chart/plot
    """
    updated: List[Dict[str, Any]] = []
    for o in objs:
        o2 = dict(o)
        is_fig_or_chart = o2.get("object_type") in ("figure", "chart")
        img_ref = (o2.get("attrs", {}).get("figure", {}) or {}).get("image_ref")

        if is_fig_or_chart and isinstance(img_ref, str):
            img_bytes = _load_image_bytes(img_ref)
            if not img_bytes:
                print(f"[warn] unable to load image bytes for {img_ref}")
            else:
                result = _openai_caption_and_classify(img_bytes, filename_hint=img_ref, model=model)
                if result:
                    o2["generated_desc"] = result.get("caption")
                    img_class = (result.get("class") or "").lower()
                    if classify and img_class in ("chart", "plot"):
                        o2["object_type"] = "chart"
                        chart_info = o2.setdefault("attrs", {}).setdefault("chart", {})
                        chart_info.setdefault("kind", "plot" if img_class == "plot" else "chart")
                        chart_info["classification"] = img_class
                    else:
                        fig_info = o2.setdefault("attrs", {}).setdefault("figure", {})
                        if img_class:
                            fig_info["classification"] = img_class

                    kws = result.get("keywords") or []
                    if o2.get("object_type") == "chart":
                        o2.setdefault("attrs", {}).setdefault("chart", {})["keywords"] = kws
                    else:
                        o2.setdefault("attrs", {}).setdefault("figure", {})["keywords"] = kws

                    cap_for_render = o2.get("extracted_caption") or o2.get("generated_desc") or ""
                    text = o2.get("text", "") or ""
                    if cap_for_render:
                        o2["render_for_embedding"] = f"caption: {cap_for_render}\n{text}".strip()

                time.sleep(rate_limit_s)

        updated.append(o2)
    return updated

# ============== I/O helpers ==============

def _save_jsonl(objs: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

def _save_json(objs: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(objs, f, ensure_ascii=False, indent=2)

# ============== Main CLI ==============

def main():
    ap = argparse.ArgumentParser(description="Convert MinerU JSON to canonical chunks; optionally classify/caption images via OpenAI VLM.")
    ap.add_argument("--input", required=True, help="Path to MinerU JSON file")
    ap.add_argument("--output", required=True, help="Path to output file (jsonl or json)")
    ap.add_argument("--doc-id", required=True, help="Document ID to use in chunk IDs")
    ap.add_argument("--no-dedup", action="store_true", help="Disable deduplication of near-duplicate blocks")
    ap.add_argument("--use-vlm", action="store_true", help="Call OpenAI VLM to classify/caption figure and chart blocks")
    ap.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI vision model (e.g., gpt-4o-mini or gpt-4o)")
    ap.add_argument("--no-classify", action="store_true", help="Do not change object_type based on VLM classification")
    ap.add_argument("--rate-limit", type=float, default=0.6, help="Seconds to sleep between VLM calls")
    ap.add_argument("--format", choices=["jsonl","json"], default="json", help="Output format")
    args = ap.parse_args()

    _load_env()

    with open(args.input, "r", encoding="utf-8") as f:
        mineru_raw = json.load(f)

    objs = flatten_mineru(mineru_raw, doc_id=args.doc_id, dedup=(not args.no_dedup))

    if args.use_vlm:
        objs = inject_vlm_metadata(
            objs,
            rate_limit_s=args.rate_limit,
            model=args.openai_model,
            classify=not args.no_classify
        )

    if args.format == "jsonl":
        _save_jsonl(objs, args.output)
    else:
        _save_json(objs, args.output)

    print(f"Wrote {len(objs)} objects to {args.output}")

if __name__ == "__main__":
    main()
