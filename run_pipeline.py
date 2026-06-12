import argparse
import json
import os
import sys
from pathlib import Path

from src.category_extractor import CategoryExtractor
from src.data_utils import (
    QueryItem,
    calculate_iou,
    discover_queries,
    draw_eval_image,
    dump_json,
    ensure_dir,
    get_video_path,
    load_gt_annotations,
)
from src.visualization import render_tagged_image


DEFAULT_ROOT = "/mnt/si00068187c7/default/myc"
DEFAULT_BASE_DIR = f"{DEFAULT_ROOT}/data/VRBS/input/dengnan"
DEFAULT_OUTPUT_DIR = f"{DEFAULT_ROOT}/data/VRBS/output/dengnan_locateanything_tag_select"
DEFAULT_VLM_MODEL = f"{DEFAULT_ROOT}/models/Qwen3.6-35B-A3B/models"
DEFAULT_SMALL_LM_MODEL = f"{DEFAULT_ROOT}/models/Qwen3.5-0.8B"
DEFAULT_EAGLE_EMBODIED = f"{DEFAULT_ROOT}/code/Eagle/Embodied"
DEFAULT_SMALL_LM_ISOLATED_DEPS = f"{DEFAULT_ROOT}/projects/vrbs_locateanything/.deps/qwen35_transformers"


def parse_query_indexes(value: str | None) -> set[int] | None:
    """Parse a compact query index expression such as 1,3,5-7."""
    if not value:
        return None
    indexes: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indexes.update(range(int(start), int(end) + 1))
        else:
            indexes.add(int(part))
    return indexes


def build_smoke_proposals(query: QueryItem) -> list[dict]:
    """Use GT boxes as fake proposals for fast wiring tests without loading LocateAnything."""
    proposals = []
    for box in query.gt_boxes:
        proposals.append(
            {
                "tag": len(proposals) + 1,
                "bbox": [int(v) for v in box],
                "source": "gt_for_smoke",
                "phrase": "gt_for_smoke",
            }
        )
    return proposals


def select_first_proposal(proposals: list[dict]) -> dict:
    """Deterministic selector for debugging proposal/evaluation code paths."""
    if not proposals:
        return {"selected_tag": None, "bbox": None, "raw_response": "no proposals"}
    first = proposals[0]
    return {
        "selected_tag": first["tag"],
        "bbox": first["bbox"],
        "raw_response": "first_proposal selector",
    }


def evaluate_prediction(pred_box, gt_boxes, iou_threshold: float) -> tuple[float, bool]:
    """Return the best IoU against all GT boxes and whether it passes the threshold."""
    max_iou = 0.0
    if pred_box and gt_boxes:
        for gt_box in gt_boxes:
            max_iou = max(max_iou, calculate_iou(pred_box, gt_box))
    return max_iou, max_iou >= iou_threshold


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_add_eagle_to_path(path: str | None) -> None:
    """Expose Eagle/Embodied so LocateAnything can be imported from the cloned repo."""
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="VRBS LocateAnything tag-selection evaluation")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--small-lm-model", default=DEFAULT_SMALL_LM_MODEL)
    parser.add_argument("--small-lm-device", default="auto")
    parser.add_argument(
        "--small-lm-isolated-deps",
        default=DEFAULT_SMALL_LM_ISOLATED_DEPS,
        help="Optional dependency path used by a subprocess when the current transformers cannot load the small LM",
    )
    parser.add_argument("--locateanything-model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--locateanything-repo", default=DEFAULT_EAGLE_EMBODIED)
    parser.add_argument("--locateanything-device", default="cuda")
    parser.add_argument("--locateanything-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--hf-endpoint", default=None, help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--disable-hf-xet", action="store_true", help="Set HF_HUB_DISABLE_XET=1 before loading HF models")
    parser.add_argument("--proposal-source", default="locateanything", choices=["locateanything", "gt_for_smoke"])
    parser.add_argument("--selector-source", default="vlm", choices=["vlm", "first_proposal"])
    parser.add_argument("--query-indexes", default=None, help="Comma/range list, e.g. 1,3,5-7")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-proposals", type=int, default=12)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--force-phase1", action="store_true")
    parser.add_argument("--generation-mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--locate-temperature", type=float, default=0.2)
    parser.add_argument("--vlm-tensor-parallel-size", type=int, default=2)
    parser.add_argument("--vlm-max-model-len", type=int, default=32768)
    parser.add_argument("--phase1-max-tokens", type=int, default=2048)
    parser.add_argument("--select-max-tokens", type=int, default=2048)
    args = parser.parse_args()

    # Hugging Face access is configured before any transformers/vLLM modules are loaded.
    # This matters on the GPU server because direct HF access may fail while hf-mirror works.
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print(f"Using HF_ENDPOINT={args.hf_endpoint}")
    if args.disable_hf_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("Using HF_HUB_DISABLE_XET=1")

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    text_dir = output_dir / "text"
    image_dir = output_dir / "images"
    proposal_dir = output_dir / "proposals"
    for directory in [text_dir, image_dir, proposal_dir]:
        ensure_dir(directory)

    # Load VRBS metadata and match each q* folder with its question, images, and GT boxes.
    gt_annotations = load_gt_annotations(base_dir)
    queries = discover_queries(base_dir, gt_annotations)
    selected_indexes = parse_query_indexes(args.query_indexes)
    if selected_indexes is not None:
        queries = [q for q in queries if q.idx in selected_indexes]
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if not queries:
        raise RuntimeError(f"No query folders found under {base_dir}")

    print(f"Found {len(queries)} queries under {base_dir}")
    print(f"Output dir: {output_dir}")

    # Stage 1: turn each user question into a broad LocateAnything grounding phrase.
    # The original detailed command is preserved for VLM tag selection below.
    # By default this uses Qwen/Qwen3.5-0.8B downloaded to DEFAULT_SMALL_LM_MODEL.
    # If the model cannot load, CategoryExtractor falls back to deterministic rules.
    category_rows = []
    extractor = CategoryExtractor(
        args.small_lm_model,
        device=args.small_lm_device,
        isolated_deps_path=args.small_lm_isolated_deps,
    )
    categories = extractor.extract_many([query.question for query in queries])
    for query, category in zip(queries, categories):
        query.category = category.prompt
        category_rows.append(
            {
                "query_idx": query.idx,
                "question": query.question,
                "category": category.category,
                "prompt": category.prompt,
                "attributes": category.attributes,
                "source": category.source,
                "raw_response": category.raw_response,
            }
        )
    extractor.close()
    write_jsonl(text_dir / "category_predictions.jsonl", category_rows)

    # Stage 2: run LocateAnything on each query scene image with the phrase produced above.
    # The proposals are saved as JSON and rendered with visible numeric tags for VLM selection.
    maybe_add_eagle_to_path(args.locateanything_repo)
    locate_adapter = None
    if args.proposal_source == "locateanything":
        from src.locateanything_adapter import LocateAnythingAdapter

        locate_adapter = LocateAnythingAdapter(
            model_path=args.locateanything_model,
            repo_path=args.locateanything_repo,
            device=args.locateanything_device,
            dtype=args.locateanything_dtype,
        )

    for query in queries:
        if args.proposal_source == "gt_for_smoke":
            proposals = build_smoke_proposals(query)
            raw_answer = "gt_for_smoke"
        else:
            assert locate_adapter is not None
            proposals, raw_answer = locate_adapter.propose(
                image_path=query.scene_image,
                phrase=query.category,
                max_proposals=args.max_proposals,
                generation_mode=args.generation_mode,
                temperature=args.locate_temperature,
            )
        query.proposals = proposals
        query.locateanything_raw = raw_answer
        proposal_path = proposal_dir / f"q{query.idx}_proposals.json"
        dump_json(
            proposal_path,
            {
                "query_idx": query.idx,
                "question": query.question,
                "category_prompt": query.category,
                "raw_answer": raw_answer,
                "proposals": proposals,
            },
        )
        tagged_path = image_dir / f"q{query.idx}_tagged.jpg"
        render_tagged_image(query.scene_image, proposals, tagged_path)
        query.tagged_image = str(tagged_path)
        print(f"q{query.idx}: {len(proposals)} proposals, tagged image: {tagged_path}")

    if locate_adapter is not None:
        locate_adapter.close()

    # Stage 3: run the distilled VLM's first-round video understanding once, then reuse
    # the cached context while asking it to choose the correct proposal tag per query.
    selector = None
    if args.selector_source == "vlm":
        from src.vlm_selector import VLMTagSelector

        selector = VLMTagSelector(
            model_path=args.vlm_model,
            tensor_parallel_size=args.vlm_tensor_parallel_size,
            max_model_len=args.vlm_max_model_len,
            phase1_max_tokens=args.phase1_max_tokens,
            select_max_tokens=args.select_max_tokens,
        )
        video_path = get_video_path(base_dir)
        phase1_cache = text_dir / "phase1_video_context.txt"
        selector.run_phase1(video_path, cache_file=phase1_cache, force=args.force_phase1)

    # Stage 4: map the selected tag back to its bbox, draw evaluation images, and compute IoU.
    results = []
    hits = 0
    for query in queries:
        if args.selector_source == "first_proposal":
            selection = select_first_proposal(query.proposals)
        else:
            assert selector is not None
            selection = selector.select_tag(
                inquirer_image=query.human_image,
                tagged_scene_image=query.tagged_image,
                question=query.question,
                category_prompt=query.category,
                proposals=query.proposals,
                query_idx=query.idx,
                log_file=text_dir / "vlm_tag_selection.log",
            )
        pred_box = selection.get("bbox")
        max_iou, hit = evaluate_prediction(pred_box, query.gt_boxes, args.iou_threshold)
        if hit:
            hits += 1
        eval_image = image_dir / f"vis_q{query.idx}_{query.gt_label or 'unknown'}.jpg"
        draw_eval_image(query.scene_image, query.gt_boxes, pred_box, eval_image)
        result = {
            "query_idx": query.idx,
            "question": query.question,
            "category_prompt": query.category,
            "gt_label": query.gt_label,
            "gt_boxes": query.gt_boxes,
            "proposals": query.proposals,
            "selection": selection,
            "pred_box": pred_box,
            "max_iou": max_iou,
            "hit": hit,
            "eval_image": str(eval_image),
        }
        results.append(result)
        status = "HIT" if hit else "MISS"
        print(f"q{query.idx}: selected={selection.get('selected_tag')} max_iou={max_iou:.4f} {status}")

    precision = (hits / len(queries) * 100.0) if queries else 0.0
    summary = {
        "total": len(queries),
        "hits": hits,
        "precision": precision,
        "iou_threshold": args.iou_threshold,
        "results": results,
    }
    dump_json(text_dir / "results.json", summary)
    print("=" * 60)
    print(f"Total Correct: {hits} / {len(queries)}")
    print(f"Precision @ IoU={args.iou_threshold}: {precision:.2f}%")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
