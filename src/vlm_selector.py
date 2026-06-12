from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
from vllm import LLM, SamplingParams

from .data_utils import ensure_dir
from .prompts import PHASE1_SYSTEM, PHASE1_USER, TAG_SELECTION_PROMPT


class VLMTagSelector:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 2,
        max_model_len: int = 32768,
        phase1_max_tokens: int = 2048,
        select_max_tokens: int = 512,
    ):
        print(f"Loading VLM: {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 2, "video": 1},
            mm_processor_kwargs={"fps": 1, "do_sample_frames": True},
            allowed_local_media_path="/",
        )
        self.phase1_params = SamplingParams(
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=phase1_max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        self.select_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.15,
            max_tokens=select_max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        self.phase1_context = ""
        self.base_chat_history = [{"role": "system", "content": PHASE1_SYSTEM}]

    def run_phase1(self, video_path: str, cache_file: str | Path | None = None, force: bool = False) -> str:
        phase1_messages = [{"role": "system", "content": PHASE1_SYSTEM}]
        message = {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": f"file://{os.path.abspath(video_path)}"}},
                {"type": "text", "text": PHASE1_USER},
            ],
        }
        phase1_messages.append(message)

        cache_path = Path(cache_file) if cache_file else None
        if cache_path and cache_path.exists() and not force:
            response_text = cache_path.read_text(encoding="utf-8").strip()
            print(f"Loaded cached phase1 context: {cache_path}")
        else:
            print("Running phase1 video understanding...")
            response_text = self._generate(phase1_messages, self.phase1_params)
            if cache_path:
                ensure_dir(cache_path.parent)
                cache_path.write_text(response_text, encoding="utf-8")
        self.phase1_context = response_text
        self.base_chat_history = [
            {"role": "system", "content": PHASE1_SYSTEM},
            message,
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
        ]
        return response_text

    def select_tag(
        self,
        inquirer_image: str,
        tagged_scene_image: str,
        question: str,
        category_prompt: str,
        proposals: list[dict],
        query_idx: int,
        log_file: str | Path | None = None,
    ) -> dict:
        candidate_lines = "\n".join(
            f'{p["tag"]}: {p["bbox"]}' for p in proposals
        ) or "No candidate boxes."
        valid_tags = [int(p["tag"]) for p in proposals]
        full_prompt = TAG_SELECTION_PROMPT.format(
            question=question,
            category_prompt=category_prompt,
            candidate_lines=candidate_lines,
        )
        message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(inquirer_image)}"}},
                {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(tagged_scene_image)}"}},
                {"type": "text", "text": full_prompt},
            ],
        }
        # Match the 0606 evaluation script: copy only the phase1 chain and append
        # this query. The query/answer is never written back, so q1-q10 stay independent.
        messages = self.base_chat_history.copy()
        messages.append(message)
        response_text = self._generate(messages, self.select_params)
        selected_tag = _parse_selected_tag(response_text, valid_tags=valid_tags)
        forced_reason = None
        if selected_tag not in valid_tags:
            if len(valid_tags) == 1:
                selected_tag = valid_tags[0]
                forced_reason = "single_candidate_forced"
            elif valid_tags:
                selected_tag = valid_tags[0]
                forced_reason = "invalid_or_missing_vlm_tag_forced_first_candidate"
        bbox = None
        if selected_tag is not None:
            for proposal in proposals:
                if int(proposal["tag"]) == selected_tag:
                    bbox = proposal["bbox"]
                    break
        selection = {
            "selected_tag": selected_tag,
            "bbox": bbox,
            "raw_response": response_text,
        }
        if forced_reason:
            selection["forced_reason"] = forced_reason
        if log_file:
            log_path = Path(log_file)
            ensure_dir(log_path.parent)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- q{query_idx} ---\n")
                f.write(f"Question: {question}\n")
                f.write(f"Category: {category_prompt}\n")
                f.write(f"Candidates:\n{candidate_lines}\n")
                f.write(f"Response:\n{response_text}\n")
                f.write(f"Selected: {selected_tag}, bbox={bbox}, forced_reason={forced_reason}\n")
        return selection

    def _generate(self, messages, sampling_params: SamplingParams) -> str:
        outputs = self.llm.chat(messages=messages, sampling_params=sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


def _parse_selected_tag(text: str, valid_tags: list[int] | None = None) -> int | None:
    valid_set = set(valid_tags or [])
    clean = text.split("</think>")[-1].strip()
    match = re.search(r"```json\s*(\{.*?\})\s*```", clean, re.DOTALL | re.IGNORECASE)
    if match:
        clean = match.group(1)
    else:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)
    try:
        data = json.loads(clean.replace("'", '"'))
        tag = data.get("selected_tag")
        if tag is None:
            return None
        parsed = int(tag)
        if not valid_set or parsed in valid_set:
            return parsed
        return None
    except Exception:
        pass
    patterns = [
        r"selected[_\s-]*tag[^0-9]*(\d+)",
        r'"selected_tag"\s*:\s*(\d+)',
        r"\bchoose(?:s|n)?\s+(?:tag|box)?\s*(\d+)\b",
        r"\b(?:tag|box)\s*(\d+)\s*(?:is|seems|looks|matches|corresponds|contains)",
        r"\b(?:tag|box)\s*(\d+)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            parsed = int(match.group(1))
            if not valid_set or parsed in valid_set:
                return parsed
    return None
