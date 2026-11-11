"""Run the open models with the settings that produced the reported scores (vLLM offline).

Prompt construction, answer-type output limits, stop sequences, and temperature come from
run_eval_vllm_offline.py; this script only fixes the per-model settings below.

Common settings: strict prompt (one-line answer), temperature 0, repetition_penalty 1.05, prefix caching.

  Qwen3-8B      thinking disabled (enable_thinking=False); output limit per answer type;
                newline stop sequences removed (an empty <think> block ends with a newline);
                Lab-D prompts exceed the native 32K context, so YaRN (factor 4) with max_model_len 38,912.
  GPT-OSS-20B   reasoning cannot be disabled, so output limits are raised per level
                (L1/L2 2,048, L3 3,072, L4/L5 16,384); newline stop sequences removed.
  Llama-3.1-8B  output limit per answer type; newline stop sequences kept.
  Mistral3-8B   output limit per answer type; newline stop sequences kept; chat template rendered as a string.

    python run_open_models.py --model Qwen3-8B --labs LabA LabB LabC --dataset-dir <NetConfigQA2.1>
    python run_open_models.py --model Qwen3-8B --labs LabD --dataset-dir <NetConfigQA2.1>
Labs whose engine settings differ (Qwen3 Lab-D) must be run separately.
Output: results/<model>/<lab>.json, scored with `python score.py --dataset <NetConfigQA2.1>/data --pred results/<model>/*.json`.
Scores can differ slightly across GPUs and vLLM versions (version used: 0.16.0).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import run_eval_vllm_offline as ev

# vLLM 0.16 reads rope_parameters (rope_scaling is ignored). rope_theta is the Qwen3 config value.
YARN = {"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": 1000000}}

MODELS = {
    "Qwen3-8B": {
        "hf_path": "Qwen/Qwen3-8B",
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": "per_type",
        "keep_newline_stop": False,
        # Lab-D: longest prompt 36,455 tokens + 2,048 output limit
        "engine_by_lab": {"default": {"max_model_len": 32768},
                          "LabD": {"max_model_len": 38912, "hf_overrides": YARN}},
    },
    "GPT-OSS-20B": {
        "hf_path": "openai/gpt-oss-20b",
        "chat_template_kwargs": {},
        "max_tokens": {"L1": 2048, "L2": 2048, "L3": 3072, "L4": 16384, "L5": 16384},
        "keep_newline_stop": False,
        "engine_by_lab": {"default": {"max_model_len": 49152}},
    },
    "Llama-3.1-8B": {
        "hf_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "chat_template_kwargs": {},
        "max_tokens": "per_type",
        "keep_newline_stop": True,
        "engine_by_lab": {"default": {"max_model_len": 40960}},
    },
    "Mistral3-8B": {
        "hf_path": "mistralai/Ministral-3-8B-Instruct-2512",
        "chat_template_kwargs": {},
        "max_tokens": "per_type",
        "keep_newline_stop": True,
        # The reported run built the chat template as a string and tokenized it afterwards
        # (control tokens such as [SYSTEM_PROMPT] are tokenized as plain text); kept for reproducibility.
        "tokenize_chat_template": False,
        "engine_by_lab": {"default": {"max_model_len": 40960}},
    },
}
REPETITION_PENALTY = 1.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--labs", nargs="+", required=True, choices=["LabA", "LabB", "LabC", "LabD"])
    ap.add_argument("--dataset-dir", required=True, help="Released dataset root (data/Lab*.jsonl, labs/Lab*/configs/)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=None, help="Run only the first N questions per Lab (debugging)")
    args = ap.parse_args()

    ev.Config.DATASET_DIR = Path(args.dataset_dir).expanduser().resolve()
    settings = MODELS[args.model]
    engines = {json.dumps(settings["engine_by_lab"].get(lab, settings["engine_by_lab"]["default"]), sort_keys=True)
               for lab in args.labs}
    if len(engines) > 1:
        sys.exit("These Labs need different engine settings; run them separately.")
    engine = json.loads(engines.pop())

    from vllm import LLM, SamplingParams

    llm = LLM(model=settings["hf_path"], max_model_len=engine["max_model_len"], hf_overrides=engine.get("hf_overrides"),
              gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for lab in args.labs:
        key = lab[-1]
        rows = ev.read_dataset_rows(ev.find_latest_dataset(key))
        if args.limit:
            rows = rows[: args.limit]
        configs = ev.ConfigManager(ev.find_config_dir(key), ev.logging.getLogger("null")).get_all_configs()

        prompts, params, metas = [], [], []
        for row in rows:
            messages = ev.build_messages(row["question"], row["answer_type"], configs, "strict")
            if settings.get("tokenize_chat_template"):
                prompt = {"prompt_token_ids": tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, **settings["chat_template_kwargs"])}
            else:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                       **settings["chat_template_kwargs"])
            atype = ev.normalize_answer_type(row["answer_type"])
            if settings["max_tokens"] == "per_type":
                max_tokens = ev.resolve_max_tokens(row["level"], atype, None, decoding_mode="strict")
            else:
                max_tokens = settings["max_tokens"][row["level"]]
            stop = ev.build_stop_sequences(atype, decoding_mode="strict")
            if not settings["keep_newline_stop"]:
                stop = [s for s in stop if s not in ("\n", "\r\n")]
            prompts.append(prompt)
            params.append(SamplingParams(temperature=ev.Config.TEMPERATURE, repetition_penalty=REPETITION_PENALTY,
                                         max_tokens=max_tokens, stop=stop))
            metas.append((row, max_tokens))

        start = time.time()
        outputs = llm.generate(prompts, params)
        duration = time.time() - start

        results = []
        for (row, max_tokens), output in zip(metas, outputs):
            completion = output.outputs[0]
            results.append({
                "question_id": row["question_id"],
                "uid": row["uid"],
                "question": row["question"],
                "gold": row["answer"],
                "answer_status": row["answer_status"],
                "level": row["level"],
                "category": row["category"],
                "answer_type": row["answer_type"],
                "raw_pred": completion.text,
                "finish_reason": completion.finish_reason,
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "request_max_tokens": max_tokens,
            })
        meta = {
            "model": args.model, "hf_path": settings["hf_path"], "lab": lab, "n": len(results),
            "engine": {k: v for k, v in engine.items()}, "chat_template_kwargs": settings["chat_template_kwargs"],
            "max_tokens": settings["max_tokens"], "keep_newline_stop": settings["keep_newline_stop"],
            "temperature": ev.Config.TEMPERATURE, "repetition_penalty": REPETITION_PENALTY, "prompt": "strict",
            "vllm_version": __import__("vllm").__version__, "duration_sec": round(duration, 1),
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        path = out_dir / f"{lab}.json"
        path.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
        truncated = sum(r["finish_reason"] == "length" for r in results)
        print(f"[{args.model} {lab}] n={len(results)} truncated={truncated} time={duration:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
