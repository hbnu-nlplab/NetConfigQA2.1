"""Compute TA-Acc for model outputs on NetConfigQA 2.1.

Predictions: either
  - JSONL with one object per question, {"uid": "<lab>/<id>", "raw_output": "<model output>"}, or
  - result files written by run_eval_vllm_offline.py / run_eval_openai_batch.py (results_raw_*.json);
    several files (one per Lab) can be given.
Dataset:     data/LabA.jsonl ... LabD.jsonl of the released dataset.

Scoring (relaxed TA-Acc, as reported in the paper):
  - the answer is extracted with extract_answer.extract_answer
  - Set: F1, Map: key-value F1, Number: normalized exact match, Text: normalized exact match
  - questions whose eval_answer is NOT_CONFIGURED are correct if the prediction states that the
    feature is absent (NetConfigQAScorer.evaluate_negative_prediction)
TA-Acc is reported per Lab, per Level, and as the unweighted mean over Labs.

    python evaluation/score.py --dataset <dataset>/data --pred predictions.jsonl [--out per_question.jsonl]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import analyze_results as ar
from extract_answer import extract_answer


def score_one(scorer, item: dict, raw_output: str) -> float:
    status = "NOT_CONFIGURED" if item["eval_answer"] == "NOT_CONFIGURED" else "OK"
    atype = ar.canonical_answer_type(item["answer_type"])
    pred = scorer.clean_prediction(extract_answer(raw_output, item["answer_type"]), atype)
    if status == "NOT_CONFIGURED":
        return 1.0 if scorer.evaluate_negative_prediction(pred, item["answer"], atype)["semantic_negative_correct"] else 0.0
    return float(scorer.score(pred, item["answer"], atype, status=status, question_id=item["uid"]).get("score", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Directory with LabA.jsonl ... LabD.jsonl")
    ap.add_argument("--pred", required=True, nargs="+", help="Predictions JSONL ({uid, raw_output}) or results_raw_*.json files")
    ap.add_argument("--out", help="Optional per-question score JSONL")
    args = ap.parse_args()

    uids = {}  # (lab, 1-based position) -> uid
    for path in sorted(Path(args.dataset).glob("Lab*.jsonl")):
        for i, line in enumerate((l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()), 1):
            item = json.loads(line)
            uids[(item["lab"], i)] = item["uid"]

    preds = {}
    for pred_path in args.pred:
        text = Path(pred_path).read_text(encoding="utf-8")
        if pred_path.endswith(".json"):
            doc = json.loads(text)
            lab = doc["meta"]["lab"].replace("-", "")  # "Lab-A" -> "LabA"
            for r in doc["results"]:
                preds[uids[(lab, int(r["question_id"]))]] = r.get("raw_pred", "")
        else:
            for line in text.splitlines():
                if line.strip():
                    p = json.loads(line)
                    preds[p["uid"]] = p.get("raw_output", "")

    scorer = ar.NetConfigQAScorer()
    by_lab, by_level, rows = defaultdict(list), defaultdict(list), []
    for path in sorted(Path(args.dataset).glob("Lab*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["uid"] not in preds:
                continue
            s = score_one(scorer, item, preds[item["uid"]])
            by_lab[item["lab"]].append(s)
            by_level[item["level"]].append(s)
            rows.append({"uid": item["uid"], "level": item["level"], "score": s})

    if not rows:
        raise SystemExit("No prediction matched a dataset uid.")
    print(f"scored {len(rows)} / {len(preds)} predictions")
    for lab, v in sorted(by_lab.items()):
        print(f"{lab}: TA-Acc {sum(v) / len(v):.4f} (n={len(v)})")
    for lv, v in sorted(by_level.items()):
        print(f"{lv}: TA-Acc {sum(v) / len(v):.4f} (n={len(v)})")
    print(f"Avg over Labs: {sum(sum(v) / len(v) for v in by_lab.values()) / len(by_lab):.4f}")
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
