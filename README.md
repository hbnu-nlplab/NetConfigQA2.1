# NetConfigQA 2.1: A Benchmark for Network Configuration Understanding

Code to regenerate the NetConfigQA 2.1 configurations and QA pairs and to score model outputs with TA-Acc.

```
config_generator/          Lab-B–D configuration generation
  topologies/lab_{b,c,d}_*.yaml    topology definitions (nodes, links, addressing, protocols, injected faults)
  templates/*.j2                   Jinja2 templates per device role (PE, P, ASBR, leaf)
  generator.py
dataset_generation/        QA generation (Batfish-based) and L1–L3 validation
  policies.json                    126 metrics (L1–L5): level, category, answer type, question template, scope
  src/main_batfish.py              entry point
  src/core_batfish/                facts extraction, L1–L3 answers (builder_core.py), L4 (l4_analyzer.py), L5 (l5_analyzer.py)
  src/verification/                independent regex parser (no Batfish) and comparison for L1–L3 answers
evaluation/                Model inference, answer extraction, TA-Acc scoring
  run_open_models.py               open models (vLLM) with the settings used for the reported scores
  run_eval_openai_batch.py         API models (OpenAI-compatible endpoint, e.g. OpenRouter)
  run_eval_vllm_offline.py         shared module: prompt, output limits, stop sequences, data loading
  extract_answer.py                answer extraction used for all reported scores
  analyze_results.py               type-aware scorer (NetConfigQAScorer)
  score.py                         TA-Acc from a predictions file
```

Lab-A configurations were exported from the emulated devices and are included in the dataset; Lab-B to Lab-D were generated with `config_generator/`.

## Requirements

Python 3.10+. `pip install -r requirements.txt`.
Dataset generation needs a Batfish server: `docker run -d -p 9997:9997 -p 9996:9996 batfish/batfish`.
Open-model inference needs a GPU with vLLM (version used: 0.16.0).

API keys: copy `.env.example` to `.env` and fill in the keys you need (`OPENAI_API_KEY` or `OPENROUTER_API_KEY` for API models, `HF_TOKEN` for gated Hugging Face models).
`run_eval_openai_batch.py` reads `.env` automatically; for `run_open_models.py`, export the variables in the shell (e.g. `set -a; source .env; set +a`).
Dataset generation, validation, and scoring need no keys.

## 1. Generate configurations (Lab-B–D)

```bash
python config_generator/generator.py --topology config_generator/topologies/lab_b_20nodes.yaml
# -> config_generator/output/<lab>/configs/*.cfg
```

## 2. Generate QA pairs

The lab directory must contain `configs/*.cfg`.

```bash
cd dataset_generation
python src/main_batfish.py --lab-path <lab_dir> --policies policies.json --batfish-host localhost --seed 42 --question-lang en
# -> <lab_dir>/Dataset/<timestamp>/*_dataset_batfish_*_en.{json,csv} and the extracted facts
```

Sampling rules (`src/main_batfish.py`, `src/core_batfish/batfish_builder.py`):

- Each metric in `policies.json` is expanded over its scope (`GLOBAL`, `DEVICE`, `DEVICE_PAIR`, `AS`, `OSPF_AREA`, `VRF`, ...).
  If a metric has more than 20 instances, 20 are sampled uniformly at random.
- L4 source–destination pairs and L5 failure scenarios are sampled from the Batfish topology after shuffling.
- All random choices use a fixed seed (42).

## 3. Validate L1–L3 answers with the independent parser

```bash
cd dataset_generation
python src/verification/run_verification.py --configs <lab_dir>/configs --dataset <dataset.json>
```

## 4. Run models

`--dataset-dir` is the root of the released dataset (it contains `data/Lab*.jsonl` and `labs/Lab*/configs/`).
Every prompt contains all device configurations of the Lab, the question, and the answer type (strict prompt, one-line answer).
All models use temperature 0; the open models also use repetition penalty 1.05.

| Model        | Command                                                                 | Settings                                                                                     |
| ------------ | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| GPT-4o mini  | `run_eval_openai_batch.py --model gpt-4o-mini --decoding-mode strict` | OpenRouter `openai/gpt-4o-mini`; output limit per answer type                              |
| Llama-3.1-8B | `run_open_models.py --model Llama-3.1-8B`                             | output limit per answer type; newline stop kept                                              |
| Mistral3-8B  | `run_open_models.py --model Mistral3-8B`                              | output limit per answer type; newline stop kept                                              |
| Qwen3-8B     | `run_open_models.py --model Qwen3-8B`                                 | thinking disabled; output limit per answer type; newline stop removed; Lab-D with YaRN (×4) |
| GPT-OSS-20B  | `run_open_models.py --model GPT-OSS-20B`                              | output limit per level (L1/L2 2,048, L3 3,072, L4/L5 16,384); newline stop removed           |

Output limits per answer type: number 32, text 512, set/map 2,048 tokens.

```bash
cd evaluation
python run_open_models.py --model Qwen3-8B --labs LabA LabB LabC --dataset-dir <NetConfigQA2.1>
python run_open_models.py --model Qwen3-8B --labs LabD --dataset-dir <NetConfigQA2.1>   # different engine settings (YaRN)
python run_open_models.py --model GPT-OSS-20B --labs LabA LabB LabC LabD --dataset-dir <NetConfigQA2.1>
# -> results/<model>/Lab<X>.json

python run_eval_openai_batch.py --model gpt-4o-mini --lab A B C D --decoding-mode strict --dataset-dir <NetConfigQA2.1>
# -> results_2/GPT-4o-mini/Lab<X>/results_raw_vllm_en_<timestamp>.json
```

Scores can differ slightly from the reported values: API outputs are not fully deterministic, and open-model outputs depend on the GPU and vLLM version (version used: 0.16.0).

## 5. Score model outputs

Predictions are either the result files from step 4 (`results/<model>/*.json`, `results_2/.../results_raw_*.json`) or a JSONL file with one line per question: `{"uid": "LabA/<id>", "raw_output": "<model output>"}`.

```bash
cd evaluation
python score.py --dataset <NetConfigQA2.1>/data --pred results/Qwen3-8B/*.json
python score.py --dataset <NetConfigQA2.1>/data --pred predictions.jsonl
```

Scoring rules:

- The answer is extracted from the raw output with `extract_answer.py` (reasoning blocks removed, first JSON array/object for Set/Map, first number for Number, first line for Text).
- Set: F1 between predicted and gold sets. Map: key-value F1. Number: numeric normalization and exact match. Text: normalized exact match (`analyze_results.NetConfigQAScorer`).
- Questions whose `eval_answer` is `NOT_CONFIGURED` count as correct when the prediction states that the feature is absent.
- TA-Acc is the mean question score per Lab; the overall value is the unweighted mean over the four Labs.

## Dataset

[huggingface.co/datasets/hbnunlp/NetConfigQA2.1](https://huggingface.co/datasets/hbnunlp/NetConfigQA2.1)


## Citation
