# Conversation-to-Spec (Python-only, Local HF)

Conversation-to-Spec is a prototype that converts **unlabeled English client-developer conversations** into a structured software requirements draft.

It is designed for junior PMs and student project leads who need to transform vague conversations into actionable requirements before development starts.

## Why no speaker labels?

Real transcripts often come from ASR/STT and may not include `Client:` / `Developer:` labels.  
This project does not require explicit speaker tags. It segments conversation text into trace units (`U1`, `U2`, ...), then extracts and rewrites requirements using semantic context.

## Pipeline

The project uses a multi-stage chain pipeline:

1. Segment transcript into conversation units (`U1`, `U2`, ...)
2. Stage 1: candidate extraction (recall-oriented)
3. Stage 2: candidate classification (`functional_requirement`, `non_functional_requirement`, `constraint`, `open_question`, `follow_up_trigger`, `note`, `discard`)
4. Stage 3: requirement rewriting (spec-style wording)
5. Stage 4: open question generation
6. Stage 5: follow-up question generation
7. Stage 6: project summary generation
8. Deterministic final assembly into `SpecOutput`
9. Semantic verification and warning generation
10. Save outputs and debug artifacts

Each LLM stage uses:
- strict JSON prompting
- parse + lightweight repair
- stage validation
- stage-only retry (`max_retries` configurable)

Artifacts:
- `output/spec.json`
- `output/spec.md`
- `output/debug/<basename>/...` stage-level raw/repaired/error logs + summary

## Output schema

`SpecOutput` includes:
- `project_summary`
- `functional_requirements`
- `non_functional_requirements`
- `constraints`
- `open_questions`
- `follow_up_questions`
- `notes`
- `conversation_units`
- `verification_warnings`

Each requirement/question/note carries `source_units` for traceability.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Single input:

```bash
python -m app.main --input samples/sample_cafe_website.txt --output output --model gemma_3_1b_it
python -m app.main --input samples/sample_cafe_website.txt --output output --mock
```

Evaluation:

```bash
python -m app.main --evaluate --dataset dataset/eval_samples.json --model gemma_3_1b_it
python -m app.main --evaluate --dataset dataset/eval_samples.json --all-models
python -m app.main --evaluate --dataset dataset/eval_samples.json --mock
```

## Model switching

Edit `configs/models.yaml`:
- `models.<alias>.hf_repo_id`
- `compare_models` for `--all-models`
- shared generation settings in `generation`

Default comparison set is mixed sizes:
- `google/gemma-3-1b-it`
- `Qwen/Qwen2.5-3B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

If a large model fails due local resources, evaluation records the failure and continues for other models.

## Evaluation metrics

Implemented metrics:
- Functional requirement Precision / Recall / F1
- Non-functional requirement Precision / Recall / F1
- Constraint Precision / Recall / F1
- Requirement Type Macro-F1 (functional / non-functional / constraint / open-question / follow-up / note)
- Open Question Recall
- Follow-up Question Coverage
- Hallucination Rate
- Schema Validity Rate (JSON parse + Pydantic)
- JSON parse success rate
- Pydantic validation success rate
- Retry success rate
- Final usable output rate
- Semantic warning rate
- Stage 1 average candidate count
- Stage 2 average discard rate
- Stage 4 average open question count
- Stage 5 average follow-up question count
- Stage failure counts
- Average latency per sample
- Normalized (lowercase/punctuation-normalized) F1 for functional, non-functional, and constraints

## Testing

```bash
pytest
```

Covered:
- segmentation behavior
- extraction/repair behavior
- retry + failure-safe pipeline behavior
- semantic verification downgrade behavior
- markdown formatter
- mock pipeline run + failure handling
- evaluation metric calculations

## Current limitations

- Each stage uses a single LLM generation plus repair/retry; there is no iterative self-critique loop.
- Exact-match metrics are strict and can under-score semantically similar wording.
- Local model quality and speed strongly depend on GPU/VRAM.
