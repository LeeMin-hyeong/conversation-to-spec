# Conversation-to-Spec v0.1.1: Source-Grounded Requirement Extraction and Verification

## Abstract

본 프로젝트는 비정형 대화를 구조화된 소프트웨어 요구사항으로 변환하는 로컬 파이프라인 `conversation-to-spec`를 개선한다. 기존 v0.1.0 파이프라인은 유효한 JSON을 생성할 수는 있었지만, 무라벨 대화 텍스트 안에서 기능 요구사항, 비기능 요구사항, 제약사항, 미해결 질문을 안정적으로 구분하는 데 한계가 있었다. 또한 evidence span, acceptance criteria, requirement-level verification metadata가 부족했다. 요구사항공학에서의 NLP 연구, LLM 기반 software specification generation 연구, grounded fact-checking 연구를 바탕으로 v0.1.1은 instruction-tuned local model, source-unit based extraction, deterministic specification construction, confidence-aware post-processing, MiniCheck-based claim-evidence verification을 중심으로 재설계되었다. v0.1.0과 v0.1.1은 동일한 30개 샘플 통합 데이터셋과 동일한 MLX 모델 `mlx-community/Qwen3-4B-Instruct-2507-4bit`을 사용하여 평가되었다. semantic requirement macro-F1은 0.691에서 0.827로 개선되었다. 가장 큰 category-level 개선은 constraint에서 나타났으며, constraint semantic F1은 0.455에서 0.731로 향상되었다. 평균 end-to-end latency는 39.52초에서 23.37초로 감소했고, 평균 LLM call 수는 샘플당 6.00회에서 4.83회로 감소했다. 결론적으로 개선된 파이프라인은 unmatched prediction rate를 낮게 유지하면서 recall과 category coverage를 개선하지만, 일부 생성 문장은 여전히 사람의 검토가 필요하다.

## Introduction

소프트웨어 요구사항은 종종 비공식적인 대화에서 처음 논의된다. 이러한 대화에는 결정사항, 질문, 제약, 가정, future-scope 항목이 섞여 있다. 이런 대화를 명확한 요구사항 명세로 변환하는 일은 유용하지만 어렵다. 시스템은 실제로 합의된 내용을 식별해야 하고, 질문을 요구사항으로 잘못 승격하지 않아야 하며, 기능 요구사항과 비기능 요구사항 및 제약사항을 구분해야 하고, 원본 대화로의 traceability를 보존해야 한다.

v0.1.0은 이 문제를 chain-style LLM pipeline으로 해결하려고 했다. 구조화된 출력은 생성할 수 있었지만, 산출물은 충분히 traceable하지 않았고 강한 verification support를 포함하지 않았다. 특히 v0.1.0은 acceptance criteria, evidence spans, requirement-level groundedness metadata를 생성하지 않았다. 또한 평가 데이터셋에서 constraint handling이 약하게 나타났다.

v0.1.1의 목표는 로컬 Apple Silicon 환경에서 실행 가능성을 유지하면서 생성된 요구사항 명세의 실용적 품질을 개선하는 것이다. 구체적인 목표는 다음과 같다.

- 기능 요구사항, 비기능 요구사항, 제약사항의 추출 품질 개선
- LLM call 수와 로컬 실행 시간 감소
- 생성 요구사항에 source-grounded evidence와 acceptance criteria 추가
- 생성 요구사항을 원본 대화에 대해 검증
- 사용자용 Markdown은 읽기 쉽게 유지하고, 상세 진단 정보는 JSON에 저장

GitHub repository: https://github.com/LeeMin-hyeong/conversation-to-spec

## Related Work

Necula, Dumitriu, and Greavu-Serban (2024)은 1991년부터 2023년까지 소프트웨어 요구사항공학에서 NLP가 어떻게 사용되었는지에 대한 systematic literature review를 제시한다. 이 논문은 요구사항공학이 elicitation, specification, modeling, validation을 포함하며, 자연어 요구사항은 ambiguity, incompleteness, inconsistency에 취약하다고 설명한다. 본 프로젝트에 적용 가능한 핵심은 요구사항 추출을 단순 요약으로 다루면 안 된다는 점이다. 요구사항 추출은 category-aware processing, ambiguity handling, traceability를 필요로 한다. 따라서 v0.1.1은 source-unit segmentation, requirement type separation, open-question handling, category별 semantic evaluation을 적용했다.

Xie et al. (2025)은 LLM이 software specification을 얼마나 효과적으로 생성할 수 있는지 연구했다. 이 연구는 일반적인 text generation이 아니라 LLM-based specification generation을 직접 다룬다는 점에서 본 프로젝트와 관련성이 높다. 핵심 교훈은 LLM이 specification generation에 도움을 줄 수 있지만, task-specific structure, careful evaluation, baseline comparison이 필요하다는 점이다. 이 관점은 본 프로젝트가 모델을 고정한 상태에서 v0.1.0과 v0.1.1 pipeline design을 비교하도록 동기화했다. 따라서 개선 결과는 단순히 더 강한 모델 때문이 아니라 pipeline-level improvement로 해석할 수 있다.

Vogelsang (2024)은 requirements engineering task에서 generative LLM을 사용하는 방식을 논의한다. 이 논문은 생성형 LLM을 요구사항공학 작업에 직접 prompting할 수 있지만, prompt quality와 output format control이 중요하다고 주장한다. 본 프로젝트에 적용 가능한 교훈은 local quantized model에게 완전한 nested specification을 자유롭게 생성하게 하는 것은 안정적이지 않다는 점이다. v0.1.1에서는 모델을 instruction-following extractor로 사용하고, 최종 specification은 deterministic code가 구성한다. 또한 base MLX 모델과 instruction-tuned MLX 모델을 비교하여 model policy를 평가했다. instruction-tuned model은 모든 F1 metric에서 우세하지는 않았지만, unmatched prediction rate와 runtime cost가 더 낮아 더 보수적인 기본값으로 선택되었다.

Tang, Laban, and Durrett (2024)은 LLM 출력이 source document에 근거하는지를 검증하는 작은 fact-checking model인 MiniCheck를 제안한다. 이 연구는 생성된 요구사항이 원본 대화에 의해 뒷받침될 때만 유용하다는 점에서 본 프로젝트와 직접적으로 관련된다. 적용 가능한 핵심은 generator output을 그대로 신뢰하지 않고, 생성된 claim을 evidence에 대해 검증해야 한다는 점이다. v0.1.1은 `verify_mode=minicheck`가 활성화된 경우 MiniCheck-based claim-evidence verification을 사용하고, requirement-level confidence와 verdict를 JSON report에 저장한다. 다만 v0.1.0에는 동등한 verification field가 없기 때문에, 이 지표들은 v0.1.0과 v0.1.1의 직접 extraction 성능 비교에는 사용하지 않았다. 대신 최종 specification의 reviewability를 높이는 기능 개선으로 해석한다.

Es et al. (2023)은 retrieval-augmented generation system을 context relevance, faithfulness, answer quality와 같은 차원에서 평가하는 RAGAS를 제안한다. 본 프로젝트는 full RAG system은 아니지만, 평가 철학은 유용하다. 생성 결과는 하나의 전체 점수만으로 평가하기보다 여러 진단 차원으로 나누어 평가해야 한다. 이 관점은 v0.1.1 평가 설계에 반영되었다. 최종 비교에서는 semantic extraction F1, unsupported prediction rate, latency, LLM calls, schema validity를 분리하여 평가한다. 또한 v0.1.1 전용 traceability field를 주요 비교 그래프에 포함하지 않았다. 이는 개선 폭을 과장하지 않기 위한 선택이다.

## Methods

### Data Preprocessing

입력은 무라벨 plain-text conversation이다. speaker label은 선택 사항이다. 시스템은 먼저 대화를 `U1`, `U2`, `U3`와 같은 source unit으로 분할한다. 생성된 각 요구사항은 이 source unit에 대한 참조를 유지한다. 이 설계는 요구사항이 원본 대화에 근거하는지 확인할 수 있게 한다.

v0.1.1에서는 원본 대화 텍스트를 하나의 긴 블록으로 처리하지 않는다. Source-unit 구조는 requirement extraction, evidence span construction, open-question detection, verification에 사용된다. Post-processing 단계에서도 인접 source unit을 사용하여 privacy prohibition, future-scope constraint, ambiguous pronoun을 정규화한다.

### Baseline Pipeline: v0.1.0

v0.1.0 pipeline은 multi-stage chain을 사용했다. Candidate extraction, classification, rewriting, open-question generation, follow-up generation, summarization을 여러 번의 LLM call로 수행했다. 이 방식은 비용이 크고 constraint extraction 측면에서 여전히 약했다.

### Proposed Pipeline: v0.1.1

v0.1.1 pipeline은 source-unit decision prompt 이후 deterministic construction과 post-processing을 수행한다. LLM은 source-grounded decision을 생성하지만, 최종 specification은 코드가 조립한다. 이 설계는 모델을 extraction과 classification에 집중시키고, 최종 schema, evidence link, 사용자용 Markdown 구조는 코드가 통제하도록 만든다.

### Verification and Post-processing

Deterministic construction 이후 pipeline은 각 요구사항에 evidence span과 acceptance criteria를 추가한다. 그 다음 `verify_mode=minicheck`가 활성화되면 MiniCheck가 main verifier로 사용된다. Verification에서는 각 생성 요구사항을 claim으로 보고, 해당 요구사항이 참조하는 source unit들의 텍스트를 이어 붙인 내용을 evidence로 사용한다. MiniCheck-based verifier는 claim이 evidence에 의해 support되는지를 추정하고, 그 결과를 requirement-level verdict와 confidence value로 JSON에 저장한다.

Verification 이후 pipeline은 low-confidence item이나 구조적으로 약한 item에 대해 confidence-aware cleanup을 수행한다. 여기에는 answered question, privacy prohibition, future-scope constraint, ambiguous pronoun 처리가 포함된다.

### Improvements over v0.1.0

Table 1은 v0.1.0 baseline에서 v0.1.1로 넘어오며 적용된 주요 기능 개선을 요약한다. 이 개선들은 모두 직접적인 성능 지표는 아니다. 일부는 생성된 specification을 사람이 더 쉽게 검토하기 위한 output 및 reviewability feature이다.

| Area | v0.1.0 baseline | v0.1.1 improvement | Expected effect |
| --- | --- | --- | --- |
| Pipeline architecture | 반복적인 LLM call을 사용하는 multi-stage chain | Source-unit decision + deterministic construction | LLM call 수 및 latency 감소 |
| Model policy | HF 중심 local model configuration | Apple Silicon MLX backend + instruction-tuned model preference | 로컬 실행 효율 및 instruction following 안정성 개선 |
| Requirement typing | future/deferred item 처리 약함 | deferred-scope statement에 대한 constraint normalization | constraint detection 개선 |
| Traceability | source unit reference 중심 | source unit + evidence span | human review 용이성 개선 |
| Acceptance criteria | 일관되게 생성되지 않음 | enrichment 단계에서 Given/When/Then-style criteria 추가 | 더 실용적인 requirements draft 생성 |
| Groundedness verification | requirement-level verifier output 없음 | MiniCheck-based support probability와 verdict를 JSON에 저장 | 약하게 grounded된 요구사항 식별 가능 |
| Diagnostic output | Markdown과 debug output 분리가 약함 | 사용자용 Markdown + 상세 `spec.json`, `verification_report.json`, `debug/spec/summary.json` | 명세 가독성과 재현 가능한 분석 데이터 확보 |
| Post-processing | 제한적인 cleanup | weak item, answered question, privacy prohibition, future constraint에 대한 confidence-aware cleanup | local-model artifact 감소 |

### Experimental Setup

두 버전은 동일한 모델로 평가되었다.

```yaml
qwen3_4b_mlx_4bit:
  repo_id: mlx-community/Qwen3-4B-Instruct-2507-4bit
  backend: mlx
```

이 모델은 main comparison에서 Apple Silicon 실행 가능성과 instruction-following behavior를 고려해 선택되었다. 모델을 고정한 이유는 v0.1.0에서 v0.1.1로의 개선이 model upgrade가 아니라 pipeline change 때문인지 보기 위해서다.

비교 실험은 `dataset/eval_samples.json`, `dataset/eval_booking.json`, `dataset/eval_services.json`, `dataset/eval_ops.json`를 합친 30개 샘플 통합 평가 데이터셋으로 수행했다. Pipeline 차이를 분리하기 위해 v0.1.0과 v0.1.1 모두 동일한 모델 `mlx-community/Qwen3-4B-Instruct-2507-4bit`을 사용했다. v0.1.0은 원래 MLX runner를 포함하지 않았기 때문에, detached v0.1.0 worktree에 temporary compatibility runner만 추가했다. v0.1.0 generation pipeline 자체는 변경하지 않았다.

최종 comparable evaluation artifact는 다음 경로에 저장되어 있다.

```text
experiments/version_compare/report_all_mlx_qwen3_4b_instruct_comparable/
```

추가로 v0.1.1 pipeline에서 base MLX model과 instruction-tuned MLX model을 비교하는 secondary model-policy experiment를 수행했다.

```text
experiments/version_compare/report_v011_instruct_vs_base_mlx/
```

보고서에서 사용하는 주요 graph file은 `docs/assets/project#2/` 아래에 저장되어 있다.

- `semantic_extraction_f1.png`
- `constraint_f1.png`
- `avg_latency_sec.png`
- `avg_llm_calls.png`
- `instruct_vs_base_semantic_f1.png`
- `instruct_vs_base_semantic_unmatched.png`

### Evaluation Metrics

주요 평가는 requirement type별 semantic F1을 사용한다. Functional requirements, non-functional requirements, constraints는 lightweight semantic token overlap과 source-unit overlap을 사용하여 gold item과 matching된다. Macro-F1은 세 semantic F1 score의 평균이다. RAGAS식 평가 철학을 따라, extraction quality, groundedness risk, runtime cost, schema validity를 하나의 global score로 합치지 않고 별도 차원으로 보고한다.

Unsupported prediction rate는 별도로 측정했다. `semantic_unmatched_prediction_rate`는 예측 요구사항이 어떤 gold requirement와도 semantic match되지 않을 때만 unsupported로 계산한다. 이는 paraphrase를 hallucination으로 잘못 해석하는 문제를 줄이기 위한 것이다. Source-aware unmatched rate는 추가로 source-unit overlap을 요구한다. Type mismatch는 별도로 보고한다. 의미적으로 유효한 요구사항이 잘못된 category로 분류된 경우는 hallucination이 아니라 classification error이기 때문이다.

Semantic F1 metric은 lightweight하고 project-specific한 지표이므로, 일반 benchmark score가 아니라 내부 비교 지표로 해석해야 한다. MiniCheck verification metadata는 주로 v0.1.1 산출물의 reviewability와 diagnostic analysis에 사용하며, v0.1.0과 v0.1.1의 primary extraction score로 직접 사용하지 않는다.

## Results

### Main Comparison: v0.1.0 Baseline vs v0.1.1 Proposed Pipeline

Table 2는 v0.1.0과 v0.1.1의 comparable evaluation result를 요약한다.

| Version | Pipeline | Semantic Macro F1 | FR F1 | NFR F1 | CON F1 | Semantic Unmatched | Source-aware Unmatched | Predictions | Latency | LLM Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.1.0 | chain | 0.691 | 0.862 | 0.755 | 0.455 | 0.047 | 0.075 | 106 | 39.52s | 6.00 |
| v0.1.1 | source-unit decision | 0.827 | 0.938 | 0.814 | 0.731 | 0.029 | 0.066 | 137 | 23.37s | 4.83 |

가장 큰 category-level improvement는 constraint detection에서 나타났다. 더 큰 평가셋에서 v0.1.0의 constraint F1은 0.455였고, v0.1.1은 0.731이었다. 이는 초기 6개 샘플 결과보다 더 보수적이고 신뢰할 수 있는 결과다. 이 결과는 v0.1.1이 future-scope 및 deferred-scope item을 constraint로 더 잘 감지한다는 것을 보여준다. 다만 이것이 constraint wording이 완벽하다는 뜻은 아니다. Non-functional requirement F1은 0.755에서 0.814로 완만하게 개선되었고, functional requirement F1은 0.862에서 0.938로 개선되었다.

Semantic macro-F1은 0.691에서 0.827로 개선되었다. 이는 개선이 특정 category 하나에만 국한되지 않았음을 보여준다. 동시에 모든 평가 데이터셋을 포함한 30개 샘플 기준이므로, 개선 폭은 초기 소규모 실험보다 더 현실적이다.

### Runtime Efficiency

Runtime도 개선되었다. 평균 end-to-end latency는 샘플당 39.52초에서 23.37초로 감소했다. 평균 LLM call 수는 샘플당 6.00회에서 4.83회로 감소했다. 이는 multi-stage chain을 source-unit decision과 deterministic post-processing 기반의 더 compact한 pipeline으로 대체한 설계가 타당했음을 보여준다. v0.1.1도 enrichment, summarization, verification-related step에서 추가 LLM call을 사용하지만, v0.1.0의 반복적인 extraction, classification, rewrite chain은 줄였다.

### Unsupported Prediction Analysis

Hallucination 관련 지표는 조심스럽게 해석해야 한다. Semantic unmatched prediction rate는 0.047에서 0.029로 감소했고, source-aware unmatched rate는 0.075에서 0.066으로 감소했다. 이 변화는 크지 않으므로, hallucination이 극적으로 감소했다고 주장하는 것은 부적절하다. 더 안전한 해석은 v0.1.1이 더 많은 요구사항을 예측하면서도 unsupported-prediction rate를 비슷하거나 약간 낮은 수준으로 유지했다는 것이다.

다음 그림은 extraction quality와 runtime efficiency의 주요 변화를 시각화한다.

![Semantic extraction F1](assets/project_2/semantic_extraction_f1.png)

![Constraint F1](assets/project_2/constraint_f1.png)

![Average latency](assets/project_2/avg_latency_sec.png)

![Average LLM calls](assets/project_2/avg_llm_calls.png)

### Secondary Ablation: Base vs Instruction-Tuned Model

강의에서 fine-tuning과 instruction tuning을 다루므로, v0.1.1 기준으로 official base-style MLX model `Qwen/Qwen3-4B-MLX-4bit`과 instruction-tuned MLX model `mlx-community/Qwen3-4B-Instruct-2507-4bit`을 추가 비교했다. Pipeline과 30개 샘플 데이터셋은 동일하게 고정했다.

| Model | Semantic Macro F1 | FR F1 | NFR F1 | CON F1 | Semantic Unmatched | Source-aware Unmatched | Latency | LLM Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B MLX base | 0.829 | 0.954 | 0.794 | 0.741 | 0.057 | 0.086 | 24.30s | 4.93 |
| Qwen3-4B Instruct MLX | 0.827 | 0.938 | 0.814 | 0.731 | 0.029 | 0.066 | 23.37s | 4.83 |

이 결과는 instruction-tuned model이 모든 extraction F1 metric에서 우세하다는 것을 보여주지는 않는다. Base model은 semantic macro-F1과 functional F1에서 근소하게 높다. 그러나 instruction-tuned model은 semantic unmatched prediction rate와 source-aware unmatched rate가 더 낮고, latency와 LLM call 수도 약간 낮다. 따라서 instruction-tuned model은 모든 점수를 이기기 때문에 선택된 것이 아니라, requirements assistant에 더 보수적이고 운영상 안정적인 output profile을 제공하기 때문에 선택되었다.

![Instruct vs base semantic F1](assets/project_2/instruct_vs_base_semantic_f1.png)

![Instruct vs base semantic unmatched](assets/project_2/instruct_vs_base_semantic_unmatched.png)

## Discussion

결과는 v0.1.1이 본 프로젝트 목표에 대해 v0.1.0보다 의미 있는 개선을 달성했음을 보여준다. 가장 중요한 점은 같은 모델을 사용했기 때문에 개선이 단순히 모델이 좋아져서 발생한 것이 아니라는 점이다. 개선은 pipeline design에서 온다. Source-unit decision format, deterministic construction, confidence-aware post-processing이 출력 안정성과 evaluation category alignment를 개선했다.

관련 연구도 이러한 방향을 뒷받침한다. NLP4RE 연구는 요구사항공학에서 ambiguity, incompleteness, semantic processing의 필요성을 강조한다. 이는 source-unit segmentation과 category-specific evaluation으로 이어졌다. LLM-based software specification 연구는 고정된 모델 아래에서 pipeline을 통제해 비교해야 한다는 필요성을 보여주었다. Generative LLM 기반 요구사항공학 연구는 prompt design과 output control의 중요성을 강조한다. 이는 controlled source-unit decision schema와 instruction-tuned model policy로 반영되었다. MiniCheck는 evidence-based verification을 직접적으로 동기화했다. v0.1.1에서는 생성 요구사항을 claim으로 보고 source evidence에 대해 검증한 뒤, diagnostic verdict를 JSON에 저장한다. RAGAS는 하나의 global score 대신 여러 평가 차원을 분리하는 평가 방식을 동기화했다.

비교 결과는 중요한 trade-off도 보여준다. v0.1.1은 v0.1.0보다 더 많은 요구사항을 예측한다. 전체 predicted requirement count는 106에서 137로 증가했다. 이는 특히 constraint에서 recall과 category coverage를 높인다. 동시에 semantic unmatched와 source-aware unmatched rate는 낮은 수준으로 유지된다. Requirements engineering assistant 관점에서, 생성 Markdown을 human review를 위한 draft로 다루고 JSON verification metadata를 함께 제공한다면 이 trade-off는 수용 가능하다. 그러나 이 시스템이 사람의 요구사항 분석을 완전히 대체한다고 주장해서는 안 된다.

한계도 존재한다. 30개 샘플은 초기 6개 샘플 비교보다 강하지만, 실제 requirements engineering benchmark에 비하면 여전히 작은 규모다. 따라서 결과는 course-project evidence로 해석해야 하며, 일반화된 성능 증명으로 보기는 어렵다. 또 다른 한계는 semantic matcher가 lightweight token-based 방식이라는 점이다. 통제된 프로젝트 평가에는 유용하지만, 향후에는 embedding-based matching 또는 LLM-judge-based semantic matching과 비교할 필요가 있다. 마지막으로 MiniCheck confidence는 약하게 grounded된 요구사항을 식별하는 데 도움을 주지만, 완전한 requirements validation method는 아니다. 예를 들어 MiniCheck는 요구사항이 source evidence에 의해 support되는지는 확인할 수 있지만, 그 요구사항이 complete한지, feasible한지, stakeholder priority와 align되는지는 판단할 수 없다.

## Conclusions

본 프로젝트는 `conversation-to-spec`을 v0.1.0에서 v0.1.1로 개선하면서 multi-stage chain pipeline을 source-unit decision pipeline으로 변경했다. 여기에 deterministic construction, instruction-tuned MLX inference, evidence enrichment, acceptance criteria generation, MiniCheck-based verification을 추가했다. 동일한 모델과 30개 샘플 데이터셋 기준으로 semantic macro-F1은 0.691에서 0.827로 개선되었다. Constraint F1은 0.455에서 0.731로 개선되었고, 평균 LLM call 수는 6.00에서 4.83으로 감소했다.

핵심 메시지는 pipeline structure가 model choice만큼 중요하다는 것이다. Local quantized model도 source-grounded decision, deterministic post-processing, verification으로 task를 분해하면 더 나은 requirements specification을 생성할 수 있다. 향후 연구에서는 데이터셋을 확장하고, source-aware matching을 개선하며, 더 강한 coreference resolution을 추가하고, 실제 stakeholder conversation으로 시스템을 평가해야 한다.

## References

- Necula, S.-C., Dumitriu, F., and Greavu-Serban, V. (2024). A Systematic Literature Review on Using Natural Language Processing in Software Requirements Engineering. Electronics, 13(11), 2055. https://doi.org/10.3390/electronics13112055
- Xie, D., Yoo, B., Jiang, N., Kim, M., Tan, L., Zhang, X., and Lee, J. S. (2025). How Effective are Large Language Models in Generating Software Specifications? Proceedings of the 32nd IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER), 1-12. https://doi.org/10.1109/SANER64311.2025.00014
- Vogelsang, A. (2024). Prompting the Future: Integrating Generative LLMs and Requirements Engineering. Joint Proceedings of REFSQ-2024 Workshops. https://ceur-ws.org/Vol-3672/NLP4RE-keynote1.pdf
- Tang, L., Laban, P., and Durrett, G. (2024). MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. Proceedings of EMNLP 2024, 8818-8847. https://aclanthology.org/2024.emnlp-main.499/
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). Ragas: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217. https://arxiv.org/abs/2309.15217
