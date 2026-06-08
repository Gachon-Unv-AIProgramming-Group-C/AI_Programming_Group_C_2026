<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=cylinder&color=gradient&height=120&text=H3Ilo%20w0rL6&animation=&fontColor=ffffff&fontSize=70" />
</div>

<div align="center">
  <strong>LLM 할루시네이션 탐지 시스템</strong><br/>
  가천대학교 스마트보안학과 AI 프로그래밍 2026 — Group C
</div>

<br/>

## 프로젝트 개요

LLM 응답에서 할루시네이션을 자동으로 탐지하는 3단계 캐스케이드 시스템입니다.
**"Verify when Uncertain"** (arXiv:2502.15845) 논문을 기반으로 MCP(Model Context Protocol) 서버로 구현했습니다.

## 아키텍처

```
사용자 질문 + LLM 응답
        │
        ▼
┌──────────────────────────────────────────────────────┐
│              3-Layer Cascade Detection               │
│                                                      │
│  Layer 1 — LSC (Lowest Span Confidence)              │
│  logprob 기반 스팬 신뢰도 측정 → 낮으면 L2로           │
│        │                                             │
│        ▼                                             │
│  Layer 2 — SINdex (Semantic Inconsistency Index)     │
│  동일 질문 다중 샘플링 → 클러스터링 → 분산 측정          │
│        │                                             │
│        ▼                                             │
│  Layer 3 — SAC³ (Cross-model Consistency)            │
│  질문 paraphrase → 독립 verifier 교차 검증             │
└──────────────────────────────────────────────────────┘
        │
        ▼
HALLUCINATION / NO_HALLUCINATION / UNCERTAIN
```

## 디렉토리 구조

```
Project/
├── Client/              # React + TypeScript 프론트엔드
├── Sever/               # NestJS MCP 서버 + Python NLI 서버
└── experiments/         # 실험 데이터셋 및 평가 스크립트
```

## 빠른 시작

```bash
# 서버
cd Project/Sever && cp .env.example .env && docker-compose up

# 클라이언트
cd Project/Client && cp .env.example .env && npm install && npm run dev

# 실험
cd Project/experiments && pip install -r requirements.txt && python run_experiment.py --limit 10
```

## 모델

| 모델 | 역할 |
|------|------|
| `serize/klue-roberta-base-nli-stage2` | Layer 3 NLI 스코어링 |
| `serize/local-qwen-paraphraser` | Layer 3 질문 paraphrase |

![Repobeats](https://repobeats.axiom.co/api/embed/9aab525420c65c37a77e23be9860edecd272db0f.svg)
