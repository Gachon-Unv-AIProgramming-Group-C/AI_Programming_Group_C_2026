# Experiments

## 파일 구조

```
experiments/
├── dataset.csv          # 실험용 Q&A 데이터셋 (30개 샘플)
├── run_experiment.py    # 배치 실험 실행 스크립트
├── evaluate.ipynb       # 결과 분석 노트북
├── requirements.txt     # Python 의존성
└── results/             # 실험 결과 저장 (자동 생성)
```

## 실행

```bash
pip install -r requirements.txt
python run_experiment.py
python run_experiment.py --limit 5 --model gpt-4o-mini
python run_experiment.py --ids 1 2 3
MCP_URL=http://192.168.1.10:8000/mcp python run_experiment.py
```

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `gpt-4o-mini` | 사용할 LLM |
| `--samples` | `5` | SINdex 샘플 수 |
| `--t1` | `0.3` | L1 통과 임계값 |
| `--t-star` | `0.7` | L1 강제 진행 임계값 |
| `--t2` | `0.5` | 최종 판정 임계값 |
| `--delay` | `1.0` | 요청 간 대기 시간(초) |
| `--limit` | 전체 | 최대 샘플 수 |
| `--ids` | 전체 | 특정 id만 실행 |
