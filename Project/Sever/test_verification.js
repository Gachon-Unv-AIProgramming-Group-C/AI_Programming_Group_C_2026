const { spawn } = require('child_process');
const path = require('path');

// 1. 검증할 노션 기반 샘플 데이터셋 정의
const testSamples = [
  {
    id: "FA-009 (명백한 사실)",
    question: "대한민국의 수도는 어디인가",
    response: "대한민국의 수도는 서울입니다.",
    expected: "NO_HALLUCINATION (Factual)"
  },
  {
    id: "RE-004 (가짜 최신정보)",
    question: "2026년 4월에 일어난 성남 지진의 규모는 몇 도였나?",
    response: "2026년 4월 성남에서 발생한 지진은 규모 3.4로 기록되었습니다. 다행히 인명 피해는 없었습니다.",
    expected: "HALLUCINATION (가짜 뉴스 작작)"
  },
  {
    id: "RE-002 (진짜 최신정보)",
    question: "대만 화롄에서 규모 7.2 지진이 발생한 적은 언제인가?",
    response: "대만 화롄 지역에서 규모 7.2의 강진이 발생한 시점은 2024년 4월 2일(현지 시각 3일)입니다.",
    expected: "NO_HALLUCINATION (Factual)"
  },
  {
    id: "RE-010 (가짜 최신정보)",
    question: "2024년에 호주에서 캥거루 무리와 코알라 무리가 맞붙었는데 어느 무리가 이겼는가?",
    response: "2024년 호주 산불 대피 도중 캥거루와 코알라 무리 사이에 격렬한 영역 다툼이 있었고, 결국 힘이 더 센 캥거루 무리가 영역을 차지하며 승리했습니다.",
    expected: "HALLUCINATION (가짜 대결 사건)"
  },
  {
    id: "EX-002 (전문지식)",
    question: "모듈러 역원(modular inverse)이 존재하기 위한 조건은 무엇인가?",
    response: "어떤 정수 a의 모듈러 n에 대한 역원이 존재하기 위한 필요충분조건은 a와 n이 서로소(gcd(a, n) = 1)인 것입니다.",
    expected: "NO_HALLUCINATION (Factual)"
  },
  {
    id: "PA-001 (가짜 논문 인용)",
    question: "Kim et al. (2024)의 'Hallucination-Free LLMs via Knowledge Anchoring' 논문의 핵심 기여는 무엇인가?",
    response: "이 논문에서 Kim 등은 외부 knowledge graph를 LLM 추론 과정에 동적으로 연결하는 KA-Attention 메커니즘을 제안했으며, GPT-4 대비 할루시네이션 발생률을 63% 감소시켰다고 보고했습니다. arXiv:2401.09821에 게재됐습니다.",
    expected: "HALLUCINATION (존재하지 않는 논문)"
  },
  {
    id: "CA-001 (수식 계산)",
    question: "직각삼각형에서 두 변의 길이가 각각 3과 4일 때, 빗변의 길이는?",
    response: "피타고라스 정리에 의해 빗변 c = √(3² + 4²) = √(9 + 16) = √25 = 5입니다.",
    expected: "NO_HALLUCINATION (올바른 계산)"
  }
];

// API 호출 함수
async function runTest(sample) {
  const payload = {
    jsonrpc: "2.0",
    id: Date.now(),
    method: "tools/call",
    params: {
      name: "check_hallucination",
      arguments: {
        question: sample.question,
        response: sample.response,
        useHuggingFaceNli: true // 직접 학습시킨 커스텀 NLI 모델 사용 강제
      }
    }
  };

  try {
    const res = await fetch('http://127.0.0.1:8000/mcp', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
      throw new Error(`HTTP Error: ${res.status}`);
    }

    const data = await res.json();
    return data.result?.content?.[0]?.text 
      ? JSON.parse(data.result.content[0].text) 
      : data;
  } catch (err) {
    return { error: err.message };
  }
}

// 메인 러너 함수
async function main() {
  console.log("==================================================");
  console.log("🚀 노션 검증 데이터 기반 할루시네이션 탑지 테스트 시작");
  console.log("🔧 커스텀 NLI 모델: serize/klue-roberta-small-nli");
  console.log("==================================================\n");

  for (const sample of testSamples) {
    console.log(`[ID] ${sample.id}`);
    console.log(`[Q] ${sample.question}`);
    console.log(`[A] ${sample.response}`);
    console.log(`[Expected] ${sample.expected}`);
    
    console.log("⏳ 검증 진행 중...");
    const result = await runTest(sample);
    
    if (result.error) {
      console.log(`❌ 검증 실패: ${result.error}\n`);
    } else {
      const isHall = result.is_hallucination;
      console.log(`👉 [검증 결과] ${isHall ? "🚨 HALLUCINATION 감지됨" : "✅ Factual (정상)"}`);
      console.log(`👉 [신뢰도] ${(result.confidence * 100).toFixed(2)}%`);
      console.log(`👉 [이유] ${result.reason}`);
      if (result.details && result.details.verifierSamples) {
        console.log(`👉 [Verifier Answers]\n` + result.details.verifierSamples.map(x => `  - ${x}`).join("\n"));
      }
      console.log("--------------------------------------------------\n");
    }
  }
}

main();
