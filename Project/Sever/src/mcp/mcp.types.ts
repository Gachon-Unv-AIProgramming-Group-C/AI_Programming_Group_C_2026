// JSON-RPC 2.0 요청 형식. Claude가 MCP 서버에 보내는 모든 요청이 이 형식을 따른다
export interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: number | string | null;    // 요청과 응답을 매칭하기 위한 식별자
  method: string;                // 호출할 메서드 이름 (예: "tools/call", "ping")
  params?: Record<string, unknown>;
}

// JSON-RPC 2.0 응답 형식. 서버가 Claude에게 돌려주는 형식
export interface JsonRpcResponse {
  jsonrpc: '2.0';
  id: number | string | null;
  result?: unknown;      // 성공 시 반환값
  error?: JsonRpcError;  // 실패 시 에러 정보
}

// 에러 발생 시 반환되는 형식
export interface JsonRpcError {
  code: number;     // 에러 코드 (-32601: 미지원 메서드, -32602: 잘못된 파라미터 등)
  message: string;
  data?: unknown;
}

// MCP 도구(tool) 하나의 명세. tools/list 요청 시 Claude에게 반환된다
export interface McpTool {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, unknown>;  // 각 입력 파라미터의 타입과 설명
    required?: string[];                   // 필수 파라미터 목록
  };
}

// MCP 서버 정보 (initialize 응답에 포함)
export interface McpServerInfo {
  name: string;
  version: string;
}

// 서버가 지원하는 기능 목록 (initialize 응답에 포함)
export interface McpCapabilities {
  tools?: Record<string, never>;
}

// tools/call 요청의 파라미터 형식
export interface ToolCallParams {
  name: string;                          // 호출할 도구 이름 (예: "check_hallucination")
  arguments: Record<string, unknown>;    // 도구에 전달할 인자
}

// check_hallucination 도구의 입력 파라미터 정의
export interface HallucinationCheckInput {
  question: string;     // 사용자가 LLM에 물어본 질문
  response: string;     // 검증할 LLM의 응답 텍스트
  context?: string;     // 선택적 추가 맥락 정보

  targetModel?: string;     // 샘플 생성에 사용할 LLM (기본: gpt-4o-mini)
  verifierModel?: string;   // 검증에 사용할 LLM (기본: gpt-4o-mini)
  nliModel?: string;        // NLI 추론에 사용할 모델 ID

  m?: number;               // Layer 2에서 생성할 샘플 수 (기본: 5)
  logprobs?: { token: string; logprob: number }[];  // Layer 1 LSC용 토큰별 로그확률 (OpenAI 전용)

  // 각 계층의 판정 임계값
  t1?: number;           // Layer 1 조기 종료 하한 임계값 (기본: 0.3)
  tStar?: number;        // Layer 1 강제 진행 상한 임계값 (기본: 0.7)
  s2_threshold?: number; // Layer 2 통과 임계값 (기본: 0.4)
  t2?: number;           // 최종 판정 임계값 (기본: 0.5)

  // 최종 점수 계산 시 각 레이어 점수의 가중치
  w1?: number;  // Layer 1 가중치 (기본: 0.2)
  w2?: number;  // Layer 2 가중치 (기본: 0.3)
  w3?: number;  // Layer 3 가중치 (기본: 0.5)

  useHuggingFaceNli?: boolean;  // HuggingFace API를 NLI 추론에 사용할지 여부
  uncertainty_margin?: number;  // UNCERTAIN 판정 마진 (기본: 0.05)
}

// check_hallucination 도구의 반환값 정의
export interface HallucinationCheckResult {
  verdict: 'HALLUCINATION' | 'NO_HALLUCINATION' | 'UNCERTAIN';  // 최종 판정
  is_hallucination: boolean;
  confidence: number;  // 판정에 대한 신뢰도 (0~1)
  reason: string;      // 판정 이유 설명 문자열
  flagged_parts?: string[];

  // 디버깅 및 분석용 상세 정보
  details?: {
    stage?: number;          // 몇 번째 계층에서 판정이 났는지
    layersRun: number[];     // 실행된 계층 번호 목록 (예: [1, 2, 3])
    mode?: string;           // 동작 모드 (llm / local-python / mock 등)

    score1?: number;  // Layer 1 LSC 점수
    score2?: number;  // Layer 2 SINdex 점수
    score3?: number;  // Layer 3 SAC³ 점수
    final_score: number;  // 가중 합산 최종 점수

    selfMpd?: number;        // Layer 2 자기 불일치 점수
    crossMpd?: number;       // Layer 3 교차 불일치 점수
    combinedMpd?: number;    // 두 점수의 평균

    dispersion?: number;           // 클러스터 분산도 (Shannon 엔트로피 정규화)
    nliInconsistency?: number;     // NLI 모순율 (1 - 평균 entailment 점수)
    majorityDisagreement?: number; // 다수 클러스터와 원본의 불일치 정도

    targetSamples?: string[];      // Layer 2에서 생성된 샘플 응답 목록
    verifierSamples?: string[];    // Layer 3에서 생성된 검증 응답 목록

    // 클러스터 정보: Agglomerative Clustering으로 생성된 각 클러스터의 정보
    clusters?: { label: string; size: number; members: string[]; similarityToOriginal: number }[];

    paraphrasedQuestions?: string[];  // Layer 3에서 생성된 패러프레이즈 질문 목록

    // 실제 사용된 임계값과 가중치 (디버깅용)
    thresholdsUsed: {
      t1: number;
      tStar: number;
      s2_threshold: number;
      t2: number;
      m: number;
    };
    weightsUsed: {
      w1: number;
      w2: number;
      w3: number;
    };
  };
}
