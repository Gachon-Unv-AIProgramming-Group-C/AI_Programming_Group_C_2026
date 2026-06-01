export interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: number | string | null;
  method: string;
  params?: Record<string, unknown>;
}

export interface JsonRpcResponse {
  jsonrpc: '2.0';
  id: number | string | null;
  result?: unknown;
  error?: JsonRpcError;
}

export interface JsonRpcError {
  code: number;
  message: string;
  data?: unknown;
}

export interface McpTool {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
  };
}

export interface McpServerInfo {
  name: string;
  version: string;
}

export interface McpCapabilities {
  tools?: Record<string, never>;
}

export interface ToolCallParams {
  name: string;
  arguments: Record<string, unknown>;
}

export interface HallucinationCheckInput {
  question: string;
  response: string;
  context?: string;
  targetModel?: string;
  verifierModel?: string;
  nliModel?: string;
  m?: number;
  logprobs?: { token: string; logprob: number }[];
  t1?: number;
  tStar?: number;
  s2_threshold?: number;
  t2?: number;
  w1?: number;
  w2?: number;
  w3?: number;
  useHuggingFaceNli?: boolean;
  uncertainty_margin?: number;
}

export interface HallucinationCheckResult {
  verdict: 'HALLUCINATION' | 'NO_HALLUCINATION' | 'UNCERTAIN';
  is_hallucination: boolean;
  confidence: number;
  reason: string;
  flagged_parts?: string[];
  details?: {
    stage?: number;
    layersRun: number[];
    mode?: string;
    score1?: number;
    score2?: number;
    score3?: number;
    final_score: number;
    selfMpd?: number;
    crossMpd?: number;
    combinedMpd?: number;
    targetSamples?: string[];
    verifierSamples?: string[];
    clusters?: { label: string; size: number; members: string[]; similarityToOriginal: number }[];
    paraphrasedQuestions?: string[];
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


