// MCP JSON-RPC 2.0 기본 타입

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

// tools/call 파라미터
export interface ToolCallParams {
  name: string;
  arguments: Record<string, unknown>;
}

// check_hallucination 도구 입력
export interface HallucinationCheckInput {
  question: string;
  response: string;
  context?: string;
}

// 환각 탐지 결과
export interface HallucinationCheckResult {
  is_hallucination: boolean;
  confidence: number;
  reason: string;
  flagged_parts?: string[];
}
