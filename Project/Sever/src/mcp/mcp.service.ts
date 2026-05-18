import { Injectable, Logger } from '@nestjs/common';
import {
  JsonRpcRequest,
  JsonRpcResponse,
  McpTool,
  ToolCallParams,
  HallucinationCheckInput,
  HallucinationCheckResult,
} from './mcp.types';

@Injectable()
export class McpService {
  private readonly logger = new Logger(McpService.name);

  // 등록된 도구 목록
  private readonly tools: McpTool[] = [
    {
      name: 'check_hallucination',
      description:
        'LLM 응답에서 환각(hallucination)을 탐지합니다. ' +
        '질문과 응답을 입력하면 신뢰도 점수와 함께 환각 여부를 반환합니다.',
      inputSchema: {
        type: 'object',
        properties: {
          question: {
            type: 'string',
            description: '사용자가 LLM에게 한 질문',
          },
          response: {
            type: 'string',
            description: '검사할 LLM 응답 텍스트',
          },
          context: {
            type: 'string',
            description: '(선택) 추가 맥락 정보',
          },
        },
        required: ['question', 'response'],
      },
    },
  ];

  handleRequest(req: JsonRpcRequest): JsonRpcResponse | null {
    this.logger.debug(`MCP 요청: ${req.method} (id=${req.id})`);

    switch (req.method) {
      case 'initialize':
        return this.handleInitialize(req);

      case 'notifications/initialized':
        // 클라이언트 초기화 완료 알림 - 응답 불필요
        return null;

      case 'ping':
        return { jsonrpc: '2.0', id: req.id, result: {} };

      case 'tools/list':
        return {
          jsonrpc: '2.0',
          id: req.id,
          result: { tools: this.tools },
        };

      case 'tools/call':
        return this.handleToolCall(req);

      default:
        return {
          jsonrpc: '2.0',
          id: req.id,
          error: {
            code: -32601,
            message: `지원하지 않는 메서드: ${req.method}`,
          },
        };
    }
  }

  private handleInitialize(req: JsonRpcRequest): JsonRpcResponse {
    return {
      jsonrpc: '2.0',
      id: req.id,
      result: {
        protocolVersion: '2024-11-05',
        capabilities: {
          tools: {},
        },
        serverInfo: {
          name: 'hallucination-detector',
          version: '1.0.0',
        },
      },
    };
  }

  private handleToolCall(req: JsonRpcRequest): JsonRpcResponse {
    const params = req.params as unknown as ToolCallParams;

    if (!params?.name) {
      return {
        jsonrpc: '2.0',
        id: req.id,
        error: { code: -32602, message: '도구 이름이 없습니다' },
      };
    }

    if (params.name === 'check_hallucination') {
      const input = params.arguments as unknown as HallucinationCheckInput;

      if (!input?.question || !input?.response) {
        return {
          jsonrpc: '2.0',
          id: req.id,
          error: { code: -32602, message: 'question, response는 필수입니다' },
        };
      }

      const result = this.checkHallucination(input);

      return {
        jsonrpc: '2.0',
        id: req.id,
        result: {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        },
      };
    }

    return {
      jsonrpc: '2.0',
      id: req.id,
      error: { code: -32602, message: `알 수 없는 도구: ${params.name}` },
    };
  }

  // TODO: 실제 환각 탐지 로직으로 교체 (Layer 1~3 cascade)
  private checkHallucination(input: HallucinationCheckInput): HallucinationCheckResult {
    this.logger.log(`환각 탐지 시작 - 응답 길이: ${input.response.length}자`);

    const suspicious = this.detectSuspiciousPatterns(input.response);
    const confidence = suspicious.length > 0 ? 0.65 + suspicious.length * 0.1 : 0.1;
    const isHallucination = confidence >= 0.5;

    return {
      is_hallucination: isHallucination,
      confidence: Math.min(confidence, 1.0),
      reason: isHallucination
        ? `의심스러운 패턴 ${suspicious.length}개 탐지됨`
        : '환각 패턴 없음',
      flagged_parts: suspicious.length > 0 ? suspicious : undefined,
    };
  }

  // 기본 휴리스틱 - 이후 LLM 기반 레이어로 교체
  // \b는 한국어에 동작하지 않으므로 단순 포함 검사로 처리
  private detectSuspiciousPatterns(text: string): string[] {
    const patterns: Array<{ check: (t: string) => boolean; label: string }> = [
      {
        check: (t) => /(항상|절대|반드시|100%|완전히)/.test(t),
        label: '과도한 확신 표현',
      },
      {
        check: (t) => /(모든 전문가|연구에 따르면|과학적으로 증명)/.test(t),
        label: '불분명한 출처 인용',
      },
      {
        check: (t) => /\d{4}년 \d{1,2}월 \d{1,2}일/.test(t),
        label: '구체적 날짜 언급',
      },
    ];

    return patterns.filter((p) => p.check(text)).map((p) => p.label);
  }
}
