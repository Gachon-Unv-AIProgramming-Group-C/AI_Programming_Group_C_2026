import { Test, TestingModule } from '@nestjs/testing';
import { McpService } from './mcp.service';
import { HallucinationCheckInput } from './mcp.types';

describe('McpService', () => {
  let service: McpService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [McpService],
    }).compile();

    service = module.get<McpService>(McpService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('checkHallucination (3-Layer Cascade Mock Mode)', () => {
    it('should pass Layer 1 when logprobs are high confidence (L1 score < t1)', async () => {
      const input: HallucinationCheckInput = {
        question: 'What is the capital of South Korea?',
        response: 'Seoul',
        logprobs: [
          { token: 'Seoul', logprob: -0.01 },
          { token: 'is', logprob: -0.01 },
          { token: 'the', logprob: -0.01 },
          { token: 'capital', logprob: -0.01 }
        ],
        t1: 0.3,
        w1: 1.0,
      };

      const result = await (service as any).checkHallucination(input);
      expect(result.is_hallucination).toBe(false);
      expect(result.details.layersRun).toContain(1);
      expect(result.details.layersRun).not.toContain(2);
      expect(result.details.score1).toBeLessThan(input.t1!);
    });

    it('should pass Layer 2 when SINdex indicates high consistency (score2 < s2_threshold)', async () => {
      const input: HallucinationCheckInput = {
        question: 'What is the capital of South Korea?',
        response: 'Seoul',
        t1: 0.3,
        s2_threshold: 0.9, // high threshold to guarantee pass
        m: 3,
      };

      const result = await (service as any).checkHallucination(input);
      expect(result.is_hallucination).toBe(false);
      expect(result.details.layersRun).toContain(1);
      expect(result.details.layersRun).toContain(2);
      expect(result.details.layersRun).not.toContain(3);
    });

    it('should proceed to Layer 3 and classify when SINdex is inconsistent', async () => {
      const input: HallucinationCheckInput = {
        question: 'An inconsistent question.',
        response: 'response',
        s2_threshold: 0.01, // low threshold to force Layer 3
        t2: 0.5,
        m: 3,
      };

      const result = await (service as any).checkHallucination(input);
      expect(result.details.layersRun).toContain(1);
      expect(result.details.layersRun).toContain(2);
      expect(result.details.layersRun).toContain(3);
      expect(result.is_hallucination).toBeDefined();
    });

    it('should use client-side sampling if no API keys are present but sampling capability is enabled', async () => {
      // Mock client capabilities
      (service as any).clientCapabilities = { sampling: {} };

      // Mock writeCallback to intercept requests and reply as if we are the client
      const sentRequests: any[] = [];
      service.registerWriteCallback((msg) => {
        sentRequests.push(msg);
        
        // Asynchronously reply to the request
        setTimeout(() => {
          if (msg.method === 'sampling/createMessage') {
            // For NLI probability: return high similarity
            if (msg.params.systemPrompt && msg.params.systemPrompt.includes('Natural Language Inference')) {
              service.handleRequest({
                jsonrpc: '2.0',
                id: msg.id,
                result: {
                  role: 'assistant',
                  content: {
                    type: 'text',
                    text: '0.9'
                  }
                }
              });
            } else {
              // For paraphrasing or other LLM calls
              service.handleRequest({
                jsonrpc: '2.0',
                id: msg.id,
                result: {
                  role: 'assistant',
                  content: {
                    type: 'text',
                    text: JSON.stringify(['Paraphrased question A', 'Paraphrased question B'])
                  }
                }
              });
            }
          }
        }, 10);
      });

      const input: HallucinationCheckInput = {
        question: 'This is a question to test the sampling capability.',
        response: 'Test answer',
        s2_threshold: 0.01, // force Layer 3
        t1: 0.3,
        tStar: 0.7,
        m: 3,
      };

      const result = await (service as any).checkHallucination(input);
      expect(sentRequests.length).toBeGreaterThan(0);
      expect(sentRequests[0].method).toBe('sampling/createMessage');
      expect(result.is_hallucination).toBeDefined();
    });
  });
});

