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

  describe('checkHallucination (Mock Mode)', () => {
    it('should pass Stage 1 when self-consistency is very high (Self MPD < t1)', async () => {
      const input: HallucinationCheckInput = {
        question: 'What is the capital of South Korea?',
        response: 'Seoul',
        t1: 0.5,
        tStar: 0.9,
        m: 3,
      };
      
      const result = await (service as any).checkHallucination(input);
      expect(result.is_hallucination).toBe(false);
      expect(result.details.stage).toBe(1);
      expect(result.details.selfMpd).toBeLessThan(input.t1);
    });

    it('should fail Stage 1 when self-consistency is very low (Self MPD > tStar)', async () => {
      const input: HallucinationCheckInput = {
        question: 'What is the size of the universe?',
        response: 'The size of the universe is unknown.',
        t1: 0.01,
        tStar: 0.1,
        m: 3,
      };

      const result = await (service as any).checkHallucination(input);
      expect(result.is_hallucination).toBe(true);
      expect(result.details.stage).toBe(1);
      expect(result.details.selfMpd).toBeGreaterThan(input.tStar);
    });

    it('should proceed to Stage 2 when Self MPD is in the uncertainty interval [t1, tStar]', async () => {
      const input: HallucinationCheckInput = {
        question: 'Purposefully asking an ambiguous consistency question.',
        response: 'This is a test response.',
        t1: 0.1,
        tStar: 0.6,
        t2: 0.5,
        m: 4,
      };

      const result = await (service as any).checkHallucination(input);
      expect(result.details.stage).toBe(2);
      expect(result.details.selfMpd).toBeGreaterThanOrEqual(input.t1);
      expect(result.details.selfMpd).toBeLessThanOrEqual(input.tStar);
      expect(result.details.crossMpd).toBeDefined();
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
            // For sub-questions generation: return standard JSON array
            if (msg.params.systemPrompt && msg.params.systemPrompt.includes('verification question generator')) {
              service.handleRequest({
                jsonrpc: '2.0',
                id: msg.id,
                result: {
                  role: 'assistant',
                  content: {
                    type: 'text',
                    text: JSON.stringify(['Verification question A', 'Verification question B'])
                  }
                }
              });
            } else if (msg.params.systemPrompt && msg.params.systemPrompt.includes('Natural Language Inference')) {
              // For NLI probability: return high similarity
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
              // For other LLM questions: return factual answer
              service.handleRequest({
                jsonrpc: '2.0',
                id: msg.id,
                result: {
                  role: 'assistant',
                  content: {
                    type: 'text',
                    text: 'This is the answer to the verification question.'
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

