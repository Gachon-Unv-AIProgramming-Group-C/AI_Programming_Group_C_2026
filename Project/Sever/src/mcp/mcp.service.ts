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
  private clientCapabilities: any = null;
  private writeCallback?: (msg: any) => void;
  private readonly pendingRequests = new Map<
    string | number,
    { resolve: (value: any) => void; reject: (reason?: any) => void }
  >();

  // List of registered tools
  private readonly tools: McpTool[] = [
    {
      name: 'check_hallucination',
      description:
        'Verify when Uncertain (Beyond Self-Consistency) algorithm is used to detect hallucinations in LLM responses. ' +
        'It dynamically combines self-consistency and cross-model consistency in a 2-stage cascade process.',
      inputSchema: {
        type: 'object',
        properties: {
          question: {
            type: 'string',
            description: 'The question asked to the LLM',
          },
          response: {
            type: 'string',
            description: 'The LLM response text to verify',
          },
          context: {
            type: 'string',
            description: '(Optional) Additional context information',
          },
          targetModel: {
            type: 'string',
            description: '(Optional) Target model name (default: gpt-4o-mini)',
          },
          verifierModel: {
            type: 'string',
            description: '(Optional) Verifier model name (default: gpt-4o-mini)',
          },
          m: {
            type: 'integer',
            description: '(Optional) Number of samples to generate (default: 5)',
          },
          t1: {
            type: 'number',
            description: '(Optional) Stage 1 lower threshold (default: 0.3)',
          },
          tStar: {
            type: 'number',
            description: '(Optional) Stage 1 upper threshold (default: 0.7)',
          },
          t2: {
            type: 'number',
            description: '(Optional) Stage 2 cross-threshold (default: 0.5)',
          },
          useHuggingFaceNli: {
            type: 'boolean',
            description: '(Optional) Whether to use Hugging Face NLI API (default: false)',
          },
        },
        required: ['question', 'response'],
      },
    },
  ];

  async handleRequest(req: any): Promise<JsonRpcResponse | null> {
    if (req.result !== undefined || req.error !== undefined) {
      // Handle response from the client
      this.handleResponse(req);
      return null;
    }

    this.logger.debug(`MCP Request: ${req.method} (id=${req.id})`);

    switch (req.method) {
      case 'initialize':
        return this.handleInitialize(req);

      case 'notifications/initialized':
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
        return await this.handleToolCall(req);

      default:
        return {
          jsonrpc: '2.0',
          id: req.id,
          error: {
            code: -32601,
            message: `Unsupported method: ${req.method}`,
          },
        };
    }
  }

  private handleInitialize(req: JsonRpcRequest): JsonRpcResponse {
    this.clientCapabilities = req.params?.capabilities;
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

  private async handleToolCall(req: JsonRpcRequest): Promise<JsonRpcResponse> {
    const params = req.params as unknown as ToolCallParams;

    if (!params?.name) {
      return {
        jsonrpc: '2.0',
        id: req.id,
        error: { code: -32602, message: 'Tool name is missing' },
      };
    }

    if (params.name === 'check_hallucination') {
      const input = params.arguments as unknown as HallucinationCheckInput;

      if (!input?.question || !input?.response) {
        return {
          jsonrpc: '2.0',
          id: req.id,
          error: { code: -32602, message: 'question and response are required' },
        };
      }

      try {
        const result = await this.checkHallucination(input);

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
      } catch (error) {
        return {
          jsonrpc: '2.0',
          id: req.id,
          error: {
            code: -32603,
            message: `Error during hallucination check: ${error.message}`,
          },
        };
      }
    }

    return {
      jsonrpc: '2.0',
      id: req.id,
      error: { code: -32602, message: `Unknown tool: ${params.name}` },
    };
  }

  // ── Verify when Uncertain (Beyond Self-Consistency) 2-Stage Cascade Algorithm ──
  // Option A: Cross-Examination Agent Method Applied
  private async checkHallucination(input: HallucinationCheckInput): Promise<HallucinationCheckResult> {
    const targetModel = input.targetModel || 'gpt-4o-mini';
    const verifierModel = input.verifierModel || 'gpt-4o-mini';
    const nliModel = input.nliModel || targetModel;
    const m = Math.min(input.m || 5, 10);
    const t1 = input.t1 !== undefined ? input.t1 : 0.3;
    const tStar = input.tStar !== undefined ? input.tStar : 0.7;
    const t2 = input.t2 !== undefined ? input.t2 : 0.5;
    const useHf = !!input.useHuggingFaceNli;

    this.logger.log(`Starting hallucination check - Question: "${input.question}" (Option A: Cross-Examination Agent, NLI Model: ${nliModel})`);

    // Check if API keys are present or client-side sampling is supported
    const hasKeys = this.hasApiKeys();
    const useSampling = this.canUseSampling();
    const canRunRealLlm = hasKeys || useSampling;

    // Stage 1: Generate sub-verification questions (generate m-1 to test m response states)
    const subQuestions = await this.generateSubQuestions(input.question, input.response, m, targetModel, canRunRealLlm);
    this.logger.log(`Generated verification sub-questions: ${JSON.stringify(subQuestions)}`);

    // ── Stage 1: Ask target model to answer sub-verification questions and compare with the original response ──
    const targetAnswers: string[] = [];
    if (canRunRealLlm) {
      for (const sq of subQuestions) {
        const prompt = `Question: "${sq}"\nProvide a direct, concise, and factual answer in one sentence.`;
        const ans = await this.callLLM(targetModel, prompt, 0.1);
        targetAnswers.push(ans);
      }
    } else {
      this.logger.warn('API keys are not set. Running in Mock mode.');
      const isCapital = input.question.includes('수도') || input.question.toLowerCase().includes('capital');
      const isUniverse = input.question.includes('우주') || input.question.toLowerCase().includes('universe');
      
      for (let i = 0; i < subQuestions.length; i++) {
        if (isCapital) {
          // High consistency (mocking identical response)
          targetAnswers.push(input.response);
        } else if (isUniverse) {
          // Low consistency (mocking contradiction)
          targetAnswers.push(`Contradictory universe response version ${i + 1}`);
        } else {
          // Moderate consistency
          if (i % 2 === 0) {
            targetAnswers.push(input.response);
          } else {
            targetAnswers.push('Another mock response with a different opinion.');
          }
        }
      }
    }

    // Stage 1 Entailment Evaluation: Does the original response entail each sub-question answer? E(response, targetAnswer_j)
    const selfScores: number[] = [];
    for (let i = 0; i < subQuestions.length; i++) {
      const score = await this.getEntailment(input.response, targetAnswers[i], useHf, hasKeys, nliModel);
      selfScores.push(score);
    }

    // Self MPD = 1 - Mean(SelfScores)
    const sumSelf = selfScores.reduce((acc, val) => acc + val, 0);
    const selfMpd = selfScores.length > 0 ? 1 - (sumSelf / selfScores.length) : 0;
    
    // Process samples for logging and display
    const targetSamples = subQuestions.map((q, idx) => `Q: ${q} -> A: ${targetAnswers[idx]}`);

    this.logger.log(`Stage 1 (Self-Consistency) finished - Self MPD: ${selfMpd.toFixed(4)} (Threshold range: t1=${t1}, t*=${tStar})`);

    // Stage 1 Decision Check
    if (selfMpd < t1) {
      return {
        is_hallucination: false,
        confidence: 1 - selfMpd,
        reason: `[Stage 1 Pass] Self-consistency divergence (Self MPD: ${selfMpd.toFixed(4)}) is below the lower threshold (t1: ${t1}). The original response entails the sub-question answers correctly.`,
        details: {
          stage: 1,
          selfMpd,
          targetSamples,
          thresholdsUsed: { t1, tStar, t2, m },
        },
      };
    }

    if (selfMpd > tStar) {
      return {
        is_hallucination: true,
        confidence: selfMpd,
        reason: `[Stage 1 Detected] Self-consistency divergence (Self MPD: ${selfMpd.toFixed(4)}) exceeds the upper threshold (t*: ${tStar}). There is a large inconsistency between the original response and sub-question answers.`,
        details: {
          stage: 1,
          selfMpd,
          targetSamples,
          thresholdsUsed: { t1, tStar, t2, m },
        },
      };
    }

    // ── Stage 2: Ask verification questions to external Verifier model since it is in the uncertainty interval ──
    this.logger.log(`Entered uncertainty interval: ${t1} <= Self MPD(${selfMpd.toFixed(4)}) <= ${tStar}. Starting Stage 2 cross-model verification.`);

    const verifierAnswers: string[] = [];
    if (canRunRealLlm) {
      for (const sq of subQuestions) {
        const prompt = `Question: "${sq}"\nProvide a direct, concise, and factual answer in one sentence.`;
        const ans = await this.callLLM(verifierModel, prompt, 0.1);
        verifierAnswers.push(ans);
      }
    } else {
      // Generate mock data (matching parts of the target answers to mimic 50% cross-consistency)
      for (let i = 0; i < subQuestions.length; i++) {
        if (i % 2 === 0) {
          verifierAnswers.push(targetAnswers[i]);
        } else {
          verifierAnswers.push(`Different verifier mock response ${i + 1}`);
        }
      }
    }

    // Stage 2 Cross Entailment Evaluation: Does the verifier response entail the target response? E(verifierAnswer_j, targetAnswer_j)
    const crossScores: number[] = [];
    for (let i = 0; i < subQuestions.length; i++) {
      const score = await this.getEntailment(verifierAnswers[i], targetAnswers[i], useHf, hasKeys, nliModel);
      crossScores.push(score);
    }

    // Cross MPD = 1 - Mean(CrossScores)
    const sumCross = crossScores.reduce((acc, val) => acc + val, 0);
    const crossMpd = crossScores.length > 0 ? 1 - (sumCross / crossScores.length) : 0;
    const isHallucination = crossMpd > t2;

    this.logger.log(`Stage 2 (Cross-Model Consistency) finished - Cross MPD: ${crossMpd.toFixed(4)} (Threshold: t2=${t2})`);

    return {
      is_hallucination: isHallucination,
      confidence: isHallucination ? crossMpd : 1 - crossMpd,
      reason: isHallucination
        ? `[Stage 2 Detected] Cross-model inconsistency (Cross MPD: ${crossMpd.toFixed(4)}) exceeds the threshold (t2: ${t2}). Classified as hallucination due to disagreement with the verifier.`
        : `[Stage 2 Pass] Cross-model inconsistency (Cross MPD: ${crossMpd.toFixed(4)}) is below or equal to the threshold (t2: ${t2}). Classified as factual due to agreement with the verifier.`,
      details: {
        stage: 2,
        selfMpd,
        crossMpd,
        targetSamples,
        verifierSamples: verifierAnswers,
        thresholdsUsed: { t1, tStar, t2, m },
      },
    };
  }

  // Verification sub-question generator agent
  private async generateSubQuestions(question: string, response: string, m: number, targetModel: string, hasKeys: boolean): Promise<string[]> {
    const numQuestions = Math.max(1, m - 1);
    
    if (!hasKeys) {
      // Mock mode
      return Array.from({ length: numQuestions }).map((_, i) => 
        `Mock verification sub-question ${i + 1} (Question: ${question.slice(0, 10)}...)`
      );
    }

    const systemPrompt = `You are a verification question generator. Given a Question and an Answer, generate exactly ${numQuestions} short, factual, and independent verification questions that check the specific facts asserted in the Answer. The questions should be clear, answerable in one sentence, and should not overlap. Output ONLY a valid JSON string array of questions. For example: ["Question 1?", "Question 2?"]`;
    const prompt = `Question: "${question}"\nAnswer: "${response}"`;
    
    try {
      const result = await this.callLLM(targetModel, prompt, 0.2, systemPrompt);
      const cleanJson = result.replace(/```json|```/g, '').trim();
      const parsed = JSON.parse(cleanJson);
      if (Array.isArray(parsed)) {
        return parsed.slice(0, numQuestions);
      }
    } catch (e) {
      this.logger.error(`Error generating verification questions, falling back to default sub-questions: ${e.message}`);
    }

    return Array.from({ length: numQuestions }).map((_, i) => 
      `Verification sub-question ${i + 1} about the answer details.`
    );
  }

  // ── Helper Methods ──

  private hasApiKeys(): boolean {
    return !!(process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY);
  }

  private calculateMpd(matrix: number[][]): number {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        sum += matrix[i][j];
        count++;
      }
    }
    const mean = count > 0 ? sum / count : 0;
    return 1 - mean;
  }

  private async getEntailment(premise: string, hypothesis: string, useHf: boolean, hasKeys: boolean, nliModel: string): Promise<number> {
    const useSampling = this.canUseSampling();
    if (!hasKeys && !useSampling) {
      // Fall back to Jaccard similarity simulation if API keys and client sampling are both unavailable
      return this.calculateJaccardSimilarity(premise, hypothesis);
    }

    if (useHf && process.env.HF_API_KEY) {
      try {
        return await this.callHuggingFaceNli(premise, hypothesis);
      } catch (e) {
        this.logger.warn(`HF Inference API call failed, falling back to LLM-based evaluation: ${e.message}`);
      }
    }

    // LLM-based entailment evaluation
    return await this.callLlmNli(premise, hypothesis, nliModel);
  }

  private calculateJaccardSimilarity(a: string, b: string): number {
    const cleanTokens = (text: string) =>
      text.toLowerCase()
        .replace(/[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]/g, '')
        .split(/\s+/)
        .filter(t => t.length > 0);

    const tokensA = cleanTokens(a);
    const tokensB = cleanTokens(b);
    
    if (tokensA.length === 0 && tokensB.length === 0) return 1.0;
    
    const setA = new Set(tokensA);
    const setB = new Set(tokensB);
    
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    const union = new Set([...setA, ...setB]);
    
    return intersection.size / union.size;
  }

  private async callLLM(model: string, prompt: string, temperature: number, systemPrompt?: string): Promise<string> {
    if (this.hasApiKeys()) {
      if (model.startsWith('claude') && process.env.ANTHROPIC_API_KEY) {
        return await this.callAnthropic(model, prompt, temperature, systemPrompt);
      } else if (process.env.OPENAI_API_KEY) {
        return await this.callOpenAI(model, prompt, temperature, systemPrompt);
      }
    } else if (this.canUseSampling()) {
      this.logger.log(`Sending LLM request using client sampling capability (Model: ${model})`);
      return await this.callClientSampling(model, prompt, temperature, systemPrompt);
    }
    throw new Error('No available API key or client sampling capability found.');
  }

  private async callOpenAI(model: string, prompt: string, temperature: number, systemPrompt?: string): Promise<string> {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY || ''}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
          { role: 'user', content: prompt },
        ],
        temperature,
        max_tokens: 256,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`OpenAI API Error (status ${response.status}): ${err}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '';
  }

  private async callAnthropic(model: string, prompt: string, temperature: number, systemPrompt?: string): Promise<string> {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY || '',
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: prompt }],
        system: systemPrompt,
        temperature,
        max_tokens: 256,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Anthropic API Error (status ${response.status}): ${err}`);
    }

    const data = await response.json();
    return data.content?.[0]?.text?.trim() || '';
  }

  private async callLlmNli(premise: string, hypothesis: string, nliModel: string): Promise<number> {
    const systemPrompt = `You are a Natural Language Inference (NLI) model. Determine the probability that the Premise entails the Hypothesis. Entailment means that if the Premise is true, the Hypothesis must also be true. Respond with a single floating-point number between 0.0 and 1.0 (where 1.0 means definite entailment and 0.0 means no entailment). Do not include any other text, reasoning, or formatting.`;
    const prompt = `Premise: "${premise}"\nHypothesis: "${hypothesis}"\nProbability of entailment:`;

    try {
      const result = await this.callLLM(nliModel, prompt, 0.0, systemPrompt);
      const score = parseFloat(result);
      if (isNaN(score)) {
        this.logger.warn(`Failed to parse NLI result as a number: "${result}"`);
        return 0.5;
      }
      return Math.max(0.0, Math.min(1.0, score));
    } catch (e) {
      this.logger.error(`LLM NLI call error: ${e.message}`);
      return 0.5; // Return neutral/uncertain value on error
    }
  }

  private async callHuggingFaceNli(premise: string, hypothesis: string): Promise<number> {
    // Use microsoft/deberta-v2-xlarge-mnli model
    const response = await fetch('https://api-inference.huggingface.co/models/microsoft/deberta-v2-xlarge-mnli', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.HF_API_KEY || ''}`,
      },
      body: JSON.stringify({
        inputs: {
          text: premise,
          text_pair: hypothesis,
        },
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`HF API Error (status ${response.status}): ${err}`);
    }

    const data = await response.json();
    if (Array.isArray(data) && Array.isArray(data[0])) {
      const entailmentObj = data[0].find((item: any) =>
        item.label && item.label.toLowerCase().includes('entail')
      );
      if (entailmentObj) {
        return entailmentObj.score;
      }
    }
    throw new Error('Invalid Hugging Face response format');
  }

  // ── Client-Side Sampling & JSON-RPC Helpers ──

  registerWriteCallback(callback: (msg: any) => void) {
    this.writeCallback = callback;
  }

  canUseSampling(): boolean {
    return !!this.clientCapabilities?.sampling;
  }

  private sendRequestToClient(method: string, params: any): Promise<any> {
    if (!this.writeCallback) {
      throw new Error('writeCallback is not registered. Cannot send requests to the client.');
    }
    const id = Math.random().toString(36).substring(2, 11);
    const req = {
      jsonrpc: '2.0' as const,
      id,
      method,
      params,
    };
    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      this.writeCallback!(req);
    });
  }

  private handleResponse(res: any) {
    const id = res.id;
    if (id !== undefined && this.pendingRequests.has(id)) {
      const { resolve, reject } = this.pendingRequests.get(id)!;
      this.pendingRequests.delete(id);
      if (res.error) {
        reject(new Error(res.error.message || JSON.stringify(res.error)));
      } else {
        resolve(res.result);
      }
    }
  }

  private async callClientSampling(
    model: string,
    prompt: string,
    temperature: number,
    systemPrompt?: string
  ): Promise<string> {
    const params = {
      messages: [
        {
          role: 'user',
          content: {
            type: 'text',
            text: prompt,
          },
        },
      ],
      modelPreferences: {
        hints: [
          {
            name: model,
          },
        ],
      },
      systemPrompt,
      temperature,
      maxTokens: 256,
    };

    const response = await this.sendRequestToClient('sampling/createMessage', params);
    
    if (response?.content?.type === 'text') {
      return response.content.text.trim();
    }
    if (Array.isArray(response?.content)) {
      const textContent = response.content.find((c: any) => c.type === 'text');
      if (textContent) {
        return textContent.text.trim();
      }
    }
    throw new Error('Invalid sampling response format.');
  }
}
