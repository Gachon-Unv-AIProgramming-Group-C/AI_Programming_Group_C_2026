import { Injectable, Logger } from '@nestjs/common';
import {
  JsonRpcRequest,
  JsonRpcResponse,
  McpTool,
  ToolCallParams,
  HallucinationCheckInput,
  HallucinationCheckResult,
} from './mcp.types';

// McpService는 MCP 프로토콜 처리와 할루시네이션 탐지 핵심 알고리즘을 담당한다
@Injectable()
export class McpService {
  private readonly logger = new Logger(McpService.name);
  private clientCapabilities: any = null;  // Claude가 보내는 클라이언트 기능 목록을 저장
  private writeCallback?: (msg: any) => void;  // stdio 모드에서 응답을 쓰는 콜백
  // Claude의 sampling 요청(LLM 호출)에 대한 응답을 비동기로 받기 위한 Promise 맵
  private readonly pendingRequests = new Map<
    string | number,
    { resolve: (value: any) => void; reject: (reason?: any) => void }
  >();

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
    this.logger.log(`[INIT] sampling=${this.canUseSampling()} hfKey=${!!(process.env.HF_API_KEY)} apiKeys=${this.hasApiKeys()}`);
    return {
      jsonrpc: '2.0',
      id: req.id,
      result: {
        protocolVersion: '2024-11-05',
        capabilities: {
          tools: {},
          sampling: {},
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

        const content: { type: string; text: string }[] = [
          { type: 'text', text: JSON.stringify(result, null, 2) },
        ];

        if (result.verdict === 'UNCERTAIN') {
          const score = result.details?.final_score?.toFixed(3) ?? 'N/A';
          content.push({
            type: 'text',
            text: `\n[UNCERTAIN] The hallucination detector could not make a confident judgment (score: ${score}). Please ask the user whether they would like to perform a manual review of this response.`,
          });
        }

        return {
          jsonrpc: '2.0',
          id: req.id,
          result: { content },
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

  private async checkHallucination(input: HallucinationCheckInput): Promise<HallucinationCheckResult> {
    const targetModel = input.targetModel || 'gpt-4o-mini';
    const verifierModel = input.verifierModel || 'gpt-4o-mini';
    const nliModel = input.nliModel || 'jhgan/ko-sroberta-nli';
    const m = Math.min(input.m || 5, 10);
    const t1 = input.t1 !== undefined ? input.t1 : 0.3;
    const tStar = input.tStar !== undefined ? input.tStar : 0.7;
    const s2_threshold = input.s2_threshold !== undefined ? input.s2_threshold : 0.4;
    const t2 = input.t2 !== undefined ? input.t2 : 0.5;
    const w1 = input.w1 !== undefined ? input.w1 : 0.2;
    const w2 = input.w2 !== undefined ? input.w2 : 0.3;
    const w3 = input.w3 !== undefined ? input.w3 : 0.5;
    const useHf = !!input.useHuggingFaceNli;

    // Determine available backends
    const hasKeys = this.hasApiKeys();
    const useSampling = this.canUseSampling();
    const hasHfKey = !!(process.env.HF_API_KEY);
    const hasLocalLlm = this.hasLocalLlm();
    const hasLocalPython = this.hasLocalPythonServer();
    const canRunRealLlm = hasKeys || useSampling || hasHfKey || hasLocalLlm || hasLocalPython;
    const mode = canRunRealLlm 
      ? (hasKeys || useSampling || hasLocalLlm ? 'llm' : 'local-python') 
      : (useHf && hasHfKey ? 'hf-nli-proxy' : 'mock');

    const layersRun: number[] = [];
    let score1 = 0;
    let score2 = 0;
    let score3 = 0;
    let forceLayers2And3 = false;

    const details: any = {
      layersRun,
      mode,
      thresholdsUsed: { t1, tStar, s2_threshold, t2, m },
      weightsUsed: { w1, w2, w3 },
    };

    this.logger.debug(`check_hallucination mode=${mode} q="${input.question.slice(0, 60)}"`);

    // ─── Layer 1: LSC (Lowest Span Confidence) ───────────────────────────────
    // 토큰별 로그확률(logprobs)이 제공된 경우에만 실행된다
    // Claude API는 logprobs를 미제공하므로, OpenAI API 사용 시에만 동작한다
    layersRun.push(1);
    if (input.logprobs && input.logprobs.length > 0) {
      let minSpanConfidence = 1.0;
      const tokens = input.logprobs;
      const windowSizes = [2, 3, 5];  // 슬라이딩 윈도우 크기 목록
      let hasValidWindow = false;

      // 슬라이딩 윈도우로 각 구간의 기하평균 확률을 계산하여 최솟값을 찾는다
      for (const w of windowSizes) {
        if (tokens.length >= w) {
          for (let i = 0; i <= tokens.length - w; i++) {
            let sumLogProb = 0;
            for (let j = 0; j < w; j++) {
              sumLogProb += tokens[i + j].logprob;
            }
            const spanProb = Math.exp(sumLogProb / w); // 로그확률 합의 지수 = 기하평균 확률
            if (spanProb < minSpanConfidence) {
              minSpanConfidence = spanProb;
            }
            hasValidWindow = true;
          }
        }
      }

      // score1 = 1 - LSC: 값이 클수록 모델이 불확실했음을 의미 (할루시네이션 위험)
      const lsc = hasValidWindow ? minSpanConfidence : 0.5;
      score1 = 1 - lsc;
      details.score1 = score1;

      // score1이 t1 미만이면 신뢰도가 충분하므로 Layer 2, 3 없이 조기 종료
      if (score1 < t1) {
        const final_score = w1 * score1;
        return {
          verdict: 'NO_HALLUCINATION' as const,
          is_hallucination: false,
          confidence: 1 - final_score,
          reason: `[Layer 1 Pass] Lowest Span Confidence (LSC) risk score ${score1.toFixed(4)} is below t1 (${t1}). Trusted response.`,
          details: { ...details, final_score, stage: 1, selfMpd: score1, combinedMpd: score1 }
        };
      }
      // score1이 tStar 이상이면 매우 위험하다고 판단하여 Layer 2, 3을 강제 실행
      if (score1 >= tStar) {
        forceLayers2And3 = true;
      }
    } else {
      // logprobs가 없으면 Layer 1을 건너뛰고 중간값(0.45)으로 설정한다
      score1 = 0.45;
      details.score1 = score1;
    }

    // ─── Layer 2: SINdex (Semantic Inconsistency Index) ─────────────────────
    // 동일 질문을 LLM에 m회 독립 샘플링하여 응답들의 의미적 분산을 측정한다
    // 분산이 클수록 모델이 불확실하다는 의미이므로 할루시네이션 위험도가 높다
    layersRun.push(2);

    // 한국어/영어를 감지하여 LLM 프롬프트 언어와 군집화 임계값을 다르게 적용한다
    // 한국어는 형태소 변형이 많아 표면형이 달라도 의미가 같은 경우가 많으므로 더 높은 임계값 사용
    const isKorean = /[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]/.test(input.question);
    const clusterMergeThreshold = isKorean ? 0.90 : 0.65;

    // Dynamically adapt prompt length based on the original response's format (short vs full sentence)
    const cleanResponse = input.response.trim();
    const isShortResponse = cleanResponse.split(/\s+/).length <= 2 || cleanResponse.length < 10;

    const sindexSystemPrompt = isKorean
      ? (isShortResponse
          ? `You are a factual question-answering assistant. Answer the question in a single word or a short phrase only. Do not write a full sentence. Example: "서울" or "대한민국".`
          : `You are a factual question-answering assistant. Answer the question in ONE short, direct sentence in Korean only. Do not add explanations or extra sentences.`)
      : (isShortResponse
          ? `You are a factual question-answering assistant. Answer the question in a single word or a short phrase only. Do not write a full sentence. Example: "Paris" or "France".`
          : `You are a factual question-answering assistant. Answer the question in ONE short, direct sentence in English only. Do not add explanations, qualifications, or extra sentences. Example: "The capital of France is Paris."`);

    // 검증 대상 응답(index 0)을 포함하여 LLM에서 m개의 샘플을 추가로 생성한다
    const targetSamples: string[] = [input.response];

    if (canRunRealLlm) {
      // LLM을 temperature=0.7로 m회 호출하여 다양한 응답 샘플을 수집한다
      for (let i = 0; i < m; i++) {
        try {
          const sample = await this.callLLM(targetModel, input.question, 0.7, sindexSystemPrompt);
          targetSamples.push(sample);
        } catch (e) {
          this.logger.error(`Error generating sample ${i + 1}: ${e.message}`);
        }
      }

    } else if (mode === 'hf-nli-proxy') {
      targetSamples.push(input.response + " (alternative phrasing)");
      targetSamples.push(input.response + " (detailed statement)");
    } else {
      // Mock mode
      for (let i = 0; i < m; i++) {
        targetSamples.push(i % 2 === 0 ? input.response : `Different mock response variation ${i + 1}`);
      }
    }

    // 모든 샘플 쌍의 의미적 유사도를 KxK 행렬로 계산한다
    // NLI entailment 점수 또는 Jaccard 유사도를 사용한다
    const K = targetSamples.length;
    const similarityMatrix: number[][] = [];
    for (let i = 0; i < K; i++) {
      try {
        const sims = await this.getBulkSimilarity(targetSamples[i], targetSamples, useHf, nliModel);
        similarityMatrix.push(sims);
      } catch (e) {
        this.logger.error(`Error calculating bulk similarity for sample ${i}: ${e.message}`);
        // NLI 실패 시 Jaccard 유사도로 대체한다
        similarityMatrix.push(targetSamples.map(t => this.calculateJaccardSimilarity(targetSamples[i], t)));
      }
    }

    // Agglomerative Clustering (Average Linkage): 유사도가 임계값 이상인 샘플들을 같은 클러스터로 병합한다
    // 클러스터 수가 많을수록 응답이 분산되어 있다는 의미 → 높은 불확실성
    let clusters: number[][] = Array.from({ length: K }).map((_, i) => [i]);
    while (clusters.length > 1) {
      let maxSim = -1;
      let mergeI = -1;
      let mergeJ = -1;

      // 모든 클러스터 쌍 중 평균 유사도가 가장 높은 쌍을 찾는다
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          let sumSim = 0;
          const cA = clusters[i];
          const cB = clusters[j];
          for (const idxA of cA) {
            for (const idxB of cB) {
              sumSim += similarityMatrix[idxA][idxB];
            }
          }
          const avgSim = sumSim / (cA.length * cB.length);  // Average Linkage
          if (avgSim > maxSim) {
            maxSim = avgSim;
            mergeI = i;
            mergeJ = j;
          }
        }
      }

      // 최대 유사도가 임계값 이상이면 두 클러스터를 병합한다. 이하면 더 이상 병합하지 않는다
      if (maxSim >= clusterMergeThreshold) {  // 한국어: 0.90, 영어: 0.65
        clusters[mergeI] = [...clusters[mergeI], ...clusters[mergeJ]];
        clusters.splice(mergeJ, 1);
      } else {
        break;
      }
    }

    // 클러스터 분포의 Shannon 엔트로피를 계산하고 log(K)로 정규화한다
    // 클러스터가 1개이면 0, 모두 다른 클러스터이면 1에 가까워진다
    let entropySum = 0;
    for (const c of clusters) {
      const pC = c.length / K;  // 각 클러스터의 비율
      if (pC > 0) {
        entropySum += pC * Math.log(pC);
      }
    }
    const rawEntropy = -entropySum;
    const dispersion = K > 1 ? rawEntropy / Math.log(K) : 0;  // 0~1로 정규화

    // NLI 기반 불일치율: 각 샘플이 원본 응답을 함의하는 정도의 평균을 구해 역수를 취한다
    // 샘플들이 원본을 모순할수록 nliInconsistency가 1에 가까워진다
    // 이 지표는 "모두 틀리게 일관된" 할루시네이션을 탐지하기 위한 핵심 척도이다
    const entailmentsToOriginal: number[] = [];
    for (let i = 1; i < K; i++) {
      try {
        // premise(전제): 생성된 샘플, hypothesis(가설): 원본 응답
        // 샘플이 원본을 함의하면 entailment 점수가 높다
        const score = await this.getEntailment(targetSamples[i], targetSamples[0], useHf, hasKeys, nliModel);
        entailmentsToOriginal.push(score);
      } catch (e) {
        this.logger.error(`Error calculating entailment for sample ${i}: ${e.message}`);
        // NLI 실패 시 Jaccard 유사도로 대체한다
        entailmentsToOriginal.push(similarityMatrix[i][0]);
      }
    }

    const avgEntailment = entailmentsToOriginal.length > 0
      ? entailmentsToOriginal.reduce((sum, s) => sum + s, 0) / entailmentsToOriginal.length
      : 1.0;
    const nliInconsistency = 1 - avgEntailment;  // 평균 entailment의 반대값

    // 각 클러스터의 크기와 원본 응답과의 유사도를 계산한다 (디버깅용)
    const clusterDetails = clusters.map((c, idx) => {
      const members = c.map(i => targetSamples[i]);
      const sumSimToOriginal = c.reduce((sum, i) => sum + similarityMatrix[i][0], 0);
      const simToOriginal = sumSimToOriginal / c.length;
      return {
        label: `Cluster ${idx + 1}`,
        size: c.length,
        members,
        similarityToOriginal: simToOriginal
      };
    });

    // 생성된 샘플(index >= 1) 중 가장 많은 수를 포함하는 다수 클러스터를 찾는다
    // 다수 클러스터가 원본과 다를수록 majorityDisagreement가 높아진다
    let majorityCluster = clusters.find(c => c.includes(0)) || clusters[0];
    let maxGenSize = majorityCluster.filter(idx => idx >= 1).length;

    for (const c of clusters) {
      const genSize = c.filter(idx => idx >= 1).length;
      if (genSize > maxGenSize) {
        maxGenSize = genSize;
        majorityCluster = c;
      }
    }

    const sumSimToOriginal = majorityCluster.reduce((sum, idx) => sum + similarityMatrix[idx][0], 0);
    const majoritySimToOriginal = majorityCluster.length > 0 ? sumSimToOriginal / majorityCluster.length : 1.0;
    const majorityDisagreement = 1 - majoritySimToOriginal;  // 다수 클러스터와 원본의 불일치 정도

    // SINdex 최종 점수: 분산(10%) + NLI 불일치(30%) + 다수 불일치(60%) 가중합산
    score2 = 0.1 * dispersion + 0.3 * nliInconsistency + 0.6 * majorityDisagreement;
    score2 = Math.max(0, Math.min(1, score2));  // 0~1 범위로 클리핑

    details.clusters = clusterDetails;
    details.score2 = score2;
    details.dispersion = dispersion;
    details.nliInconsistency = nliInconsistency;
    details.majorityDisagreement = majorityDisagreement;
    details.targetSamples = targetSamples;

    if (score2 < s2_threshold && !forceLayers2And3) {
      const final_score = w1 * score1 + w2 * score2;
      return {
        verdict: 'NO_HALLUCINATION' as const,
        is_hallucination: false,
        confidence: 1 - final_score,
        reason: `[Layer 2 Pass] SINdex score ${score2.toFixed(4)} is below threshold (${s2_threshold}). Consistent response.`,
        details: { ...details, final_score, stage: 2, selfMpd: score2, combinedMpd: (score1 + score2) / 2 }
      };
    }

    // ─── Layer 3: SAC³ (Self-Adaptive Cross-model Consistency) ─────────────
    // 원본 질문을 패러프레이즈하여 다른 표현으로 LLM에 재질의한다
    // 응답들이 원본 응답과 의미적으로 불일치하면 할루시네이션으로 판정한다
    layersRun.push(3);
    let paraphrasedQuestions: string[] = [];

    try {
      paraphrasedQuestions = await this.callLocalParaphraser(input.question);
    } catch (e) {
      this.logger.warn(`Local paraphraser unavailable: ${e.message}`);
    }

    if (paraphrasedQuestions.length === 0 && canRunRealLlm) {
      const systemPrompt = `You are a question paraphrasing expert. Generate exactly 3 semantic variations of the given question. They must ask for the exact same factual information but using different phrasing or structure in the same language. Output ONLY a valid JSON string array of the 3 questions.`;
      try {
        const result = await this.callLLM(targetModel, `Question: "${input.question}"`, 0.3, systemPrompt);
        const cleanJson = result.replace(/```json|```/g, '').trim();
        const parsed = JSON.parse(cleanJson);
        if (Array.isArray(parsed) && parsed.length > 0) {
          paraphrasedQuestions = parsed.map((q: any) => String(q).trim()).slice(0, 3);
        }
      } catch (e) {
        this.logger.error(`Error generating paraphrased questions: ${e.message}`);
      }
    }

    if (paraphrasedQuestions.length === 0) {
      paraphrasedQuestions = [
        `Rephrase: ${input.question}`,
        `Restate: ${input.question}`,
      ];
    }
    details.paraphrasedQuestions = paraphrasedQuestions;

    const nSac = 2;
    const sacTargetAnswers: string[] = [];
    const sacVerifierAnswers: string[] = [];

    if (canRunRealLlm) {
      for (const pq of paraphrasedQuestions) {
        for (let i = 0; i < nSac; i++) {
          try {
            const tAns = await this.callLLM(targetModel, pq, 0.7, sindexSystemPrompt);
            sacTargetAnswers.push(tAns);
          } catch (e) {
            this.logger.error(`Error sampling target answer for "${pq}": ${e.message}`);
          }
          try {
            const vAns = await this.callLLM(verifierModel, pq, 0.7, sindexSystemPrompt);
            sacVerifierAnswers.push(vAns);
          } catch (e) {
            this.logger.error(`Error sampling verifier answer for "${pq}": ${e.message}`);
          }
        }
      }
    } else {
      for (const pq of paraphrasedQuestions) {
        for (let i = 0; i < nSac; i++) {
          sacTargetAnswers.push(i === 0 ? input.response : `Mock target SAC3 response for: ${pq}`);
          sacVerifierAnswers.push(i === 0 ? input.response : `Mock verifier SAC3 response for: ${pq}`);
        }
      }
    }

    const allSacAnswers = [...sacTargetAnswers, ...sacVerifierAnswers];
    const entailmentScores: number[] = [];

    for (const ans of allSacAnswers) {
      const score = await this.getEntailment(input.response, ans, useHf, hasKeys, nliModel);
      entailmentScores.push(score);
    }

    const sumEntailment = entailmentScores.reduce((sum, s) => sum + s, 0);
    score3 = entailmentScores.length > 0 ? 1 - (sumEntailment / entailmentScores.length) : 0.5;
    score3 = Math.max(0, Math.min(1, score3));

    details.score3 = score3;
    details.verifierSamples = allSacAnswers;

    const final_score = w1 * score1 + w2 * score2 + w3 * score3;
    const uncertaintyMargin = input.uncertainty_margin ?? 0.05;

    let verdict: 'HALLUCINATION' | 'NO_HALLUCINATION' | 'UNCERTAIN';
    if (final_score > t2 + uncertaintyMargin) {
      verdict = 'HALLUCINATION';
    } else if (final_score < t2 - uncertaintyMargin) {
      verdict = 'NO_HALLUCINATION';
    } else {
      verdict = 'UNCERTAIN';
    }
    const isHallucination = verdict === 'HALLUCINATION';

    this.logger.log(`Final 3-Layer Cascade decision: final_score=${final_score.toFixed(4)}, t2=${t2}±${uncertaintyMargin} → ${verdict}`);

    return {
      verdict,
      is_hallucination: isHallucination,
      confidence: verdict === 'UNCERTAIN' ? 0.5 : (isHallucination ? final_score : 1 - final_score),
      reason: verdict === 'HALLUCINATION'
        ? `[Layer 3 Detected] Combined risk score ${final_score.toFixed(4)} (L1: ${score1.toFixed(4)}, L2: ${score2.toFixed(4)}, L3: ${score3.toFixed(4)}) exceeds threshold t2 (${t2}). Hallucination detected.`
        : verdict === 'UNCERTAIN'
        ? `[Layer 3 Uncertain] Combined risk score ${final_score.toFixed(4)} is within uncertainty zone t2±${uncertaintyMargin} (${(t2 - uncertaintyMargin).toFixed(2)}~${(t2 + uncertaintyMargin).toFixed(2)}). Manual review recommended.`
        : `[Layer 3 Pass] Combined risk score ${final_score.toFixed(4)} (L1: ${score1.toFixed(4)}, L2: ${score2.toFixed(4)}, L3: ${score3.toFixed(4)}) is within threshold t2 (${t2}). Factual.`,
      details: {
        ...details,
        final_score,
        stage: 3,
        selfMpd: score2,
        crossMpd: score3,
        combinedMpd: (score2 + score3) / 2
      }
    };
  }

  // Verification sub-question generator agent
  // When LLM is available: use LLM to generate sub-questions.
  // When HF-proxy mode (no LLM, has HF): use rule-based decomposition from the question/response.
  // Fallback: mock mode for testing.
  private async generateSubQuestions(question: string, response: string, m: number, targetModel: string, hasKeys: boolean): Promise<string[]> {
    const numQuestions = Math.max(1, m - 1);
    const hasHfKey = !!(process.env.HF_API_KEY);
    
    if (hasKeys) {
      // LLM-based sub-question generation
      const systemPrompt = `You are a hallucination verification question generator. Given a Question and an Answer, generate exactly ${numQuestions} YES/NO verification questions that each check one specific factual claim from the Answer.

Rules:
1. CRITICAL: Generate questions in the SAME LANGUAGE as the Question and Answer.
2. Each question MUST directly reference the specific facts stated in the Answer (exact dates, numbers, magnitudes, names, locations).
3. Phrase as a closed YES/NO question. Example: "Is it true that the earthquake occurred on April 2, 2024?"
4. Do NOT ask open-ended questions.
5. Questions must be independent and non-overlapping.
6. Output ONLY a valid JSON string array.`;
      const prompt = `Question: "${question}"\nAnswer: "${response}"`;

      try {
        const result = await this.callLLM(targetModel, prompt, 0, systemPrompt);
        const cleanJson = result.replace(/```json|```/g, '').trim();
        const parsed = JSON.parse(cleanJson);
        if (Array.isArray(parsed)) {
          const valid = parsed
            .filter((q: any) => typeof q === 'string' && q.trim().length >= 10 && /[가-힣a-zA-Z]{4,}/.test(q))
            .slice(0, numQuestions);
          if (valid.length > 0) return valid;
        }
      } catch (e) {
        this.logger.error(`Error generating verification questions, falling back: ${e.message}`);
      }
    }

    if (hasHfKey) {
      // Rule-based sub-question generation for HF-proxy mode.
      // Decompose the response into multiple verification angles.
      this.logger.log('[HF-proxy] Generating rule-based sub-questions from question/response.');

      const candidates: string[] = [];

      // 1. Paraphrase the original question
      candidates.push(question);

      // 2. Extract key entities and ask about them individually
      // Split response into clauses to extract claims
      const clauses = response
        .split(/[,;，；]/)
        .map(s => s.trim())
        .filter(s => s.length > 5);
      for (const clause of clauses) {
        candidates.push(`Is it true that "${clause}"?`);
      }

      // 3. Negation check: assert the opposite
      candidates.push(`Is the opposite of "${response.slice(0, 50)}" true?`);

      // 4. Specificity check: ask about key details
      candidates.push(`What specific facts can be confirmed about: ${question}`);

      // 5. Consistency check: rephrase with different wording
      candidates.push(`What is the accurate answer to: ${question}`);

      // Return up to numQuestions unique candidates
      return candidates.slice(0, numQuestions);
    }
    
    // Mock mode fallback (no LLM, no HF key)
    return Array.from({ length: numQuestions }).map((_, i) => 
      `Mock verification sub-question ${i + 1} (Question: ${question.slice(0, 10)}...)`
    );
  }



  private hasApiKeys(): boolean {
    return !!(process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY);
  }

  /**
   * Check if a local LLM server (e.g. Ollama, llama.cpp, vLLM) is configured.
   * Set LOCAL_LLM_URL to an OpenAI-compatible endpoint, e.g.:
   *   - Ollama:    http://127.0.0.1:11434/v1
   *   - llama.cpp: http://127.0.0.1:8080/v1
   *   - vLLM:      http://127.0.0.1:8000/v1
   * Optionally set LOCAL_LLM_MODEL (default: auto-detected or 'qwen2.5:7b').
   */
  private hasLocalLlm(): boolean {
    return !!(process.env.LOCAL_LLM_URL);
  }

  private hasLocalPythonServer(): boolean {
    return true;
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

  private async getEntailment(premise: string, hypothesis: string, useHf: boolean, hasKeys: boolean, nliModel: string, useLocalModel = false): Promise<number> {
    // If one contains a negation and the other does not, they are logically contradictory, so entailment must be 0.0.
    if (this.hasNegation(premise) !== this.hasNegation(hypothesis)) {
      return 0.0;
    }

    // Prioritize local Python NLI server if configured to avoid external API calls/quota issues
    if (this.hasLocalPythonServer()) {
      try {
        return await this.callLocalNli(premise, hypothesis);
      } catch (e) {
        this.logger.warn(`Local NLI server call failed: ${e.message}. Trying other methods.`);
      }
    }

    // 1. Local NLI server — only when explicitly requested (Stage 2).
    //    KLUE-RoBERTa is Korean-specialized; using it on Stage 1 (KO premise + EN hypothesis) causes false positives.
    if (useLocalModel) {
      try {
        return await this.callLocalNli(premise, hypothesis);
      } catch (e) {
        this.logger.warn(`Local NLI server unavailable: ${e.message}`);
      }
    }

    // 2. HF Inference API fallback (requires credits)
    if (useHf && process.env.HF_API_KEY) {
      try {
        const hfModelId = nliModel.includes('/') ? nliModel : undefined;
        return await this.callHuggingFaceNli(premise, hypothesis, hfModelId);
      } catch (e) {
        this.logger.warn(`HF Inference API call failed: ${e.message}. Trying local NLI model fallback.`);
        try {
          return await this.callLocalNli(premise, hypothesis);
        } catch (localErr) {
          this.logger.warn(`Local NLI fallback also failed: ${localErr.message}`);
        }
      }
    }

    // 3. LLM-based entailment evaluation
    try {
      return await this.callLlmNli(premise, hypothesis, nliModel);
    } catch (e) {
      this.logger.warn(`LLM NLI call failed: ${e.message}. Trying local NLI model fallback.`);
      try {
        return await this.callLocalNli(premise, hypothesis);
      } catch (localErr) {
        this.logger.warn(`Local NLI fallback failed: ${localErr.message}`);
      }
    }

    // 4. Jaccard similarity fallback
    return this.calculateJaccardSimilarity(premise, hypothesis);
  }

  private stemKoreanToken(token: string): string {
    if (!/[\uAC00-\uD7AF]/.test(token)) return token;

    let stemmed = token;
    const suffixes = [
      '였습니까', '였습니다', '입니까', '입니다', '였으며', '이었고', '이고', '이며',
      '에서', '으로', '부터', '까지', '보다', '처럼', '만큼',
      '은', '는', '이', '가', '을', '를', '의', '에', '로', '와', '과', '도', '만', '랑', '나'
    ];

    for (const suffix of suffixes) {
      if (stemmed.endsWith(suffix) && stemmed.length > suffix.length) {
        stemmed = stemmed.slice(0, -suffix.length);
        break;
      }
    }
    return stemmed;
  }

  private hasNegation(text: string): boolean {
    const clean = text.toLowerCase().replace(/[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]/g, '');
    const tokens = clean.split(/\s+/);
    
    // Comprehensive Korean negation words and suffixes
    const krNegations = [
      '아닙니다', '아니', '않습니다', '않다', '안', '못', '아님', '아니다', 
      '않음', '않은', '아닌', '말고', '말라', '말아', '없다', '없습니다', '없음', '없는'
    ];
    // Comprehensive English negation words and contractions (stripped of apostrophes)
    const enNegations = [
      'not', 'never', 'no', 'cannot', 'cant', 'isnt', 'arent', 'didnt', 
      'wasnt', 'werent', 'havent', 'hasnt', 'hadnt', 'dont', 'doesnt',
      'wont', 'shouldnt', 'couldnt', 'wouldnt', 'mustnt', 'shant', 'mightnt',
      'none', 'nothing', 'neither', 'nor', 'nobody', 'nowhere', 'lack', 'without'
    ];

    return tokens.some(t => {
      if (enNegations.includes(t) || krNegations.includes(t)) return true;
      return krNegations.some(neg => t.includes(neg));
    });
  }

  private calculateJaccardSimilarity(a: string, b: string): number {
    const cleanTokens = (text: string) =>
      text.toLowerCase()
        .replace(/[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]/g, '')
        .split(/\s+/)
        .filter(t => t.length > 0)
        .map(t => this.stemKoreanToken(t));

    const tokensA = cleanTokens(a);
    const tokensB = cleanTokens(b);
    
    if (tokensA.length === 0 && tokensB.length === 0) return 1.0;
    
    const setA = new Set(tokensA);
    const setB = new Set(tokensB);
    
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    const union = new Set([...setA, ...setB]);
    
    let sim = intersection.size / union.size;

    // Apply severe penalty if one statement is negated and the other is not (logical contradiction)
    if (this.hasNegation(a) !== this.hasNegation(b)) {
      sim = sim * 0.1;
    }

    return sim;
  }

  private async callLLM(model: string, prompt: string, temperature: number, systemPrompt?: string): Promise<string> {
    // Priority: Client Sampling (자체 에이전트) > Local Python Server (Qwen) > API Keys (OpenAI/Anthropic) > Local LLM (Ollama) > HF API
    try {
      if (this.canUseSampling()) {
        this.logger.log(`Sending LLM request using client sampling capability (Model: ${model})`);
        return await this.callClientSampling(model, prompt, temperature, systemPrompt);
      }
      if (this.hasLocalPythonServer()) {
        this.logger.log(`Using local Python server (Qwen) for LLM generation.`);
        return await this.callLocalPythonGenerate(prompt, temperature, systemPrompt);
      }
      if (this.hasApiKeys()) {
        if (model.startsWith('claude') && process.env.ANTHROPIC_API_KEY) {
          return await this.callAnthropic(model, prompt, temperature, systemPrompt);
        } else if (process.env.OPENAI_API_KEY) {
          return await this.callOpenAI(model, prompt, temperature, systemPrompt);
        }
      }
      if (this.hasLocalLlm()) {
        // Local LLM server (Ollama, llama.cpp, vLLM, etc.)
        const localModel = process.env.LOCAL_LLM_MODEL || 'qwen2.5:7b';
        this.logger.log(`Sending LLM request to local server: ${process.env.LOCAL_LLM_URL} (Model: ${localModel})`);
        return await this.callLocalLlm(localModel, prompt, temperature, systemPrompt);
      }
      if (process.env.HF_API_KEY) {
        const hfModel = (model.startsWith('gpt') || model.startsWith('claude') || model === 'gpt-4o-mini')
          ? (process.env.HF_LLM_MODEL_ID || 'Qwen/Qwen2.5-7B-Instruct')
          : model;
        this.logger.log(`Sending LLM request to Hugging Face Serverless Chat Completion API (Model: ${hfModel})`);
        return await this.callHuggingFaceLLM(hfModel, prompt, temperature, systemPrompt);
      }
    } catch (err) {
      this.logger.warn(`Standard LLM call failed: ${err.message}. Trying local Python LLM generation fallback.`);
      try {
        return await this.callLocalPythonGenerate(prompt, temperature, systemPrompt);
      } catch (localErr) {
        this.logger.error(`Local Python LLM fallback also failed: ${localErr.message}`);
      }
      throw err;
    }
    
    // If we get here, no standard method ran because no keys/configurations are set. Try local python server fallback.
    this.logger.log(`No keys configured. Querying local Python LLM generation fallback.`);
    return await this.callLocalPythonGenerate(prompt, temperature, systemPrompt);
  }

  /**
   * Call a local LLM server using OpenAI-compatible Chat Completion API.
   * Works with Ollama, llama.cpp server, vLLM, LM Studio, etc.
   */
  private async callLocalLlm(model: string, prompt: string, temperature: number, systemPrompt?: string): Promise<string> {
    const baseUrl = process.env.LOCAL_LLM_URL!.replace(/\/$/, '');
    const url = `${baseUrl}/chat/completions`;
    const body = {
      model,
      messages: [
        ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
        { role: 'user', content: prompt },
      ],
      temperature,
      max_tokens: 512,
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000), // Local LLM can be slow
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Local LLM API Error (status ${response.status}): ${err}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '';
  }

  private async callHuggingFaceLLM(model: string, prompt: string, temperature: number, systemPrompt?: string): Promise<string> {
    const url = 'https://router.huggingface.co/v1/chat/completions';
    const body = {
      model,
      messages: [
        ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
        { role: 'user', content: prompt }
      ],
      temperature,
      max_tokens: 256,
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.HF_API_KEY || ''}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`HF LLM API Error (status ${response.status}): ${err}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '';
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

  private get nliBaseUrl(): string {
    return process.env.NLI_SERVER_URL || `http://127.0.0.1:${process.env.NLI_SERVER_PORT || '8001'}`;
  }

  private async callLocalParaphraser(question: string): Promise<string[]> {
    const url = `${this.nliBaseUrl}/paraphrase`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) throw new Error(`Local Paraphraser server error: ${response.status}`);
    const data = await response.json();
    return data.paraphrases as string[];
  }

  private async callLocalPythonGenerate(prompt: string, temperature: number, systemPrompt?: string): Promise<string> {
    const url = `${this.nliBaseUrl}/generate`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, system_prompt: systemPrompt, temperature }),
      signal: AbortSignal.timeout(30000),
    });
    if (!response.ok) throw new Error(`Local Python generate server error: ${response.status}`);
    const data = await response.json();
    return data.text as string;
  }

  private async callHuggingFaceSimilarityBulk(source: string, targets: string[], customModelId?: string): Promise<number[]> {
    const modelId = customModelId || process.env.HF_MODEL_ID || 'jhgan/ko-sroberta-nli';
    this.logger.log(`Calling Hugging Face Inference API for bulk similarity with model: ${modelId}`);

    const url = `https://router.huggingface.co/hf-inference/models/${modelId}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.HF_API_KEY || ''}`,
      },
      body: JSON.stringify({
        inputs: {
          source_sentence: source,
          sentences: targets
        }
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`HF Bulk Similarity Error (status ${response.status}): ${err}`);
    }

    const data = await response.json();
    if (Array.isArray(data)) {
      return data.map((item: any) => {
        if (typeof item === 'number') return item;
        if (typeof item?.score === 'number') return item.score;
        return 0;
      });
    }
    throw new Error(`Unexpected bulk similarity response format: ${JSON.stringify(data)}`);
  }

  private async getBulkSimilarity(source: string, targets: string[], useHf: boolean, modelId: string): Promise<number[]> {
    // If using local Python server, avoid calling external Hugging Face similarity APIs to prevent credit depletion/timeouts.
    if (this.hasLocalPythonServer()) {
      return targets.map(t => this.calculateJaccardSimilarity(source, t));
    }

    if (useHf && process.env.HF_API_KEY) {
      try {
        return await this.callHuggingFaceSimilarityBulk(source, targets, modelId);
      } catch (err) {
        this.logger.warn(`HF Bulk Similarity failed: ${err.message}. Falling back to Jaccard similarity.`);
      }
    }
    return targets.map(t => this.calculateJaccardSimilarity(source, t));
  }

  private async callLocalNli(premise: string, hypothesis: string): Promise<number> {
    const url = `${this.nliBaseUrl}/nli`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ premise, hypothesis }),
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) throw new Error(`Local NLI server error: ${response.status}`);
    const data = await response.json();
    return data.entailment as number;
  }

  private async callLlmNli(premise: string, hypothesis: string, nliModel: string): Promise<number> {
    const systemPrompt = `You are a Natural Language Inference (NLI) model. Determine the probability that the Premise entails the Hypothesis. Entailment means that if the Premise is true, the Hypothesis must also be true. Respond with a single floating-point number between 0.0 and 1.0 (where 1.0 means definite entailment and 0.0 means no entailment). Do not include any other text, reasoning, or formatting.`;
    const prompt = `Premise: "${premise}"\nHypothesis: "${hypothesis}"\nProbability of entailment:`;

    try {
      const result = await this.callLLM(nliModel, prompt, 0.0, systemPrompt);
      // Extract first float in [0,1] range — handles responses like "The probability is 0.9."
      const match = result.match(/\b(1(?:\.0+)?|0(?:\.\d+)?)\b/);
      const score = match ? parseFloat(match[1]) : NaN;
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

  private async callHuggingFaceNli(premise: string, hypothesis: string, customModelId?: string): Promise<number> {
    const modelId = customModelId || process.env.HF_MODEL_ID || 'microsoft/deberta-v2-xlarge-mnli';
    this.logger.log(`Calling Hugging Face Inference API with model: ${modelId}`);

    const isSentenceSimilarity = modelId.includes('jhgan/ko-sroberta-nli') || modelId.includes('sroberta') || modelId.includes('sentence');
    const url = `https://router.huggingface.co/hf-inference/models/${modelId}`;

    let bodyObj: any;
    if (isSentenceSimilarity) {
      bodyObj = {
        inputs: {
          source_sentence: premise,
          sentences: [hypothesis]
        }
      };
    } else {
      bodyObj = {
        inputs: {
          text: premise,
          text_pair: hypothesis,
        }
      };
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.HF_API_KEY || ''}`,
      },
      body: JSON.stringify(bodyObj),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`HF API Error (status ${response.status}): ${err}`);
    }

    const data = await response.json();
    
    if (isSentenceSimilarity) {
      if (Array.isArray(data) && typeof data[0] === 'number') {
        return data[0];
      }
      if (Array.isArray(data) && typeof data[0]?.score === 'number') {
        return data[0].score;
      }
    } else {
      if (Array.isArray(data) && Array.isArray(data[0])) {
        const entailmentObj = data[0].find((item: any) => {
          if (!item.label) return false;
          const labelLower = item.label.toLowerCase();
          return labelLower.includes('entail') || labelLower === 'label_0';
        });
        if (entailmentObj) {
          return entailmentObj.score;
        }
      }
      if (Array.isArray(data) && typeof data[0] === 'object') {
        const entailmentObj = data.find((item: any) => {
          if (!item.label) return false;
          const labelLower = item.label.toLowerCase();
          return labelLower.includes('entail') || labelLower === 'label_0';
        });
        if (entailmentObj) {
          return entailmentObj.score;
        }
      }
    }

    throw new Error(`Invalid Hugging Face response format: ${JSON.stringify(data)}`);
  }


  // Uses a different model than Stage 1 (cross-encoder vs sentence similarity).
  // Returns entailment probability (0~1). Uses HF zero-shot-classification/NLI pipeline.
  private async callCrossEncoderNli(premise: string, hypothesis: string, modelId: string): Promise<number> {
    this.logger.log(`[CrossEncoder] Calling ${modelId} for NLI`);

    const isCustomModel = modelId.includes('serize') || modelId.includes('klue') || modelId.includes('stage2');
    if (isCustomModel) {
      this.logger.log(`[CrossEncoder] Detected custom classification model: ${modelId}. Querying standard classification API directly.`);
      try {
        return await this.callHuggingFaceNli(premise, hypothesis, modelId);
      } catch (e) {
        this.logger.warn(`[CrossEncoder] Custom model standard query failed: ${e.message}. Falling back to Stage 1 model.`);
        return await this.callHuggingFaceNli(premise, hypothesis);
      }
    }

    const url = `https://router.huggingface.co/hf-inference/models/${modelId}`;
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.HF_API_KEY || ''}`,
        },
        body: JSON.stringify({
          inputs: premise,
          parameters: {
            candidate_labels: [hypothesis, 'contradiction'],
          },
        }),
      });

      if (!response.ok) {
        const err = await response.text();
        this.logger.warn(`[CrossEncoder] API error (${response.status}): ${err}. Trying standard text-classification on Stage 2 model.`);
        return await this.callHuggingFaceNli(premise, hypothesis, modelId);
      }

      const data = await response.json();

      if (Array.isArray(data)) {
        const entailmentObj = data.find((item: any) =>
          typeof item.label === 'string' &&
          item.label.trim().toLowerCase() === hypothesis.trim().toLowerCase()
        );
        if (entailmentObj && typeof entailmentObj.score === 'number') {
          return entailmentObj.score;
        }
      }

      if (data && Array.isArray(data.labels) && Array.isArray(data.scores)) {
        // Handle standard zero-shot response mapping (dict with labels & scores arrays)
        const labelIndex = data.labels.findIndex((lbl: string) => 
          lbl.trim().toLowerCase() === hypothesis.trim().toLowerCase()
        );
        if (labelIndex !== -1 && typeof data.scores[labelIndex] === 'number') {
          return data.scores[labelIndex];
        }
      }

      this.logger.warn(`[CrossEncoder] Unexpected zero-shot response format: ${JSON.stringify(data)}. Trying standard text-classification.`);
      return await this.callHuggingFaceNli(premise, hypothesis, modelId);
    } catch (err) {
      this.logger.warn(`[CrossEncoder] Connection error: ${err.message}. Trying standard text-classification on Stage 2 model.`);
      try {
        return await this.callHuggingFaceNli(premise, hypothesis, modelId);
      } catch (e2) {
        this.logger.warn(`[CrossEncoder] Stage 2 standard classification query failed: ${e2.message}. Falling back to Stage 1 model.`);
        return await this.callHuggingFaceNli(premise, hypothesis);
      }
    }
  }

  // Generate a negated/contradictory version of the response for contradiction detection.
  // Used in Stage 2 to check if the NLI model can distinguish the original from its negation.
  private generateNegation(text: string): string {
    // Korean negation patterns (suffix-based)
    const negations: [RegExp, string][] = [
      [/입니다/g, '이(가) 아닙니다'],
      [/했습니다/g, '하지 않았습니다'],
      [/됩니다/g, '되지 않습니다'],
      [/있습니다/g, '없습니다'],
      [/됐습니다/g, '되지 않았습니다'],
      [/이다/g, '이(가) 아니다'],
      [/한다/g, '하지 않는다'],
    ];

    let negated = text;
    let applied = false;
    for (const [pattern, replacement] of negations) {
      if (pattern.test(negated)) {
        negated = negated.replace(pattern, replacement);
        applied = true;
        break; // Apply only the first matching negation to avoid double negation
      }
    }

    // English fallback
    if (!applied) {
      // Simple English negation
      if (negated.toLowerCase().includes(' is ')) {
        negated = negated.replace(/ is /gi, ' is not ');
      } else if (negated.toLowerCase().includes(' are ')) {
        negated = negated.replace(/ are /gi, ' are not ');
      } else {
        negated = `It is not true that ${negated}`;
      }
    }

    return negated;
  }



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
