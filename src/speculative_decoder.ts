/**
 * 🦅 Speculative Decoding Benchmark (Draft/Target Empirical Approximation)
 *
 * HONESTY NOTE: This does NOT implement true EAGLE/Medusa speculative decoding.
 * Real EAGLE requires access to the target model's internal hidden states and a
 * feature-level draft head trained on them, plus tree-attention parallel
 * verification inside the inference engine itself. Ollama's HTTP API exposes
 * neither hidden states nor logit-level draft verification, so that mechanism
 * cannot be genuinely built on top of it.
 *
 * What this module DOES do for real, against a live local Ollama server:
 *  1. Runs the SAME prompt on a small "draft" model and a larger "target" model.
 *  2. Measures REAL wall-clock throughput (tokens/sec) for both, using Ollama's
 *     own `eval_count` / `eval_duration` counters returned per request.
 *  3. Computes a REAL token-overlap "acceptance rate" by diffing the draft
 *     model's actual generated tokens against the target model's actual
 *     generated tokens (word-level LCS-style prefix/greedy match) — i.e. how
 *     much of what the small model produced would plausibly have been
 *     "accepted" if it were used as a draft, approximated empirically rather
 *     than via true logit verification.
 *  4. Reports the genuine speedup you would observe if you simply used the
 *     draft model's throughput vs the target model's throughput (a real,
 *     measurable floor for what speculative decoding tries to approach).
 *
 * If Ollama is unreachable, this throws — callers must surface the failure
 * instead of fabricating numbers.
 */

export interface SpeculativeBenchmarkResult {
  mode: "live-ollama-draft-target" | "live-localai-draft-target";
  draftModel: string;
  targetModel: string;
  draftTokensPerSec: number;
  targetTokensPerSec: number;
  measuredSpeedupFactor: string;
  tokenOverlapAcceptanceRate: string;
  draftTokensGenerated: number;
  targetTokensGenerated: number;
  draftLatencyMs: number;
  targetLatencyMs: number;
  draftOutputPreview: string;
  targetOutputPreview: string;
  notes: string;
}

interface OllamaGenerateResponse {
  response: string;
  eval_count?: number;
  eval_duration?: number; // nanoseconds
}

export class SpeculativeDecodingEngine {
  private draftTreeDepth: number;
  private ollamaHost: string;
  private draftModel: string;

  constructor(draftTreeDepth = 4, ollamaHost = "http://localhost:11434", draftModel = "granite3-dense:2b") {
    this.draftTreeDepth = draftTreeDepth;
    this.ollamaHost = ollamaHost;
    this.draftModel = draftModel;
  }

  /**
   * Ollama's native /api/generate reports its own internal eval_count/
   * eval_duration (pure generation time, excluding model load & network),
   * which is what this used exclusively before. LocalAI's OpenAI-compatible
   * /v1/chat/completions doesn't expose that internal timer in its response
   * envelope, so its tokens/sec is computed from wall-clock latency (already
   * measured here) divided into the real completion_tokens count from
   * `usage` — a real measurement, just of a slightly different quantity
   * (wall clock includes network + connection overhead that Ollama's
   * internal counter excludes). Labelled honestly via `mode` and `notes`
   * rather than presented as identical to the Ollama numbers.
   */
  private async generate(model: string, prompt: string): Promise<{ text: string; tokens: number; tps: number; latencyMs: number }> {
    const backend = (process.env.LLM_BACKEND || "ollama").trim().toLowerCase() === "localai" ? "localai" : "ollama";

    if (backend === "localai") {
      const localaiHost = process.env.LOCALAI_HOST || "http://localhost:8080";
      const start = performance.now();
      const res = await fetch(`${localaiHost}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          max_tokens: 96,
          stream: false
        }),
        signal: AbortSignal.timeout(60000)
      });
      const latencyMs = performance.now() - start;
      if (!res.ok) {
        throw new Error(`LocalAI returned HTTP ${res.status} for model ${model}`);
      }
      const data: any = await res.json();
      const text = data.choices?.[0]?.message?.content || "";
      const tokens = data.usage?.completion_tokens || 0;
      const durationSec = latencyMs / 1000;
      const tps = durationSec > 0 && tokens > 0 ? Number((tokens / durationSec).toFixed(2)) : 0;
      return { text, tokens, tps, latencyMs: Number(latencyMs.toFixed(1)) };
    }

    const start = performance.now();
    const res = await fetch(`${this.ollamaHost}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, prompt, stream: false, options: { num_predict: 96 } }),
      signal: AbortSignal.timeout(60000)
    });
    const latencyMs = performance.now() - start;
    if (!res.ok) {
      throw new Error(`Ollama returned HTTP ${res.status} for model ${model}`);
    }
    const data = (await res.json()) as OllamaGenerateResponse;
    const tokens = data.eval_count || 0;
    const durationSec = (data.eval_duration || 0) / 1e9;
    const tps = durationSec > 0 && tokens > 0 ? Number((tokens / durationSec).toFixed(2)) : 0;
    return { text: data.response || "", tokens, tps, latencyMs: Number(latencyMs.toFixed(1)) };
  }

  /**
   * Real word-level greedy overlap between the draft and target generations,
   * used as an empirical proxy for "how often would the draft's guess have
   * been accepted". This is NOT logit-level verification.
   */
  private computeTokenOverlap(draftText: string, targetText: string): number {
    const dWords = draftText.trim().split(/\s+/).filter(Boolean);
    const tWords = targetText.trim().split(/\s+/).filter(Boolean);
    if (dWords.length === 0 || tWords.length === 0) return 0;

    let matched = 0;
    const len = Math.min(dWords.length, tWords.length);
    for (let i = 0; i < len; i++) {
      if (dWords[i].toLowerCase() === tWords[i].toLowerCase()) matched++;
    }
    return Number((matched / len).toFixed(3));
  }

  /**
   * Executes a real draft-vs-target throughput comparison against local Ollama.
   * Throws if Ollama (or either model) is unreachable — no fabricated fallback.
   */
  public async benchmark(prompt: string, targetModel: string = "qwen2.5:7b"): Promise<SpeculativeBenchmarkResult> {
    const cleanPrompt = prompt.trim() || "Generate high performance code";

    const [draft, target] = await Promise.all([
      this.generate(this.draftModel, cleanPrompt),
      this.generate(targetModel, cleanPrompt)
    ]);

    const acceptance = this.computeTokenOverlap(draft.text, target.text);
    const speedup = target.tps > 0 ? draft.tps / target.tps : 0;
    const backend = (process.env.LLM_BACKEND || "ollama").trim().toLowerCase() === "localai" ? "localai" : "ollama";

    return {
      mode: backend === "localai" ? "live-localai-draft-target" : "live-ollama-draft-target",
      draftModel: this.draftModel,
      targetModel,
      draftTokensPerSec: draft.tps,
      targetTokensPerSec: target.tps,
      measuredSpeedupFactor: `${speedup.toFixed(2)}x`,
      tokenOverlapAcceptanceRate: `${(acceptance * 100).toFixed(1)}%`,
      draftTokensGenerated: draft.tokens,
      targetTokensGenerated: target.tokens,
      draftLatencyMs: draft.latencyMs,
      targetLatencyMs: target.latencyMs,
      draftOutputPreview: draft.text.slice(0, 160),
      targetOutputPreview: target.text.slice(0, 160),
      notes: backend === "localai"
        ? "Empirical draft-vs-target throughput/overlap measured live against LocalAI. Tokens/sec here is wall-clock latency / real completion_tokens (LocalAI's OpenAI-compatible response doesn't expose Ollama's internal eval_duration timer), so it includes request/connection overhead the Ollama-mode numbers don't. Not true EAGLE/Medusa logit-level speculative verification (neither API exposes the hidden state / draft-head hooks that would require)."
        : "Empirical draft-vs-target throughput/overlap measured live against Ollama. Not true EAGLE/Medusa logit-level speculative verification (Ollama's API does not expose the hidden state / draft-head hooks that would require)."
    };
  }
}
