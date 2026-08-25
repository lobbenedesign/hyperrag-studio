/**
 * 🧬 DSPy-Inspired Prompt Signature Compiler
 *
 * HONESTY NOTE: This is NOT the real DSPy library and does not run DSPy's
 * actual BootstrapFewShot teleprompter (which trains against a labeled
 * dataset with a real metric function over many trials). What this module
 * does for real:
 *
 *  1. Structural compilation: turns a typed signature (name/description/
 *     inputs/outputs) into an explicit instruction-following prompt.
 *  2. Live bootstrap demonstration: if a local Ollama server is reachable,
 *     it actually calls the model once on the raw prompt to capture a REAL
 *     generated example, and uses that real output as the one-shot
 *     demonstration embedded in the compiled prompt (instead of a canned
 *     placeholder string).
 *  3. Live A/B scoring: if Ollama is reachable, it actually sends BOTH the
 *     naive prompt and the compiled prompt to the model and scores the two
 *     REAL responses with a measurable structural-adherence heuristic
 *     (does the output honor the requested format / include the requested
 *     output field / avoid empty responses). This is a real, computed
 *     comparison of live model behavior, not a hardcoded percentage.
 *  4. Offline fallback: if Ollama is unreachable, it returns a structural
 *     completeness heuristic for the prompt text alone, explicitly labeled
 *     as "not live-evaluated" rather than presenting it as a measured
 *     accuracy gain.
 */

export interface DSPySignature {
  name: string;
  description?: string;
  inputs?: { field: string; type: string; desc?: string }[];
  outputs?: { field: string; type: string; desc?: string }[];
}

export interface DSPyCompiledPipeline {
  signature: string;
  originalPrompt: string;
  compiledPrompt: string;
  liveEvaluated: boolean;
  baselineAccuracy: number;
  compiledAccuracy: number;
  accuracyGain: string;
  optimizationMetric: string;
  baselineResponsePreview?: string;
  compiledResponsePreview?: string;
  notes: string;
}

export class DSPyCompilerEngine {
  private ollamaHost: string;
  private model: string;

  constructor(ollamaHost = "http://localhost:11434", model = "qwen2.5:7b") {
    this.ollamaHost = ollamaHost;
    this.model = model;
  }

  private async callOllama(prompt: string): Promise<string | null> {
    try {
      const res = await fetch(`${this.ollamaHost}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.model, prompt, stream: false, options: { num_predict: 150 } }),
        signal: AbortSignal.timeout(45000)
      });
      if (!res.ok) return null;
      const data: any = await res.json();
      return typeof data.response === "string" ? data.response : null;
    } catch {
      return null;
    }
  }

  /**
   * Measurable structural-adherence score for a given output against the
   * requested signature (0.0 - 1.0). Real, deterministic function of the
   * actual text — not a fixed constant.
   */
  private scoreOutput(sig: DSPySignature, output: string): number {
    if (!output || output.trim().length === 0) return 0;
    let score = 0.3; // non-empty baseline
    const lower = output.toLowerCase();

    const wantsJson = (sig.outputs || []).some(o => o.type === "json" || o.type === "object");
    if (wantsJson) {
      try {
        JSON.parse(output.trim());
        score += 0.3;
      } catch {
        if (output.includes("{") && output.includes("}")) score += 0.1;
      }
    } else {
      score += 0.15; // no strict format requirement to fail
    }

    for (const outField of sig.outputs || []) {
      if (lower.includes(outField.field.toLowerCase())) score += 0.1;
    }

    if (output.trim().length > 20 && output.trim().length < 4000) score += 0.15;
    if (/```/.test(output)) score += 0.1;

    return Math.max(0, Math.min(1, Number(score.toFixed(3))));
  }

  private buildCompiledPrompt(sig: DSPySignature, prompt: string, liveDemo: string | null): string {
    const inputsDesc = (sig.inputs || []).map(i => `  - ${i.field} (${i.type}): ${i.desc || ""}`).join("\n") || "  - (none specified)";
    const outputsDesc = (sig.outputs || []).map(o => `  - ${o.field} (${o.type}): ${o.desc || ""}`).join("\n") || "  - (none specified)";

    const demoBlock = liveDemo
      ? `-- LIVE DEMONSTRATION (real model output, captured this run) --\nInput: ${prompt}\nOutput: ${liveDemo.trim().slice(0, 500)}\n\n`
      : `-- DEMONSTRATION --\n(No live model available to bootstrap a real example this run; structural template only.)\n\n`;

    return `[DSPy-Inspired Compiled Signature: ${sig.name}]\n` +
      `${sig.description ? `Task: ${sig.description}\n` : ""}` +
      `-- INPUT FIELDS --\n${inputsDesc}\n\n` +
      `-- OUTPUT FIELDS --\n${outputsDesc}\n\n` +
      demoBlock +
      `-- TASK INPUT --\n${prompt}\n\n` +
      `-- INSTRUCTIONS --\n` +
      `Respond ONLY with the requested output field(s), following the exact type/format above. Be precise and complete.`;
  }

  /**
   * Compiles a raw prompt into a structured, few-shot-augmented prompt for
   * the given signature, and (when Ollama is reachable) live-scores the
   * improvement against the same model's real output on the naive prompt.
   */
  public async compileSignature(signature: DSPySignature | string, prompt: string): Promise<DSPyCompiledPipeline> {
    const sig: DSPySignature = typeof signature === "string" ? { name: signature } : signature;
    const sigLabel = sig.name || "UnnamedSignature";

    // Attempt live bootstrap: one real call to capture a genuine demonstration.
    const liveDemo = await this.callOllama(prompt);
    const compiledPrompt = this.buildCompiledPrompt(sig, prompt, liveDemo);

    if (liveDemo !== null) {
      // Live A/B: score the naive-prompt output we already have (liveDemo)
      // against a fresh real call using the compiled prompt.
      const compiledResponse = await this.callOllama(compiledPrompt);
      const baselineAccuracy = this.scoreOutput(sig, liveDemo);
      const compiledAccuracy = compiledResponse !== null ? this.scoreOutput(sig, compiledResponse) : baselineAccuracy;
      const gain = baselineAccuracy > 0 ? ((compiledAccuracy - baselineAccuracy) / baselineAccuracy) * 100 : 0;

      return {
        signature: sigLabel,
        originalPrompt: prompt,
        compiledPrompt,
        liveEvaluated: true,
        baselineAccuracy,
        compiledAccuracy,
        accuracyGain: `${gain >= 0 ? "+" : ""}${gain.toFixed(1)}%`,
        optimizationMetric: "Live structural-adherence heuristic scored against real Ollama responses (not a labeled-dataset accuracy metric)",
        baselineResponsePreview: liveDemo.slice(0, 200),
        compiledResponsePreview: (compiledResponse || "").slice(0, 200),
        notes: `Live-evaluated against local Ollama model '${this.model}'. Scores reflect structural adherence to the requested signature format on this one real generation, not a benchmarked task-accuracy metric.`
      };
    }

    // Offline fallback: no live model available. Score prompt structure only,
    // and say so honestly instead of presenting it as a measured gain.
    const hasInstructions = /istruzioni|step|instructions/i.test(prompt);
    const hasFormat = /json|output/i.test(prompt);
    const baselineAccuracy = Number((0.3 + (hasInstructions ? 0.1 : 0) + (hasFormat ? 0.1 : 0)).toFixed(2));
    const compiledAccuracy = Number((baselineAccuracy + 0.2).toFixed(2));
    const gain = baselineAccuracy > 0 ? ((compiledAccuracy - baselineAccuracy) / baselineAccuracy) * 100 : 0;

    return {
      signature: sigLabel,
      originalPrompt: prompt,
      compiledPrompt,
      liveEvaluated: false,
      baselineAccuracy,
      compiledAccuracy,
      accuracyGain: `+${gain.toFixed(1)}%`,
      optimizationMetric: "Offline structural-completeness heuristic (no live model reachable)",
      notes: "Ollama was unreachable, so no live demonstration or scoring was performed. These numbers are a rough structural heuristic on the prompt text only, NOT a measured accuracy gain."
    };
  }
}
