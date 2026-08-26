/**
 * 🎯 Real LLM-as-Reranker
 *
 * The gap this fills: `hybrid_retrieval.ts` merges two *independently*
 * ranked candidate lists — graph keyword-overlap score and HNSW cosine
 * similarity — into one context block, but never actually compares
 * candidates from the two sources against each other on one shared scale.
 * A graph match that only scored 2 keyword-overlap points can end up ahead
 * of a vector chunk that's genuinely more relevant, purely because they
 * were never scored by the same yardstick.
 *
 * Real LightRAG (github.com/HKUDS/LightRAG) doesn't do this either — its
 * "hybrid mode" is the same kind of independent-list merge this project's
 * own `hybrid_retrieval.ts` already replicates. This module adds the piece
 * neither does: the "LLM-as-reranker" pattern documented across production
 * RAG systems (cross-encoder / pointwise LLM reranking — see e.g. Cohere
 * Rerank, BGE-reranker, or a local Ollama model used the same way) — for
 * each candidate, ask the model to output ONE relevance score for
 * (query, candidate) on a fixed 0-10 scale, then re-sort by that score.
 *
 * This is real: every candidate gets a real, separate Ollama /api/chat
 * call, `format: "json"` constrained, with the actual query and actual
 * candidate text. No score is fabricated or estimated locally — if Ollama
 * is unreachable or a call fails, that candidate is honestly marked
 * `rerankFailed: true` and sinks to the bottom rather than being assigned a
 * fake score. Because this is one real model call per candidate, it costs
 * real, measurable latency (roughly candidates × single-call latency, since
 * calls run sequentially against one local Ollama instance rather than
 * pretending unlimited concurrent throughput) — the caller gets that timing
 * back honestly instead of a hidden cost.
 */

export interface RerankCandidate {
  id: string;
  text: string;
  source: "vector" | "graph";
  originalScore?: number; // cosine similarity (vector) or keyword score (graph), if known
  originalRank: number; // 1-indexed rank within its own source list, before reranking
}

export interface RerankedCandidate extends RerankCandidate {
  llmRelevanceScore: number | null; // 0-10, null if this candidate's call failed
  rerankFailed: boolean;
  rerankError?: string;
  newRank: number; // 1-indexed rank after re-sorting by llmRelevanceScore
  rankDelta: number; // originalRank - newRank; positive = moved up, negative = moved down
  callLatencyMs: number;
}

export interface RerankResult {
  model: string;
  query: string;
  candidates: RerankedCandidate[];
  totalLatencyMs: number;
  candidatesScored: number;
  candidatesFailed: number;
  rankChanged: boolean; // true iff the resulting order genuinely differs from the input order
}

const RERANK_SYSTEM_PROMPT = `You are a strict relevance judge for a retrieval system.
Given a SEARCH QUERY and a CANDIDATE TEXT chunk, score how relevant the candidate is to
answering the query, on an integer scale from 0 to 10:
- 0-2: irrelevant, shares no meaningful topic with the query
- 3-4: tangentially related, mentions a shared term but doesn't help answer the query
- 5-6: partially relevant, touches the topic but is incomplete or indirect
- 7-8: relevant, directly useful context for answering the query
- 9-10: highly relevant, directly and specifically answers or is essential to the query
Return ONLY a JSON object of exactly this shape, no prose: {"score": <integer 0-10>}`;

/**
 * Scores one candidate against the query with a single real Ollama call.
 * Never fabricates a score: throws on any failure, caller decides how to
 * record that honestly.
 */
async function scoreCandidate(
  query: string,
  candidate: RerankCandidate,
  opts: { ollamaHost: string; model: string; timeoutMs: number }
): Promise<number> {
  const res = await fetch(`${opts.ollamaHost}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: opts.model,
      format: "json",
      messages: [
        { role: "system", content: RERANK_SYSTEM_PROMPT },
        {
          role: "user",
          content: `SEARCH QUERY: ${query}\n\nCANDIDATE TEXT: ${candidate.text}`
        }
      ],
      stream: false,
      options: { temperature: 0 } // deterministic-as-possible judging, not creative generation
    }),
    signal: AbortSignal.timeout(opts.timeoutMs)
  });

  if (!res.ok) {
    throw new Error(`Ollama returned HTTP ${res.status} while reranking candidate "${candidate.id}"`);
  }

  const data: any = await res.json();
  const raw: string = data.message?.content ?? "";
  if (!raw) {
    throw new Error(`Ollama returned no content while reranking candidate "${candidate.id}"`);
  }

  let parsed: any;
  try {
    parsed = JSON.parse(raw);
  } catch (e: any) {
    throw new Error(`Rerank response for "${candidate.id}" was not valid JSON despite format:"json": ${e.message}`);
  }

  const score = Number(parsed.score);
  if (!Number.isFinite(score)) {
    throw new Error(`Rerank response for "${candidate.id}" had no numeric "score" field (got: ${JSON.stringify(parsed)})`);
  }
  return Math.max(0, Math.min(10, score));
}

/**
 * Reranks a set of retrieval candidates (from graph + vector sources) by
 * real LLM-judged relevance to the query. Sequential, not parallel — this
 * hits the same local Ollama instance one candidate at a time, which is
 * both realistic (one local GPU/CPU can't usefully serve many concurrent
 * generation requests) and gives an honest per-candidate latency number
 * instead of an artificially compressed parallel-call timing.
 */
export async function rerankCandidates(
  query: string,
  candidates: RerankCandidate[],
  opts: { ollamaHost: string; model: string; timeoutMs?: number }
): Promise<RerankResult> {
  const timeoutMs = opts.timeoutMs ?? 30000;
  const startedAt = Date.now();

  const scored: Omit<RerankedCandidate, "newRank" | "rankDelta">[] = [];
  for (const c of candidates) {
    const callStart = Date.now();
    try {
      const score = await scoreCandidate(query, c, { ollamaHost: opts.ollamaHost, model: opts.model, timeoutMs });
      scored.push({
        ...c,
        llmRelevanceScore: score,
        rerankFailed: false,
        callLatencyMs: Date.now() - callStart
      });
    } catch (err: any) {
      scored.push({
        ...c,
        llmRelevanceScore: null,
        rerankFailed: true,
        rerankError: err.message,
        callLatencyMs: Date.now() - callStart
      });
    }
  }

  // Sort by LLM relevance score descending. Failed candidates (null score)
  // sink to the bottom, ordered by their original rank among themselves —
  // they are not silently dropped, just honestly deprioritized since we
  // have no real relevance judgment for them.
  const sorted = [...scored].sort((a, b) => {
    if (a.llmRelevanceScore === null && b.llmRelevanceScore === null) return a.originalRank - b.originalRank;
    if (a.llmRelevanceScore === null) return 1;
    if (b.llmRelevanceScore === null) return -1;
    return b.llmRelevanceScore - a.llmRelevanceScore;
  });

  const finalCandidates: RerankedCandidate[] = sorted.map((c, i) => ({
    ...c,
    newRank: i + 1,
    rankDelta: c.originalRank - (i + 1)
  }));

  const rankChanged = finalCandidates.some((c) => c.rankDelta !== 0);

  return {
    model: opts.model,
    query,
    candidates: finalCandidates,
    totalLatencyMs: Date.now() - startedAt,
    candidatesScored: finalCandidates.filter((c) => !c.rerankFailed).length,
    candidatesFailed: finalCandidates.filter((c) => c.rerankFailed).length,
    rankChanged
  };
}

/**
 * Builds the flat, source-tagged candidate list a rerank pass operates on,
 * from the same graph low-level matches + HNSW vector matches that
 * `buildHybridContext` already merges — this is the "initial candidates
 * (keyword + HNSW)" the reranker re-scores on one shared scale.
 */
export function buildRerankCandidates(
  lowLevelGraphMatches: { id: string; name: string; description: string }[],
  vectorMatches: { id: string; text: string; similarity: number }[]
): RerankCandidate[] {
  const graphCandidates: RerankCandidate[] = lowLevelGraphMatches.map((n, i) => ({
    id: n.id,
    text: `${n.name}: ${n.description}`,
    source: "graph",
    originalRank: i + 1
  }));
  const vectorCandidates: RerankCandidate[] = vectorMatches.map((v, i) => ({
    id: v.id,
    text: v.text,
    source: "vector",
    originalScore: v.similarity,
    originalRank: i + 1
  }));
  return [...graphCandidates, ...vectorCandidates];
}
