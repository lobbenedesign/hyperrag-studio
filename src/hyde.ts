/**
 * 🌫️ HyDE — Hypothetical Document Embeddings (query expansion for vector search)
 *
 * Source and gap. Real technique from Gao, Ma, Lin & Callan, "Precise
 * Zero-Shot Dense Retrieval without Relevance Labels" (arXiv:2212.10496,
 * ACL 2023) — verified via arxiv.org/abs/2212.10496, not assumed from
 * memory. The paper's method: given a query, zero-shot-prompt an
 * instruction-following LLM to write a hypothetical answer/document (which
 * may itself contain hallucinated specifics — that's fine, it's never
 * shown to the user), then embed THAT generated text instead of the raw
 * query and use its embedding to search the corpus. The intuition, straight
 * from the paper: a short query and a long passage that answers it often
 * sit far apart in embedding space even when the passage is exactly what
 * the query wants, because embedding models are better at document-to-
 * document similarity than short-query-to-document similarity. A
 * fabricated *answer-shaped* passage embeds much closer to a real matching
 * passage than the bare query does. It's also documented as a real,
 * shipped feature of Haystack (docs.haystack.deepset.ai/docs/
 * hypothetical-document-embeddings-hyde) and LangChain's HypotheticalDocumentEmbedder,
 * so this is not a one-off research toy.
 *
 * The gap in this project: every retrieval mode in `hybrid_retrieval.ts`
 * and `/api/query` embeds the user's raw prompt string directly via
 * `HNSWVectorIndex.search(prompt, k)`. For short, terse queries (the kind a
 * human actually types — "how does X relate to Y", not a full paragraph)
 * against this project's own longer descriptive text chunks
 * (`document_ingest.ts` sentence-boundary chunks, `server.ts`'s seeded
 * `chunk-1..5`), that query/passage embedding mismatch is exactly what
 * HyDE targets.
 *
 * Implementation notes (own code, not copied from any repo):
 * - `generateHypotheticalDocument()` makes ONE real Ollama `/api/chat` call
 *   asking the model to write a short, direct passage that plausibly
 *   *answers* the query, framed as an encyclopedia/documentation entry (the
 *   framing the paper itself uses for the passage-retrieval tasks, e.g. its
 *   "Write a passage that answers the question" prompt template on
 *   fact-based datasets) — not the query re-stated, not a list of
 *   keywords.
 * - No fabricated fallback: if Ollama is unreachable or errors, this
 *   throws rather than silently returning the raw query re-labeled as a
 *   "hypothetical document" — callers can decide whether to degrade to a
 *   plain vector search.
 * - Embedding and search themselves are unchanged: the hypothetical text is
 *   handed to the EXISTING real `HNSWVectorIndex.search()` /
 *   `RealVectorEmbedder`, so a HyDE search still goes through the same real
 *   cosine-similarity HNSW graph as every other mode — only the text that
 *   gets embedded changes.
 */

export interface HydeResult {
  originalQuery: string;
  hypotheticalDocument: string;
  model: string;
  latencyMs: number;
}

/**
 * Generates one hypothetical, answer-shaped passage for `query` via a real
 * local Ollama chat call. This text is never shown to the end user as a
 * real answer — it exists only to be embedded and used as the vector
 * search query, per the HyDE paper's method.
 */
export async function generateHypotheticalDocument(
  query: string,
  options: { ollamaHost: string; model: string }
): Promise<HydeResult> {
  const trimmed = query.trim();
  if (!trimmed) {
    throw new Error("HyDE: query must be non-empty");
  }

  const start = performance.now();

  const res = await fetch(`${options.ollamaHost}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: options.model,
      stream: false,
      options: { temperature: 0.3 },
      messages: [
        {
          role: "system",
          content:
            "You write short hypothetical documentation passages for a technical retrieval system. " +
            "Given a question, write a single short passage (2-4 sentences) that DIRECTLY ANSWERS it, " +
            "in the confident, factual style of a technical doc or encyclopedia entry — even if you are " +
            "not fully certain of the specifics. Do not mention that it is hypothetical. Do not ask " +
            "clarifying questions. Output only the passage, no preamble."
        },
        { role: "user", content: trimmed }
      ]
    }),
    signal: AbortSignal.timeout(30000)
  });

  if (!res.ok) {
    throw new Error(`HyDE: Ollama returned HTTP ${res.status} while generating hypothetical document`);
  }

  const data: any = await res.json();
  const text: string = data?.message?.content?.trim() || "";
  if (!text) {
    throw new Error("HyDE: Ollama returned an empty hypothetical document");
  }

  return {
    originalQuery: trimmed,
    hypotheticalDocument: text,
    model: options.model,
    latencyMs: Math.round(performance.now() - start)
  };
}
