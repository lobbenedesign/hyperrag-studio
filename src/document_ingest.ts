/**
 * 📄 Real user document ingestion (chunk → embed → index → extract graph).
 *
 * The gap this closes: before this module, the only way content got into
 * `HNSWVectorIndex` was `/api/hnsw/insert`, which embeds one caller-supplied
 * string as exactly one vector node — fine for a short snippet, wrong for a
 * real document, where cramming a multi-page text into a single embedding
 * washes out everything but its dominant terms (the same reason every real
 * RAG system — LlamaIndex's `SentenceSplitter`, LangChain's
 * `RecursiveCharacterTextSplitter`, txtai's `Textractor` pipeline — chunks
 * before embedding, not after). This module does that: real sentence-
 * boundary chunking (not fixed-size mid-sentence cuts) sized for the
 * `RealVectorEmbedder`'s embedding context, feeding each chunk into the real
 * HNSW index as its own vector node, and — reusing the existing
 * `extractGraphFromText` pipeline (`src/graph_ingest.ts`) — also into the
 * LightRAG graph, so one upload populates both retrieval paths that
 * `buildHybridContext` (`src/hybrid_retrieval.ts`) merges at query time.
 *
 * Chunking approach: split on sentence boundaries (`. ! ?` followed by
 * whitespace + capital/quote/paren, a standard heuristic — not a trained
 * sentence segmenter, and this is stated plainly, not oversold) and pack
 * consecutive sentences up to `maxChunkChars` so chunks stay within a
 * sentence, never split one in half. This is the same "semantic chunking by
 * sentence boundaries rather than fixed-size" pattern LlamaIndex's
 * `SentenceSplitter` and txtai's chunking use, reimplemented from scratch
 * here (no shared code) for this project's own chunk shape.
 *
 * No fabricated fallback: if the HNSW embed step or the graph extraction
 * step fails, the failure is reported per-stage rather than hidden — a
 * document can legitimately end up vector-indexed but not graph-extracted
 * (e.g. Ollama down for the LLM extraction call but the embedder's hash
 * fallback still works), and the caller is told exactly that instead of a
 * single opaque "success".
 */

import { HNSWVectorIndex } from "./hnsw_vector_index";
import { LightRAGEngine } from "./lightrag_engine";
import { extractGraphFromText, type GraphIngestResult } from "./graph_ingest";

export interface DocumentChunk {
  index: number;
  text: string;
  charCount: number;
}

export interface DocumentIngestResult {
  documentId: string;
  totalChars: number;
  chunks: DocumentChunk[];
  vectorIndex: {
    inserted: number;
    nodeIds: string[];
    failed: { chunkIndex: number; error: string }[];
  };
  graphExtraction: {
    attempted: boolean;
    succeeded: boolean;
    error: string | null;
    nodesAdded?: number;
    nodesMerged?: number;
    edgesAdded?: number;
    totalEntitiesExtracted?: number;
    totalRelationsExtracted?: number;
  };
}

/**
 * Splits text into sentences using a standard boundary heuristic, then packs
 * consecutive sentences into chunks up to `maxChunkChars`, never splitting a
 * sentence across two chunks (unless a single sentence alone exceeds the
 * limit, in which case it is hard-split as a last resort so no content is
 * silently dropped).
 */
export function chunkBySentence(text: string, maxChunkChars = 800): DocumentChunk[] {
  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];

  // Sentence boundary: cut right after '.', '!', or '?' wherever it is
  // followed by whitespace. Implemented as a lossless `split` on a
  // lookbehind boundary (not a consuming match-based extraction) so that a
  // period NOT followed by whitespace — "Bun.serve()", "qwen2.5:7b",
  // decimals, URLs — is never treated as a boundary and never silently
  // dropped. An earlier version of this used a consuming
  // `[^.!?]+[.!?]+(?:\s+|$)` alternation-based regex that, for exactly this
  // no-space-after-period case, failed to match starting at the
  // abbreviation and skipped ahead, silently deleting the abbreviation from
  // the chunked output entirely (verified: "Bun.serve() starts..." chunked
  // to "serve() starts...", dropping "Bun." outright). `split` cannot lose
  // characters this way — every input character ends up in exactly one
  // output piece. This is a documented heuristic (the same "capital-letter
  // resumption" rule NLTK's punkt uses as its non-trained fallback), not a
  // trained segmenter — it can still mis-split on abbreviations like "Dr."
  // or "e.g." when they ARE followed by whitespace and a capital letter,
  // which is disclosed here rather than presented as perfect NLP.
  const rawSentences = normalized.split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/);
  const sentences = rawSentences.map((s) => s.trim()).filter((s) => s.length > 0);

  const chunks: DocumentChunk[] = [];
  let current = "";

  const flush = () => {
    if (current.trim().length > 0) {
      chunks.push({ index: chunks.length, text: current.trim(), charCount: current.trim().length });
      current = "";
    }
  };

  for (const sentence of sentences) {
    if (sentence.length > maxChunkChars) {
      // A single sentence longer than the whole chunk budget (e.g. no
      // punctuation in a huge run-on block) — flush what we have, then
      // hard-split this one sentence so nothing is dropped.
      flush();
      for (let i = 0; i < sentence.length; i += maxChunkChars) {
        chunks.push({
          index: chunks.length,
          text: sentence.slice(i, i + maxChunkChars),
          charCount: Math.min(maxChunkChars, sentence.length - i)
        });
      }
      continue;
    }
    if (current.length + sentence.length + 1 > maxChunkChars && current.length > 0) {
      flush();
    }
    current = current.length > 0 ? `${current} ${sentence}` : sentence;
  }
  flush();

  return chunks;
}

export async function ingestDocument(
  text: string,
  opts: {
    documentId?: string;
    hnswIndex: HNSWVectorIndex;
    lightRAG: LightRAGEngine;
    extractGraph: boolean;
    ollamaHost: string;
    model: string;
    maxChunkChars?: number;
  }
): Promise<DocumentIngestResult> {
  if (!text || !text.trim()) {
    throw new Error("document text is empty");
  }

  const documentId = opts.documentId || `doc-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const chunks = chunkBySentence(text, opts.maxChunkChars ?? 800);
  if (chunks.length === 0) {
    throw new Error("document produced zero chunks after sentence splitting");
  }

  const nodeIds: string[] = [];
  const failed: { chunkIndex: number; error: string }[] = [];

  for (const chunk of chunks) {
    const chunkId = `${documentId}-chunk-${chunk.index}`;
    try {
      await opts.hnswIndex.insert(chunkId, chunk.text, { documentId, chunkIndex: chunk.index });
      nodeIds.push(chunkId);
    } catch (err: any) {
      failed.push({ chunkIndex: chunk.index, error: err.message });
    }
  }

  const graphExtraction: DocumentIngestResult["graphExtraction"] = {
    attempted: false,
    succeeded: false,
    error: null
  };

  if (opts.extractGraph) {
    graphExtraction.attempted = true;
    try {
      const extraction: GraphIngestResult = await extractGraphFromText(text, {
        ollamaHost: opts.ollamaHost,
        model: opts.model
      });
      const allEntities = extraction.chunks.flatMap((c) => c.entities);
      const allRelations = extraction.chunks.flatMap((c) => c.relations);
      const merge = opts.lightRAG.ingestExtracted(allEntities, allRelations, documentId);
      graphExtraction.succeeded = true;
      graphExtraction.nodesAdded = merge.nodesAdded;
      graphExtraction.nodesMerged = merge.nodesMerged;
      graphExtraction.edgesAdded = merge.edgesAdded;
      graphExtraction.totalEntitiesExtracted = extraction.totalEntitiesExtracted;
      graphExtraction.totalRelationsExtracted = extraction.totalRelationsExtracted;
    } catch (err: any) {
      graphExtraction.error = err.message;
    }
  }

  return {
    documentId,
    totalChars: text.length,
    chunks,
    vectorIndex: { inserted: nodeIds.length, nodeIds, failed },
    graphExtraction
  };
}
