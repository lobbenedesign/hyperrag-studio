/**
 * 🔀 Hybrid Retrieval (graph + vector)
 *
 * Real LightRAG's documented "hybrid mode" merges its graph-based local/global
 * entity retrieval with vector retrieval of raw text chunks — see
 * github.com/hkuds/lightrag ("Hybrid Mode integrates local and global
 * retrieval methods... simultaneously leverages both structured knowledge
 * and unstructured text"). Before this module, `/api/query` in this project
 * only ever ran `LightRAGEngine.query()` (graph keyword-overlap matching);
 * the real `HNSWVectorIndex` + `RealVectorEmbedder` already existed in this
 * codebase but were only reachable via the separate `/api/hnsw/search`
 * endpoint, never combined with a graph query. This module does the merge
 * for real: it runs both retrieval paths against the same prompt and
 * assembles one combined context block, each source clearly labeled so
 * nothing here claims to be more than what it is (a graph match is still
 * keyword overlap over a small in-memory graph; a vector match is still
 * cosine similarity over a 384-dim embedding that may itself be a real
 * Ollama embedding or this project's deterministic hash-based fallback,
 * exactly as `RealVectorEmbedder` already documents).
 */

import type { GraphNode } from "./lightrag_engine";
import type { SearchResult } from "./hnsw_vector_index";

export interface HybridQueryResult {
  lowLevelMatches: GraphNode[];
  highLevelMatches: GraphNode[];
  relationalPaths: { from: string; to: string; relation: string }[];
  vectorMatches: SearchResult[];
  synthesizedContext: string;
}

export interface GraphQueryResult {
  lowLevelMatches: GraphNode[];
  highLevelMatches: GraphNode[];
  relationalPaths: { from: string; to: string; relation: string }[];
}

/**
 * Combines a graph query result with real HNSW vector search results into
 * one synthesized context block, in the same spirit as LightRAG's hybrid
 * mode: graph gives structured entity/relation context, vector search gives
 * raw-text-chunk context the graph's small hand-authored/ingested node set
 * won't otherwise surface.
 */
export function buildHybridContext(
  graph: GraphQueryResult,
  vectorMatches: SearchResult[]
): HybridQueryResult {
  const context = [
    `=== 🔀 HYBRID CONTEXT (graph + vector, LightRAG "hybrid mode" style) ===`,
    `[High-Level Architecture Themes — graph]:`,
    ...graph.highLevelMatches.map((h) => `• ${h.name}: ${h.description}`),
    `\n[Low-Level Concrete Entities — graph]:`,
    ...graph.lowLevelMatches.map(
      (l) => `• ${l.name} (${l.type}): ${l.description} [${l.fileLocation || "workspace"}]`
    ),
    `\n[Relational Topology — graph]:`,
    ...graph.relationalPaths
      .slice(0, 4)
      .map((e) => `• (${e.from}) --[${e.relation}]--> (${e.to})`),
    `\n[Retrieved Text Chunks — real HNSW cosine similarity search]:`,
    ...(vectorMatches.length > 0
      ? vectorMatches.map(
          (v) => `• [sim=${v.similarity.toFixed(3)}] (${v.id}) ${v.text}`
        )
      : ["• (no vector index chunks matched above the search's own ranking — index may be empty)"]),
    `=====================================`,
  ].join("\n");

  return {
    lowLevelMatches: graph.lowLevelMatches,
    highLevelMatches: graph.highLevelMatches,
    relationalPaths: graph.relationalPaths,
    vectorMatches,
    synthesizedContext: context,
  };
}

/**
 * 🔢 Reciprocal Rank Fusion (RRF)
 *
 * The weighted merge above (`buildHybridContext`) just concatenates the
 * graph section and the vector section under separate headings — it never
 * actually produces one fused ranking across both sources. RRF is the
 * standard, well-documented alternative used by real hybrid search systems
 * (OpenSearch, Elasticsearch, Azure AI Search, Weaviate all default to it
 * for combining a keyword/BM25 ranked list with a vector ranked list): for
 * each item, sum 1/(k + rank) across every ranked list it appears in, where
 * `rank` is its 1-indexed position in that list. Items ranking high in
 * multiple lists — or consistently well in one — float to the top; a low
 * rank in any one list barely hurts, so RRF is deliberately more robust to
 * one source's scoring being on a totally different scale than the other's
 * (graph keyword-overlap points vs. cosine similarity here), which is
 * exactly the mismatch `buildHybridContext`'s simple concatenation doesn't
 * address. k=60 is the standard default from the original RRF paper
 * (Cormack et al., 2009) and is what most of the systems above ship with.
 *
 * This is real rank fusion over the same two real ranked lists
 * (`LightRAGEngine.query()`'s keyword-scored low-level matches, and
 * `HNSWVectorIndex.search()`'s cosine-similarity-scored matches) — no
 * synthetic scores, no LLM call, pure arithmetic over real ranks.
 */
export interface RRFCandidate {
  id: string;
  text: string;
  sources: ("graph" | "vector")[];
  ranks: { source: "graph" | "vector"; rank: number }[];
  rrfScore: number;
}

export function reciprocalRankFusion(
  graphRanked: { id: string; name: string; description: string }[],
  vectorRanked: { id: string; text: string }[],
  k: number = 60
): RRFCandidate[] {
  const byId = new Map<string, RRFCandidate>();

  graphRanked.forEach((n, i) => {
    const rank = i + 1;
    byId.set(n.id, {
      id: n.id,
      text: `${n.name}: ${n.description}`,
      sources: ["graph"],
      ranks: [{ source: "graph", rank }],
      rrfScore: 1 / (k + rank),
    });
  });

  vectorRanked.forEach((v, i) => {
    const rank = i + 1;
    const existing = byId.get(v.id);
    if (existing) {
      existing.sources.push("vector");
      existing.ranks.push({ source: "vector", rank });
      existing.rrfScore += 1 / (k + rank);
    } else {
      byId.set(v.id, {
        id: v.id,
        text: v.text,
        sources: ["vector"],
        ranks: [{ source: "vector", rank }],
        rrfScore: 1 / (k + rank),
      });
    }
  });

  return Array.from(byId.values()).sort((a, b) => b.rrfScore - a.rrfScore);
}
