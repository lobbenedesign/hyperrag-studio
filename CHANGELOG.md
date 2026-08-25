# Changelog

## Unreleased — real LLM-driven graph ingestion

**Gap and source.** The "Honest Status" table in the README (added during
the prior audit, see `1451e98`) called out that `src/lightrag_engine.ts`'s
graph was hand-seeded data, "not a real ingestion/indexing pipeline over a
codebase." Real LightRAG and Microsoft GraphRAG both build their knowledge
graphs from an LLM entity/relationship extraction pass over document
chunks — GraphRAG's own repo calls this step "element extraction"
(checked against https://github.com/microsoft/graphrag, not assumed from
memory). This project had a real HNSW index and real embeddings already,
but no equivalent for the graph side — the graph was still just seed data.

### Added
- `src/graph_ingest.ts` — `extractGraphFromText()`: chunks input text on
  paragraph boundaries, calls local Ollama's `/api/chat` with
  `format: "json"` and a schema-constrained extraction prompt per chunk,
  validates the response shape, and drops any relation whose endpoints
  aren't among the entities the model actually extracted (no dangling
  edges from a hallucinated relation target). Throws instead of returning
  a fake extraction if Ollama is unreachable, returns a non-2xx status, or
  its output doesn't parse as JSON.
- `LightRAGEngine.ingestExtracted()` (`src/lightrag_engine.ts`) — merges
  extracted entities/relations into the live graph, deduping by
  case-insensitive entity name so re-ingesting the same or overlapping
  text doesn't create duplicate nodes; reports `nodesAdded` vs
  `nodesMerged` vs `edgesAdded`.
- `POST /api/graph/ingest` (`server.ts`) — `{text, sourceLabel?, model?}`
  → runs the above and returns extraction + merge stats, or a
  `{success:false, error}` with HTTP 502 on any Ollama failure, or 400 on
  empty input.

### What this is not
Not GraphRAG's actual extraction prompts, and not its later
community-detection/summarization stages (Leiden clustering, hierarchical
community reports) — only the entity/relation extraction step, applied to
this project's existing simpler two-level (`low_level`/`high_level`) node
model.

### Verified (2026-08-25)
- `bun build server.ts --target=bun --outfile=/dev/null` — bundles clean,
  8 modules, no syntax/type errors.
- Started the real server (`bun server.ts`) against a real local Ollama
  (`qwen2.5:7b`, confirmed loaded via `ollama ps`) and hit
  `POST /api/graph/ingest` with a real two-paragraph technical text
  describing this project's own `HNSWVectorIndex`/`TurboQuantEngine`
  classes: `{"success":true,"chunksProcessed":1,"totalEntitiesExtracted":4,
  "totalRelationsExtracted":4,"merge":{"nodesAdded":4,"nodesMerged":0,
  "edgesAdded":4},"graphStats":{"nodes":10,"edges":9}}`. Confirmed via
  `GET /api/graph` that the extracted nodes/edges (e.g.
  `ing-hnswvectorindex-6 --[implements]--> ing-hierarchical-navigable-small-world-graph-7`)
  were actually present, and via `POST /api/query` with the prompt "how
  does HNSW approximate nearest neighbor search work" that the newly
  ingested nodes were retrieved as real low/high-level matches.
- Re-ingested the identical text: `{"nodesAdded":0,"nodesMerged":2,
  "edgesAdded":0}` — confirms the dedupe-by-name path actually merges
  instead of duplicating (partial, because the second extraction pass
  didn't re-produce every entity name verbatim — expected LLM
  non-determinism, not a bug in the merge logic, which correctly matched
  what it *did* re-extract).
- `model: "not-a-real-model-xyz"` → HTTP 502,
  `{"success":false,"error":"Ollama returned HTTP 404 while extracting
  entities from chunk 1/1"}` — a real Ollama 404 surfaced honestly, not a
  fabricated graph.
- `text: ""` → HTTP 400 validation error, no Ollama call made (confirmed no
  corresponding request in the server log).

## 1.0.0 — honesty audit (see README "Honest Status" for full detail)

Removed fabricated EAGLE/DSPy performance claims (3.5x speedup, +28.5%
accuracy) that didn't match the code. Fixed a real bug
(`dspyCompiler.compileSignature` was called but never defined — every
`/api/dspy/compile` request would have crashed). Replaced a silent
`/api/query` fallback that fabricated a fake "synthesis" string when Ollama
was unreachable with an honest `llmUsed:false` + `synthesisError`. Replaced
hardcoded `accelerationMultiplier`/`dspyOptimizationAvgGain` in
`/api/status` with real last-measured values, `"not yet measured"` until a
real benchmark has run. See commit `1451e98` and earlier history for detail.
