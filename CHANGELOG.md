# Changelog

## Unreleased — optional LocalAI backend for graph ingestion and the speculative-decoding benchmark

**Gap identified:** `src/graph_ingest.ts` and `src/speculative_decoder.ts` were both hardwired to Ollama's native API (`/api/chat` with `format:"json"`, and `/api/generate` reading `eval_count`/`eval_duration`). Neither is a URL-only swap to [LocalAI](https://github.com/mudler/LocalAI): the request shapes differ (OpenAI's `response_format:{type:"json_object"}` vs Ollama's `format:"json"`), the response envelopes differ (`choices[0].message.content` vs `message.content`), and LocalAI's OpenAI-compatible response doesn't expose Ollama's internal `eval_duration` timer at all.

**What was built:**
- `src/graph_ingest.ts`: extraction now goes through a new internal `callChatJSON()` that branches on `LLM_BACKEND` (`ollama` default, unchanged; `localai` targets `LOCALAI_HOST`, default `http://localhost:8080`, via `/v1/chat/completions` with `response_format:{type:"json_object"}`). Same per-chunk error context, same "never fabricate on a parse failure, throw instead" policy on both backends.
- `src/speculative_decoder.ts`: `generate()` now branches the same way. On LocalAI, tokens/sec is computed from wall-clock latency divided into the real `usage.completion_tokens` count, since LocalAI's response doesn't carry an internal generation-duration field the way Ollama's does — a real measurement, but of a slightly different quantity (it includes request/connection overhead Ollama's internal timer excludes), labelled honestly via the new `mode: "live-localai-draft-target"` value and an explicit note in the result, not presented as identical to the Ollama numbers.

**Verified live, not just typechecked:**
1. `tsc --noEmit` — clean.
2. `POST /api/speculative/bench` and `POST /api/graph/ingest` both re-run against this machine's real local Ollama instance with `LLM_BACKEND` unset — identical behaviour to before (`mode: "live-ollama-draft-target"`, real entity/relation extraction), confirming the refactor didn't change the default path.
3. Both endpoints re-run with `LLM_BACKEND=localai` and no LocalAI instance running in this environment — both failed honestly (`"Unable to connect..."` / `success:false`), no fabricated benchmark numbers or graph nodes.
4. No LocalAI instance was available in this environment to verify a real end-to-end LocalAI response (real tokens/sec, real extracted entities) — implemented against LocalAI's documented/source-confirmed OpenAI-compatible request/response shape, stated explicitly rather than hidden.

## Unreleased — real HyDE (Hypothetical Document Embeddings)

**Gap and source.** Real technique from Gao, Ma, Lin & Callan, "Precise
Zero-Shot Dense Retrieval without Relevance Labels" (arXiv:2212.10496, ACL
2023), also shipped in Haystack and LangChain. Every existing retrieval mode
embedded the user's raw prompt directly; HyDE instead asks an LLM to write a
short hypothetical answer passage first, and embeds/searches with that
instead, closing the query/passage embedding-space gap for short queries.

### Added
- `src/hyde.ts` — `generateHypotheticalDocument()`: one real Ollama
  `/api/chat` call that writes a 2-4 sentence answer-shaped passage for the
  query. Throws (rather than fabricating a fallback "hypothetical document")
  if Ollama is unreachable, errors, or returns empty output.
- `server.ts` — `POST /api/query` gains `"vector_hyde"` and `"hybrid_hyde"`
  modes: the vector-search half embeds the HyDE-generated passage instead of
  the raw prompt. Optional `"hydeModel"` field picks the generator (defaults
  to `activeModel`). On generation failure, degrades to plain raw-prompt
  vector search and reports `hydeError` — never silently swaps in a fake
  passage.

### Verified, 2026-08-26 — real local Ollama, 3 real test queries, `vector` vs `vector_hyde`
- 2 of 3 queries: HyDE generation succeeded (22.4s, 29.7s respectively with
  `llama3.2:3b`), and top-1 retrieval **matched plain vector search exactly**
  — no ranking improvement on this project's small, topically-distinct
  5-chunk toy corpus. Consistent with the paper's own framing that HyDE's
  benefit scales with corpus size/semantic gap, both of which this toy
  corpus lacks.
- 1 of 3 queries: HyDE generation itself timed out (30s), and the
  degraded-fallback vector search then *also* missed the embedder's
  separate 2.5s Ollama timeout (model-swap contention from the just-run
  HyDE call), cascading into this project's pre-existing hash-based
  embedding fallback and landing on a **different, wrong top-1 chunk** —
  a real, measured regression versus plain vector search, not a
  hypothetical failure mode.
- **Honest cost**: HyDE adds one full LLM generation (22-30s on this
  CPU-only dev machine, no GPU, `llama3.2:3b`) before every embedding call.
  Full comparison data and per-query similarity scores are in README.md's
  "Real HyDE" section.

## Unreleased — real LLM-as-reranker + retrieval mode selector + RRF

**Gap and source.** Researched the current (2026) real state of LightRAG,
LlamaIndex, and other local-first RAG frameworks specifically for reranking
approaches (cross-encoder/LLM-as-reranker, RRF for hybrid search) and mode
selectors (LightRAG documents `naive`/`local`/`global`/`hybrid`/`mix`,
verified against https://github.com/HKUDS/LightRAG). This project's
`hybrid_retrieval.ts` already merged graph keyword-overlap matches and HNSW
vector matches, but never compared them on one shared relevance scale — a
low-scoring keyword match and a genuinely relevant vector chunk were never
judged against each other, just concatenated under separate headings.

### Added
- `src/reranker.ts` — `rerankCandidates()`: real LLM-as-reranker. One real,
  separate Ollama `/api/chat` call per candidate (`format: "json"`,
  `temperature: 0`), asking for an integer 0-10 relevance score against the
  actual query on a fixed rubric; candidates re-sorted by that real score.
  Sequential calls (not parallel) for honest per-candidate latency. A
  failed call is marked `rerankFailed: true` and sinks to the bottom — never
  assigned a fabricated score. `buildRerankCandidates()` builds the
  candidate list from the same graph low-level matches + HNSW vector
  matches `hybrid_retrieval.ts` already produces.
- `src/hybrid_retrieval.ts` — `reciprocalRankFusion()`: real RRF
  (`Σ 1/(k+rank)` per ranked list an item appears in, `k=60` default),
  the rank-based hybrid-search fusion method used by
  OpenSearch/Elasticsearch/Azure AI Search, as an alternative to the
  existing weighted-concatenation merge.
- `POST /api/query` (`server.ts`) gains an optional `"mode"` field:
  `"keyword"` (graph-only, HNSW genuinely not called), `"vector"`
  (HNSW-only, graph genuinely not queried), `"hybrid"` (existing default
  behavior, unchanged), `"hybrid_rerank"` (runs the LLM reranker above and
  prepends the reranked order to the actual synthesis context — not
  decorative), `"hybrid_rrf"` (fuses via RRF). Response gains `rerank` and
  `rrf` fields (null unless that mode ran) with full per-candidate detail.

### Verified, 2026-08-26, against real local Ollama
- `mode: "hybrid_rerank"`, `rerankModel: "llama3.2:3b"`, 7 real candidates
  (3 graph + 4 vector) on the query *"How does the HNSW vector index
  perform approximate nearest neighbor search and how does it relate to
  quantization?"* — real reordering (`rankChanged: true`): the graph's
  keyword-overlap rank-1 match (`OmniBrowserAgent.navigate()`, unrelated to
  the query) scored `0/10` and fell to rank 3; the vector search's
  genuinely relevant top match (HNSW description) scored `9/10` and stayed
  rank 1. Total wall-clock **18,680ms** for 7 candidates (per-call latency
  430ms–15,855ms, one real outlier included honestly, not filtered out).
- `mode: "hybrid_rrf"` on the same query — the same irrelevant graph rank-1
  match tied the relevant vector rank-1 match at `rrfScore=0.01639`
  (`1/(60+1)` both), a real, measured illustration of RRF's blind spot
  (rank-only, no notion of *why* something ranked first) vs. what the LLM
  reranker caught in the same run.
- `mode: "keyword"` → response's `rag.vectorMatches` is `[]` (HNSW never
  called); `mode: "vector"` → `rag.lowLevelMatches` is `[]` (graph never
  queried) — confirmed mode isolation is real, not label-only.

## Unreleased — real hybrid retrieval + real document ingestion

**Gap and source.** With graph ingestion done, two gaps remained from
studying real comparable projects (LightRAG, LlamaIndex, txtai, DSPy):
(1) this project's real graph query and real HNSW vector search ran as two
disconnected paths — `/api/query` never used HNSW at all — even though real
LightRAG's own documented "hybrid mode" is exactly merging graph + vector
retrieval (checked against https://github.com/HKUDS/LightRAG); (2) the only
way to get a user's own document into the system was one string → one
vector (`/api/hnsw/insert`) or graph-only (`/api/graph/ingest`), with no
real chunking and no single path that fed both retrieval mechanisms.

### Added
- `src/hybrid_retrieval.ts` — `buildHybridContext()`: merges a
  `LightRAGEngine.query()` graph result with real HNSW `search()` results
  into one labeled context block (graph themes/entities/relations + vector
  chunks with their cosine similarity scores), in the same spirit as
  LightRAG's hybrid mode.
- `POST /api/query` (`server.ts`) now runs both the graph query and a real
  HNSW vector search over the same prompt and merges them via
  `buildHybridContext()` before LLM synthesis; response gains
  `retrievalMode: "hybrid (graph + HNSW vector, LightRAG hybrid-mode
  style)"`. HNSW search failing degrades to graph-only context instead of
  failing the request.
- `src/document_ingest.ts` — `chunkBySentence()` (lossless sentence-boundary
  splitting via `String.split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/)`) and
  `ingestDocument()`, which chunks a user-supplied document, inserts every
  chunk into the real HNSW index, and (optionally) runs the whole document
  through the existing `extractGraphFromText` pipeline into the LightRAG
  graph — one call populates both retrieval paths the new hybrid query
  merges. Reports vector-index and graph-extraction success/failure
  independently rather than one combined flag.
- `POST /api/document/ingest` (`server.ts`) — `{text, documentId?,
  extractGraph?, model?, maxChunkChars?}` → runs the above.

### Fixed
- `chunkBySentence()`'s first implementation used a consuming regex
  (`[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$`) that silently *deleted* text
  immediately after a period not followed by whitespace — measured on a
  real test document: `"Bun.serve() starts an HTTP server..."` chunked to
  `"serve() starts an HTTP server..."` (dropping `"Bun."`), `"qwen2.5:7b"`
  dropped its `"qwen2."` prefix. Fixed by switching to a lossless
  `String.split` on a lookbehind boundary, which cannot discard characters
  the way the consuming alternation could. Re-verified against the same
  document afterward: both tokens preserved intact in the extracted graph
  node names.
- `src/graph_ingest.ts`'s per-chunk Ollama extraction call timeout raised
  from 60000ms to 120000ms after measuring real local inference time on
  this dev machine: a single JSON-constrained extraction call to
  `qwen2.5:7b` took ~68s wall clock (CPU-bound, ~10 tok/s) — the old 60s
  timeout was cutting off real in-progress correct responses, not catching
  hung requests. This also benefits the pre-existing `/api/graph/ingest`
  endpoint.

### Verified
- Real local Ollama (`qwen2.5:7b`), real HNSW index, real graph — see
  README "Real hybrid retrieval" and "Real user document ingestion"
  sections for the specific measured requests/responses (similarity
  rankings, chunk counts, entity/edge counts, the bad-model 404 path, the
  empty-text 400 path).

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
