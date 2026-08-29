# 🚀 HyperRAG Studio

[![Bun](https://img.shields.io/badge/Bun-v1.4+-black.svg?logo=bun)](https://bun.sh/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg?logo=typescript)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-experimental%20%2F%20heuristic-orange.svg)](#-honest-status)

[English 🇬🇧](#english) • [Italiano 🇮🇹](#italiano)

> A small local Bun/TypeScript playground exploring GraphRAG-style dual-level retrieval, a real HNSW vector index, and a couple of research-technique-*inspired* toy modules. This README was rewritten to accurately describe what the code actually does — see [Honest Status](#-honest-status) below.

![HyperRAG Studio Dashboard](./public/screenshot.jpg)

---

<a name="english"></a>
## 🇬🇧 English Documentation

### 🔍 Honest Status

This project's earlier README claimed a full **LightRAG + EAGLE Speculative Decoding + DSPy** stack with measured performance numbers (3.5x speedup, +28.5% accuracy). After an audit, those claims did not match the code. Here is what is real today:

| Module | Claim (old) | What's actually there |
|---|---|---|
| `src/lightrag_engine.ts` | Dual-level LightRAG knowledge graph | Real: a small in-memory graph with a genuine keyword-overlap scoring query (low-level vs high-level node matching). The graph now also has a **real ingestion path** — see below — so it's no longer only seed data, though the seed nodes from the original release remain until you ingest something. |
| `src/hnsw_vector_index.ts` + `src/real_vector_embedder.ts` | (implied vector search) | Real: an actual HNSW (Malkov & Yashunin) graph implementation with real cosine similarity. Embeddings come from a live call to Ollama's embedding API when available, or fall back to a deterministic FNV-1a trigram/word hash embedding (NOT a trained semantic embedding model) when Ollama/the embedding model isn't reachable. |
| `src/speculative_decoder.ts` | "EAGLE/Medusa speculative decoding, 3.0x–3.8x faster, zero quality loss" | **Not real EAGLE/Medusa.** True EAGLE speculative decoding needs a draft head trained on the target model's hidden states plus tree-attention logit verification inside the inference engine — Ollama's HTTP API exposes none of that. What runs instead: a **real** benchmark that sends the same prompt to a small "draft" model (`granite3-dense:2b`) and a larger "target" model (`qwen2.5:7b`) via local Ollama (or, with `LLM_BACKEND=localai`, [LocalAI](https://github.com/mudler/LocalAI)), measures **real** tokens/sec (from Ollama's own counters, or from wall-clock latency / real `completion_tokens` on LocalAI — labelled differently, see `notes` in the response, since they measure slightly different things), and computes a **real** (but approximate) word-overlap rate between the two live outputs as a stand-in for an "acceptance rate". No hidden state, no logit verification, no fabricated numbers — but also not true speculative decoding. Throws an honest error if the configured backend is unreachable instead of returning fake numbers. |
| `src/dspy_compiler.ts` | "DSPy declarative prompt compiler, +28.5% accuracy" | **Not the real DSPy library.** No dataset, no real teleprompter training loop. What runs instead: it structurally compiles a typed signature into an explicit instruction prompt, and — when Ollama is reachable — actually calls the model once to capture a real one-shot demonstration and again to A/B-score the naive vs. compiled prompt using a real, computed structural-adherence heuristic (JSON validity, presence of requested output fields, length sanity). If Ollama is unreachable it clearly reports `liveEvaluated: false` and labels the numbers as an offline heuristic, not a measured accuracy gain. |
| `src/turboquant.ts` | "Google TurboQuant 4-bit quantization" | Real, working 4-bit scalar quantization + 1-bit sign-residual correction and cosine-similarity estimation on quantized vectors. Inspired by the published TurboQuant idea; this is an independent from-scratch implementation, not Google's code. |

**Bugs found and fixed during the audit:**
- `server.ts` called `dspyCompiler.compileSignature(...)`, a method that did not exist on `DSPyCompilerEngine` (only `compile()` did) — every `/api/dspy/compile` request would have crashed. Fixed by renaming/reworking the method to match, and awaiting it (it's now genuinely async because it calls Ollama).
- `public/app.js` read `data.optimizedPrompt` from the compile response, but the server actually returned `compiledPrompt` — the compiled prompt box would always show `undefined`. Fixed.
- `/api/query`'s Ollama fallback silently fabricated a fake "synthesis" string when Ollama was unreachable, making failures look like successful LLM answers. Fixed: the endpoint now returns `llmUsed: false` and a real `synthesisError` message instead of pretending an LLM responded.
- `/api/status` returned hardcoded `accelerationMultiplier: "3.42x"` and `dspyOptimizationAvgGain: "+28.5%"` regardless of whether any benchmark had ever run. Fixed: these now reflect the last **real** measured result from `/api/speculative/bench` and `/api/dspy/compile`, and say "not yet measured" until you actually run one.

### 🧩 Real LLM-driven graph ingestion (`src/graph_ingest.ts`)

The "not a real ingestion pipeline" gap above was real, and it's the same gap
between a toy graph and what actual LightRAG / Microsoft GraphRAG do: both
build their knowledge graphs by running an LLM entity/relationship
extraction pass over document chunks — GraphRAG's own repo calls this step
"element extraction" (verified against
[github.com/microsoft/graphrag](https://github.com/microsoft/graphrag), not
assumed from memory).

`POST /api/graph/ingest` with `{ "text": "...", "sourceLabel"?: "...",
"model"?: "qwen2.5:7b" }` now does the same *technique* against local
Ollama:

1. Splits the input into paragraph-bounded chunks (~1400 chars).
2. For each chunk, calls Ollama's `/api/chat` with `format: "json"` and a
   schema-constrained prompt asking for `{entities, relations}`.
3. Validates the response: unknown `type`/`level` values are coerced to safe
   defaults, and — importantly — **any relation whose `source` or `target`
   doesn't match a `name` the model also extracted as an entity is dropped**,
   so a hallucinated edge can't reference a node that doesn't exist.
4. Merges into `LightRAGEngine`'s live graph via the new
   `ingestExtracted()` method, deduping by case-insensitive entity name so
   re-ingesting the same or an overlapping document doesn't pile up
   duplicate nodes — it reports `nodesAdded` vs `nodesMerged` so you can see
   which happened.

If Ollama is unreachable, returns a non-JSON response, or returns JSON that
doesn't parse into the expected shape, the endpoint returns
`{"success": false, "error": "..."}` with HTTP 502 — it does **not** fall
back to inserting a fake or empty graph, matching this project's existing
honesty policy.

**What this is not**: it does not reimplement GraphRAG's actual extraction
prompts, nor its later community-detection/summarization stages (Leiden
clustering, hierarchical community reports) — only the entity/relation
extraction step, applied to `LightRAGEngine`'s existing (simpler) two-level
node model, not GraphRAG's multi-level community hierarchy.

**Verified, 2026-08-25**, against a real local Ollama (`qwen2.5:7b`):
- Ingested a real two-paragraph technical text describing this project's own
  `HNSWVectorIndex`/`TurboQuantEngine` classes → 4 entities, 4 relations
  extracted and merged (`nodesAdded: 4, nodesMerged: 0, edgesAdded: 4`),
  confirmed present in `GET /api/graph` and retrievable via `POST
  /api/query` afterward.
- Re-ingested the identical text → `nodesAdded: 0, nodesMerged: 2,
  edgesAdded: 0` (partial dedupe: the model didn't re-extract identical
  entities/relations verbatim the second time, which is expected LLM
  non-determinism, but the entities it did re-extract by the same name
  correctly merged into the existing nodes rather than duplicating them).
- `model: "not-a-real-model-xyz"` → HTTP 502,
  `{"success":false,"error":"Ollama returned HTTP 404 while extracting entities from chunk 1/1"}`
  — a real Ollama 404, not a fabricated graph.
- `text: ""` → HTTP 400 with a validation error, no Ollama call made.

### 🔀 Real hybrid retrieval (`src/hybrid_retrieval.ts`)

Before this module, `POST /api/query` only ever ran `LightRAGEngine.query()`
(graph keyword-overlap matching). The real `HNSWVectorIndex` +
`RealVectorEmbedder` already existed in this codebase but were only
reachable through the separate `/api/hnsw/search` endpoint — never combined
with the graph query, even though real LightRAG's own documented "hybrid
mode" is exactly this merge (verified against
[github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG): "Hybrid Mode
integrates local and global retrieval methods... simultaneously leverages
both structured knowledge and unstructured text"). `/api/query` now runs
*both* retrieval paths against the same prompt — the graph query and a real
HNSW cosine-similarity vector search — and merges them into one labeled
context block via `buildHybridContext()`, added to the response as
`retrievalMode: "hybrid (graph + HNSW vector, LightRAG hybrid-mode style)"`.
Nothing here claims to be more than what it already was: a graph match is
still keyword overlap over a small in-memory graph, a vector match is still
cosine similarity over a 384-dim embedding (real Ollama embedding or the
deterministic hash fallback, exactly as documented above) — this only
connects the two paths that previously ran in isolation. If the vector
search errors (e.g. an empty index), the request degrades to graph-only
context rather than failing outright.

**Verified, 2026-08-25**, against a real local Ollama (`qwen2.5:7b`) and the
five HNSW chunks seeded at server startup: querying "How does the HNSW
vector index perform approximate nearest neighbor search and how does it
relate to quantization?" returned `vectorMatches` correctly ranked by
similarity — `chunk-5` (the HNSW description) highest at `sim=0.234`, down
to `chunk-3` (the LightRAG description, least related) lowest at
`sim=0.059` — alongside the graph's own (unrelated, keyword-only) matches,
confirming the two paths are genuinely independent scores being merged, not
one path silently masking the other.

### 📄 Real user document ingestion (`src/document_ingest.ts`)

Addresses the gap the earlier version of this README's roadmap called out —
until now the only user-facing ingestion path was `/api/hnsw/insert`, which
embeds one caller-supplied string as exactly *one* vector, and
`/api/graph/ingest`, which extracts a graph but never touches the vector
index. `POST /api/document/ingest` with `{ "text": "...", "documentId"?,
"extractGraph"?: true, "model"?, "maxChunkChars"?: 800 }` does real
sentence-boundary chunking of the whole document — not naive fixed-size
mid-sentence cuts — then feeds every chunk into the real HNSW index as its
own vector node **and** (unless `extractGraph: false`) runs the whole
document through the existing `extractGraphFromText` pipeline into the
LightRAG graph, so one upload populates both retrieval paths that the new
hybrid `/api/query` merges. Same pattern real chunkers use (LlamaIndex's
`SentenceSplitter`, txtai's `Textractor`) — chunk before embedding so one
vector doesn't have to represent an entire document — reimplemented from
scratch here, not copied.

Each stage reports its own success/failure rather than one opaque flag: a
document can legitimately end up vector-indexed but not graph-extracted (the
embedder's hash fallback still works even when the LLM extraction call
doesn't), and the response's `vectorIndex` / `graphExtraction` fields say
exactly which happened.

**A real bug found and fixed during verification**: the first
implementation of the sentence splitter used a consuming regex
(`[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$`) that, for a period *not* followed by
whitespace — `Bun.serve()`, `qwen2.5:7b` — failed to match starting at that
token and silently skipped it, deleting it from the chunked output
entirely (measured: `"Bun.serve() starts an HTTP server..."` chunked down to
`"serve() starts an HTTP server..."`, `"qwen2.5:7b"` down to `"5:7b"`).
Fixed by switching to a lossless `String.split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/)`
approach, which cannot drop characters the way a consuming alternation can —
confirmed by round-tripping the same test document afterward with both
abbreviation-like tokens intact in the extracted graph nodes.

**Verified, 2026-08-25**, against real local Ollama (`qwen2.5:7b`) on this
project's actual dev machine:
- A ~670-char real technical paragraph (about Bun, Ollama, and this
  project's own HNSW index) ingested with default settings → 1 chunk (under
  `maxChunkChars`), 1 HNSW node inserted, graph extraction succeeded with
  `nodesAdded: 8, edgesAdded: 5` — including `Bun.serve()` and `qwen2.5:7b`
  as correctly-preserved entity names, confirming the chunking-bug fix.
- The same text with `maxChunkChars: 150` → 4 sentence-respecting chunks
  (84–142 chars each, no sentence split mid-way), 4 HNSW nodes inserted.
- `text: ""` → HTTP 400, no Ollama call made.
- `model: "not-a-real-model-xyz"` → vector indexing still succeeded
  (`inserted: 1`), while `graphExtraction` honestly reported
  `succeeded: false, error: "Ollama returned HTTP 404 while extracting
  entities from chunk 1/1"` — a real Ollama 404, not a fabricated graph, and
  proof the two ingestion stages fail independently rather than one hiding
  the other's failure.
- **A genuinely measured, not assumed, hardware finding**: a single
  JSON-constrained entity-extraction call to `qwen2.5:7b` on this dev
  machine took ~68s wall clock (`eval_duration` ≈ 66.5s for 671 output
  tokens ≈ 10 tok/s — CPU-bound local inference, not GPU-accelerated). The
  original 60s timeout in `src/graph_ingest.ts` was cutting off real,
  correct, in-progress responses before they finished, not catching
  genuinely hung requests — raised to 120s after measuring this, which also
  benefits the pre-existing `/api/graph/ingest` endpoint.

### 🎯 Real reranking + retrieval mode selector (`src/reranker.ts`, `src/hybrid_retrieval.ts`)

The gap the previous version of this README's own status table implicitly
left open: `hybrid_retrieval.ts` merged the graph's keyword-overlap matches
and the HNSW vector matches into one context block, but never actually
compared candidates from the two sources against each other on one shared
scale — a graph node that scored a couple of keyword-overlap points could
sit above a vector chunk that was genuinely more relevant, purely because
the two lists were never judged by the same yardstick. Real LightRAG
(verified against [github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG))
doesn't solve this either — it documents four/five modes (`naive`, `local`,
`global`, `hybrid`, `mix`) but its "hybrid" mode is the same kind of
independent-list merge this project already had. This project now adds two
real, separately verifiable pieces: an **LLM-as-reranker** pass (the pattern
used by production cross-encoder/pointwise rerankers like Cohere Rerank or
BGE-reranker, adapted here to a local Ollama chat model since this project
has no cross-encoder weights) and **Reciprocal Rank Fusion** (the
rank-based fusion default in OpenSearch/Elasticsearch/Azure AI Search).

`POST /api/query` now takes an optional `"mode"` field, LightRAG-mode-selector
style, so the retrieval strategies can be compared side-by-side on the same
prompt:

| Mode | What actually runs |
|---|---|
| `"keyword"` | `LightRAGEngine.query()` only — no HNSW call at all. |
| `"vector"` | `HNSWVectorIndex.search()` only — no graph query at all. |
| `"hybrid"` (default) | Existing behavior: both, concatenated under separate headings by `buildHybridContext()`. |
| `"hybrid_rerank"` | Both retrieved, then every candidate (graph low-level match text + vector chunk text) gets **one real, separate Ollama `/api/chat` call** (`format: "json"`, `temperature: 0`) asking for an integer 0-10 relevance score against the actual query, on a fixed rubric. Candidates are re-sorted by that real score, and the reranked block — not the original hybrid block — is what's actually prepended to the synthesis LLM's context, so this changes what the model sees, not just what the API response displays. Optional `"rerankModel"` field picks the judge model (defaults to `activeModel`, `qwen2.5:7b`). |
| `"hybrid_rrf"` | Both retrieved, then fused via real Reciprocal Rank Fusion (`rrfScore = Σ 1/(k + rank)` per list an item appears in, `k=60` by default, overridable via `"rrfK"`) — pure arithmetic over the two real ranked lists, no LLM call. |
| `"vector_hyde"` | `HNSWVectorIndex.search()` only, but the query text embedded is a HyDE-generated hypothetical answer passage (`src/hyde.ts`, one real Ollama call) instead of the raw prompt. Degrades to raw-prompt vector search with `hydeError` set if generation fails. See [Real HyDE](#-real-hyde--hypothetical-document-embeddings-srchydets) below. |
| `"hybrid_hyde"` | Same as `"hybrid"`, but the vector-search half uses the HyDE-generated passage instead of the raw prompt; the graph half is unaffected. |

**A real, honest limitation this surfaced, not assumed in advance**: RRF is
rank-only — it has no idea *why* something ranks first in a list, only
*that* it does. In the verification run below, the graph's rank-1 match
(pure keyword-overlap noise, completely unrelated to the query) tied in RRF
score with the vector search's rank-1 match (the actually-relevant chunk),
because both were simply "rank 1 in their own list." The LLM reranker did
not make this mistake — it scored the same irrelevant graph match `0/10`
and the relevant vector chunk `9/10`. This is a genuine, measured tradeoff,
not a talking point: RRF is fast (no LLM call, sub-millisecond) and
correctly source-agnostic, but blind to actual relevance when a low-quality
source's top rank collides with a good source's top rank; LLM reranking is
slow (one real model call per candidate) but catches exactly this case.

**Verified, 2026-08-26**, against real local Ollama and the seeded HNSW
chunks + graph nodes:
- Query *"How does the HNSW vector index perform approximate nearest
  neighbor search and how does it relate to quantization?"*, `mode:
  "hybrid_rerank"`, `rerankModel: "llama3.2:3b"` — **7 real candidates**
  (3 graph, 4 vector) scored, **total wall-clock 18,680ms** (individual
  calls ranged 430ms–15,855ms; one call was a real outlier, most landed
  400-600ms — reported as measured, not smoothed). Real reordering
  happened (`rankChanged: true`):
  - `HNSW Hierarchical Navigable Small World graphs enable logarithmic
    time approximate nearest neighbor search.` — vector rank 1 → LLM
    score **9/10** → stayed rank 1.
  - `TurboQuant provides 4-bit vector quantization with QJL 1-bit
    residual error correction...` — vector rank 2 → LLM score **6/10** →
    stayed rank 2 (correctly recognized as the quantization half of the
    query).
  - `OmniBrowserAgent.navigate(): Performs autonomous HTTP fetching, DOM
    cleanup, and title extraction.` — **graph rank 1** (highest
    keyword-overlap score in its own list) → LLM score **0/10** → fell to
    rank 3. This is the exact failure mode reranking exists to fix: a
    keyword match that scores well against generic terms in the prompt
    but has nothing to do with what's actually being asked.
  - All other candidates (2 more graph nodes, 2 more vector chunks) →
    LLM score **0/10**, correctly sunk to the bottom.
- Same query, `mode: "hybrid_rrf"`, default `k=60` — the graph's irrelevant
  rank-1 match and the vector search's relevant rank-1 match tied at
  `rrfScore=0.01639` (both `1/(60+1)`), confirming the RRF-vs-LLM-reranker
  tradeoff described above with a real number, not a hypothetical.
- `mode: "keyword"` → `vectorMatches: []` (HNSW genuinely not called).
  `mode: "vector"` → `lowLevelMatches: []` (graph genuinely not queried).
  Confirms mode isolation is real, not label-only.

**Honest cost**: LLM reranking is the slowest thing in this codebase per
call — ~450ms-16s per candidate on this dev machine with `llama3.2:3b`,
run strictly sequentially against one local Ollama instance (not
parallelized: one local model can't usefully serve concurrent generation
requests without degrading each call's own latency, so sequential gives an
honest number instead of an artificially compressed one). For `N`
candidates this is realistically `N × 0.5–2s` depending on model and
hardware — noticeable for interactive use, and it scales linearly with
however many candidates the hybrid retrieval step returns. `hybrid_rrf` has
no such cost (pure arithmetic, sub-millisecond) but inherits the accuracy
limitation described above. Neither mode is "better" unconditionally —
which is why both are exposed rather than one silently replacing
`"hybrid"`.

### 🔮 Real HyDE — Hypothetical Document Embeddings (`src/hyde.ts`)

**Source, verified not assumed**: Gao, Ma, Lin & Callan, *"Precise
Zero-Shot Dense Retrieval without Relevance Labels"* (arXiv:2212.10496, ACL
2023) — [arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496). Also a
real, shipped feature in Haystack
([docs.haystack.deepset.ai](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde))
and LangChain's `HypotheticalDocumentEmbedder`, so this is an established
technique, not a research toy.

**The idea**: every mode above still embeds the user's raw prompt string
directly. For a short query against longer descriptive passages, a query
and the passage that actually answers it often sit further apart in
embedding space than two passages would, because embedding models are
generally better at document-to-document similarity than short-query-to-
document similarity. HyDE's fix: ask an LLM to write a short *hypothetical
answer* to the query — framed as a confident documentation/encyclopedia
passage, possibly containing invented specifics, which is fine because it
is never shown to the user — then embed **that** generated passage instead
of the raw query, and search with it.

`POST /api/query` gains two new `"mode"` values: `"vector_hyde"` (HNSW-only,
query embedded via a HyDE-generated passage) and `"hybrid_hyde"` (graph +
HNSW, same HyDE substitution on the vector side only). Optional
`"hydeModel"` field picks the generator model (defaults to `activeModel`).
If HyDE generation fails or times out (30s), the endpoint degrades honestly
to a plain vector search on the raw prompt and reports `hydeError` — it
never fabricates a fake hypothetical document.

**Verified, 2026-08-26**, against real local Ollama
(`llama3.2:3b` as the HyDE generator, seeded 5-chunk HNSW index), 3 real
test queries, plain `"vector"` vs `"vector_hyde"`:

| Query | Plain vector top-1 (sim) | HyDE top-1 (sim) | HyDE latency | Result |
|---|---|---|---|---|
| "How do I make nearest neighbor search fast?" | chunk-5, HNSW passage (0.688) | chunk-5, same passage (0.698) | 22,461ms | Same top-1, marginally higher similarity. No ranking change. |
| "What technique speeds up LLM token generation using a smaller model?" | chunk-2, speculative decoding (0.671) | **HyDE generation itself timed out (30s)**, degraded to raw-query search — but the fallback run's *embedding* call then also missed Ollama's 2.5s embedder timeout (contended by the just-finished HyDE call swapping models in Ollama) and silently fell back further to this project's existing hash-based embedding (`real_vector_embedder.ts`'s documented fallback), landing on the wrong top-1 (chunk-4) with `similarity: 0` reported. | timeout | **Negative result**: on contended/slow local hardware, the extra HyDE hop can cascade into an unrelated pre-existing fallback path and make retrieval *worse* than doing nothing. |
| "How can I shrink vector storage size?" | chunk-1, TurboQuant quantization (0.530) | chunk-1, same passage (0.083 — much lower magnitude, correct chunk) | 29,748ms | Same top-1; 3rd/4th place swapped (chunk-2/chunk-3) vs plain. |

**Honest conclusion**: on this project's tiny, topically-distinct 5-chunk
toy corpus, HyDE did not improve ranking in either successful run — plain
vector search already picked the correct top-1 chunk both times, and HyDE
agreed rather than correcting it. This matches the paper's own framing:
HyDE's advantage grows with corpus size and query/passage semantic distance,
neither of which this toy corpus has much of. What HyDE reliably cost:
**one full extra LLM generation** before every embedding call —
22.4s–29.7s per query on this CPU-bound local hardware with `llama3.2:3b`
(no GPU) — and, in the one query where the local Ollama instance was slow
enough to hit both timeouts back to back, it produced a real regression via
this project's pre-existing hash-embedding fallback rather than gracefully
degrading to the raw-prompt plain-vector result a user would expect. That
fallback interaction is a genuine, measured limitation of running HyDE
against a small local model on constrained hardware, not a hypothetical
edge case — it happened on 1 of 3 real test runs.

### 🌟 Core Modules

* **`src/hyde.ts`**: Real HyDE (Hypothetical Document Embeddings, Gao et al. 2023) — one real Ollama call generates a hypothetical answer passage, which is what actually gets embedded and searched, for `/api/query`'s `"vector_hyde"`/`"hybrid_hyde"` modes.
* **`src/lightrag_engine.ts`**: Dual-level graph + keyword-overlap query engine, now with real merge/dedupe ingestion (`ingestExtracted()`).
* **`src/graph_ingest.ts`**: Real LLM entity/relation extraction from text via local Ollama (`format: "json"`) or, with `LLM_BACKEND=localai`, a local [LocalAI](https://github.com/mudler/LocalAI) instance's OpenAI-compatible `/v1/chat/completions` (`LOCALAI_HOST`, default `http://localhost:8080`), feeding `ingestExtracted()`. Honest failure (no fake graph) if the configured backend is unreachable or returns unparseable output.
* **`src/hybrid_retrieval.ts`**: Merges the graph query and a real HNSW vector search into one labeled context block, LightRAG "hybrid mode" style, plus a real Reciprocal Rank Fusion implementation (`reciprocalRankFusion()`) as an alternative fusion strategy — used by `/api/query`.
* **`src/reranker.ts`**: Real LLM-as-reranker — one real Ollama call per retrieval candidate scoring 0-10 relevance, re-sorted onto one shared scale across graph + vector sources — used by `/api/query`'s `"hybrid_rerank"` mode.
* **`src/document_ingest.ts`**: Real sentence-boundary chunking of user-supplied documents, feeding both the HNSW vector index and graph extraction — used by `/api/document/ingest`.
* **`src/hnsw_vector_index.ts`** / **`src/real_vector_embedder.ts`**: Real HNSW ANN index over real (or hash-fallback) embeddings.
* **`src/speculative_decoder.ts`**: Real draft-vs-target throughput benchmark against local Ollama (not true EAGLE).
* **`src/dspy_compiler.ts`**: Structural prompt compiler with live Ollama A/B scoring when available (not real DSPy).
* **`src/turboquant.ts`**: Real 4-bit vector quantization + residual correction.
* **`public/`**: Cyberpunk dashboard with an HTML5 Canvas graph visualizer and live panels for each module above.

---

### 🛠️ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/lobbenedesign/hyperrag-studio.git
cd hyperrag-studio

# 2. (Optional but recommended) Run Ollama locally for live LLM synthesis,
#    embeddings, and the speculative/DSPy live-scoring paths.
#    Models used by default: qwen2.5:7b (target), granite3-dense:2b (draft),
#    nomic-embed-text (embeddings, optional).

# 3. Run with Bun (Instant startup)
bun server.ts
```

Open your browser at **`http://localhost:3003`**.

---

<a name="italiano"></a>
## 🇮🇹 Documentazione in Italiano

### 🔍 Stato Onesto

La versione precedente di questo README dichiarava uno stack completo **LightRAG + EAGLE Speculative Decoding + DSPy** con numeri di performance misurati (3.5x, +28.5%). Dopo un audit del codice, questi claim non corrispondevano all'implementazione reale. Vedi la tabella in inglese sopra ("Honest Status") per il dettaglio modulo per modulo: in sintesi, il grafo LightRAG è reale ma con dati seed (non un'indicizzazione automatica del progetto), l'indice HNSW e la quantizzazione TurboQuant sono implementazioni reali e funzionanti, mentre "EAGLE Speculative Decoding" e "DSPy" NON sono le tecniche di ricerca reali con quel nome — sono state sostituite con benchmark e compilatori onesti che eseguono chiamate reali a Ollama locale e riportano errori reali invece di dati inventati quando Ollama non è raggiungibile.

### 🛠️ Avvio Rapido

```bash
git clone https://github.com/lobbenedesign/hyperrag-studio.git
cd hyperrag-studio
bun server.ts
```

Apri il browser su **`http://localhost:3003`**.

---

## 📄 License / Licenza
Released under the [MIT License](LICENSE).
