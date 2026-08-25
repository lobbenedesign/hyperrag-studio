#!/usr/bin/env bun
/**
 * 🚀 HYPERRAG STUDIO SERVER (v1.0.0)
 * Dual-Level LightRAG, EAGLE Speculative Decoding & DSPy Auto-Compiler
 */

import { LightRAGEngine } from "./src/lightrag_engine";
import { extractGraphFromText } from "./src/graph_ingest";
import { buildHybridContext } from "./src/hybrid_retrieval";
import { ingestDocument } from "./src/document_ingest";
import { TurboQuantEngine } from "./src/turboquant";
import { SpeculativeDecodingEngine } from "./src/speculative_decoder";
import { DSPyCompilerEngine } from "./src/dspy_compiler";
import { HNSWVectorIndex } from "./src/hnsw_vector_index";
import { RealVectorEmbedder } from "./src/real_vector_embedder";
import { join } from "path";
import { existsSync } from "fs";

const PORT = Number(process.env.PORT) || 3003;
const OLLAMA_HOST = process.env.OLLAMA_HOST || "http://localhost:11434";

const lightRAG = new LightRAGEngine();
const speculativeEngine = new SpeculativeDecodingEngine(4);
const dspyCompiler = new DSPyCompilerEngine();
const hnswIndex = new HNSWVectorIndex();
const embedder = new RealVectorEmbedder();
const turboQuant = new TurboQuantEngine(64);

// Seed initial knowledge base chunks into real HNSW index
(async () => {
  await hnswIndex.insert("chunk-1", "TurboQuant provides 4-bit vector quantization with QJL 1-bit residual error correction for inner products.");
  await hnswIndex.insert("chunk-2", "Speculative decoding uses a smaller draft model to predict future tokens validated in parallel by the target LLM.");
  await hnswIndex.insert("chunk-3", "LightRAG combines low-level concrete code entities with high-level architectural knowledge graphs.");
  await hnswIndex.insert("chunk-4", "DSPy compiler optimizes prompt pipelines and signatures via bootstrap few-shot optimization.");
  await hnswIndex.insert("chunk-5", "HNSW Hierarchical Navigable Small World graphs enable logarithmic time approximate nearest neighbor search.");
  console.log("🌲 Real HNSW Vector Index seeded with 5 knowledge chunks.");
})();

let activeModel = "qwen2.5:7b";
let totalQueriesServed = 0;
// These are updated from REAL measured results of the last /api/speculative/bench
// and /api/dspy/compile calls, not hardcoded marketing numbers. Null until a
// real benchmark/compile has actually run in this process.
let lastMeasuredSpeedupFactor: string | null = null;
let lastMeasuredDspyGain: string | null = null;

console.log(`\n======================================================`);
console.log(`🚀 HYPERRAG STUDIO running on http://localhost:${PORT}`);
console.log(`🕸️ LightRAG Dual-Level Graph Engine: Active`);
console.log(`🦅 Draft/Target Speculative Benchmark (real Ollama measurement): Ready`);
console.log(`🧬 DSPy-Inspired Prompt Compiler (live Ollama-scored when reachable): Online`);
console.log(`⚡ TurboQuant-style 4-Bit Vector Engine: Online (QJL-style residual)`);
console.log(`======================================================\n`);

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    const headers = {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization"
    };

    if (req.method === "OPTIONS") return new Response(null, { headers });

    // Serve Static UI Assets
    if (url.pathname === "/" || url.pathname === "/index.html") {
      const p = join(__dirname, "public", "index.html");
      return new Response(Bun.file(p), { headers: { "Content-Type": "text/html" } });
    }
    if (url.pathname === "/app.js") {
      const p = join(__dirname, "public", "app.js");
      return new Response(Bun.file(p), { headers: { "Content-Type": "application/javascript" } });
    }
    if (url.pathname === "/style.css") {
      const p = join(__dirname, "public", "style.css");
      return new Response(Bun.file(p), { headers: { "Content-Type": "text/css" } });
    }
    if (url.pathname.startsWith("/public/")) {
      const p = join(__dirname, url.pathname);
      if (existsSync(p)) return new Response(Bun.file(p));
    }

    // 1. Status API
    if (url.pathname === "/api/status" && req.method === "GET") {
      const g = lightRAG.getGraph();
      return new Response(JSON.stringify({
        status: "online",
        version: "1.0.0-hyperrag",
        activeModel,
        graphNodesCount: g.nodes.length,
        graphEdgesCount: g.edges.length,
        totalQueriesServed,
        accelerationMultiplier: lastMeasuredSpeedupFactor || "not yet measured (run a speculative benchmark)",
        dspyOptimizationAvgGain: lastMeasuredDspyGain || "not yet measured (run a DSPy compile)"
      }), { headers });
    }

    // 2. LightRAG Graph API
    if (url.pathname === "/api/graph" && req.method === "GET") {
      return new Response(JSON.stringify(lightRAG.getGraph()), { headers });
    }

    // 3. Dual-Level LightRAG Query
    if (url.pathname === "/api/query" && req.method === "POST") {
      try {
        const body: any = await req.json();
        const prompt = body.prompt || "";
        totalQueriesServed += 1;

        const graphResult = lightRAG.query(prompt);

        // Real hybrid retrieval: merge the graph query above with a real
        // HNSW cosine-similarity vector search over the same prompt, same
        // idea as LightRAG's own documented "hybrid mode" (graph structure
        // + raw text chunks). Vector search failing (e.g. embedder falling
        // back, or an empty index) degrades to graph-only context rather
        // than failing the whole request — it's an honest empty section,
        // not a fabricated match.
        let vectorMatches: Awaited<ReturnType<typeof hnswIndex.search>> = [];
        try {
          vectorMatches = await hnswIndex.search(prompt, 4);
        } catch (err: any) {
          console.warn(`hybrid retrieval: HNSW vector search failed: ${err.message}`);
        }
        const ragResult = buildHybridContext(graphResult, vectorMatches);

        // Perform LLM Synthesis against a live local Ollama model.
        // IMPORTANT: if Ollama is unreachable or errors, we report that
        // honestly instead of fabricating a fake "synthesis" string that
        // pretends the LLM answered.
        let synthesis = "";
        let synthesisError: string | null = null;
        try {
          const ollamaRes = await fetch(`${OLLAMA_HOST}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model: activeModel,
              messages: [
                { role: "system", content: `You are HyperRAG, an advanced AI synthesis engine using Dual-Level Knowledge Graphs.\n${ragResult.synthesizedContext}` },
                { role: "user", content: prompt }
              ],
              stream: false
            }),
            signal: AbortSignal.timeout(45000)
          });
          if (ollamaRes.ok) {
            const data: any = await ollamaRes.json();
            synthesis = data.message?.content || "";
            if (!synthesis) synthesisError = "Ollama responded but returned no content.";
          } else {
            synthesisError = `Ollama returned HTTP ${ollamaRes.status}.`;
          }
        } catch (err: any) {
          synthesisError = `Ollama unreachable at ${OLLAMA_HOST}: ${err.message}`;
        }

        return new Response(JSON.stringify({
          success: true,
          prompt,
          retrievalMode: "hybrid (graph + HNSW vector, LightRAG hybrid-mode style)",
          rag: ragResult,
          synthesis,
          synthesisError,
          llmUsed: synthesisError ? false : true
        }), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
      }
    }

    // 4. EAGLE Speculative Decoding Benchmark
    if (url.pathname === "/api/speculative/bench" && req.method === "POST") {
      try {
        const body: any = await req.json();
        const prompt = body.prompt || "Generate high performance code";
        const bench = await speculativeEngine.benchmark(prompt, body.model || activeModel);
        lastMeasuredSpeedupFactor = bench.measuredSpeedupFactor;
        return new Response(JSON.stringify(bench), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
      }
    }

    // 5. DSPy Declarative Prompt Compilation
    if (url.pathname === "/api/dspy/compile" && req.method === "POST") {
      try {
        const body: any = await req.json();
        const sig = body.signature || {
          name: "CodeSynthesisSignature",
          description: "Generates high-performance, strictly-typed code",
          inputs: [{ field: "spec", type: "string", desc: "Feature requirement" }],
          outputs: [{ field: "code", type: "string", desc: "Production code" }]
        };
        const result = await dspyCompiler.compileSignature(sig, body.prompt || "");
        lastMeasuredDspyGain = result.accuracyGain;
        return new Response(JSON.stringify(result), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
      }
    }

    // 6. Google TurboQuant 4-Bit Vector Compression & Search API
    if (url.pathname === "/api/turboquant/compress" && req.method === "POST") {
      try {
        let body: any = {};
        try { body = await req.json(); } catch {}
        const vector: number[] = body.vector || Array.from({ length: 64 }, () => Math.random() * 2 - 1);
        const compressed = turboQuant.compress(body.id || `vec-${Date.now()}`, vector);
        return new Response(JSON.stringify(compressed), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
      }
    }

    if (url.pathname === "/api/turboquant/search" && req.method === "POST") {
      try {
        let body: any = {};
        try { body = await req.json(); } catch {}
        const queryVec: number[] = body.queryVector || Array.from({ length: 64 }, () => Math.random() * 2 - 1);
        const corpus = body.corpus || [
          { id: "vec-code-1", name: "Rust Memory Management", text: "Zero-cost abstractions and borrow checker rules." },
          { id: "vec-code-2", name: "LightRAG Knowledge Graph", text: "Dual-level topological entity relations." },
          { id: "vec-code-3", name: "EAGLE Speculative Trees", text: "3.5x faster parallel draft token decoding." }
        ];

        const scored = corpus.map((item: any, i: number) => {
          const mockVec = Array.from({ length: 64 }, (_, idx) => Math.sin(i * 10 + idx * 0.2));
          const comp = turboQuant.compress(item.id, mockVec);
          const score = turboQuant.estimateSimilarity(queryVec, comp);
          return { ...item, similarityScore: score, compressionRatio: comp.compressionRatio };
        });

        scored.sort((a: any, b: any) => b.similarityScore - a.similarityScore);
        return new Response(JSON.stringify({ results: scored }), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
      }
    }

    // 7. Real HNSW Vector Search
    if (url.pathname === "/api/hnsw/search" && req.method === "POST") {
      try {
        let body: any = {};
        try { body = await req.json(); } catch {}
        const query = body.query || "fast inference server";
        const k = Number(body.k) || 4;
        const results = await hnswIndex.search(query, k);
        return new Response(JSON.stringify({ query, results, stats: hnswIndex.getStats() }), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
      }
    }

    // 8. Real HNSW Vector Document Insertion
    if (url.pathname === "/api/hnsw/insert" && req.method === "POST") {
      try {
        let body: any = {};
        try { body = await req.json(); } catch {}
        const id = body.id || `doc-${Date.now()}`;
        const text = body.text || "";
        const node = await hnswIndex.insert(id, text, body.metadata);
        return new Response(JSON.stringify({ success: true, node: { id: node.id, level: node.level, text: node.text } }), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
      }
    }

    // 9. Real HNSW Index Topology Stats
    if (url.pathname === "/api/hnsw/stats" && req.method === "GET") {
      return new Response(JSON.stringify(hnswIndex.getStats()), { headers });
    }

    // 10. Real LLM-driven graph ingestion — extracts entities/relations from
    // real text via a live Ollama call (see src/graph_ingest.ts) and merges
    // them into the LightRAG graph, replacing hand-seeded data with content
    // actually derived from what you feed it. Fails honestly (no fake graph)
    // if Ollama is unreachable or its output doesn't parse.
    if (url.pathname === "/api/graph/ingest" && req.method === "POST") {
      try {
        const body: any = await req.json();
        const text: string = body.text || "";
        const sourceLabel: string | undefined = body.sourceLabel;
        const model: string = body.model || activeModel;

        if (!text.trim()) {
          return new Response(JSON.stringify({ error: "body.text is required and must be non-empty" }), {
            status: 400,
            headers
          });
        }

        const extraction = await extractGraphFromText(text, { ollamaHost: OLLAMA_HOST, model });

        const allEntities = extraction.chunks.flatMap((c) => c.entities);
        const allRelations = extraction.chunks.flatMap((c) => c.relations);
        const merge = lightRAG.ingestExtracted(allEntities, allRelations, sourceLabel);

        return new Response(
          JSON.stringify({
            success: true,
            model: extraction.model,
            chunksProcessed: extraction.chunks.length,
            totalEntitiesExtracted: extraction.totalEntitiesExtracted,
            totalRelationsExtracted: extraction.totalRelationsExtracted,
            merge,
            graphStats: { nodes: lightRAG.getGraph().nodes.length, edges: lightRAG.getGraph().edges.length }
          }),
          { headers }
        );
      } catch (e: any) {
        return new Response(JSON.stringify({ success: false, error: e.message }), { status: 502, headers });
      }
    }

    // 11. Real document ingestion: sentence-boundary chunking → real HNSW
    // vector indexing of every chunk + (optionally) real LLM graph
    // extraction over the whole document, reusing the existing
    // extractGraphFromText pipeline. This is what actually lets a user feed
    // their own document into both retrieval paths that /api/query's hybrid
    // mode merges, instead of only the hand-seeded HNSW chunks and hand-
    // seeded graph nodes from server startup.
    if (url.pathname === "/api/document/ingest" && req.method === "POST") {
      try {
        const body: any = await req.json();
        const text: string = body.text || "";
        if (!text.trim()) {
          return new Response(JSON.stringify({ error: "body.text is required and must be non-empty" }), {
            status: 400,
            headers
          });
        }
        const result = await ingestDocument(text, {
          documentId: body.documentId,
          hnswIndex,
          lightRAG,
          extractGraph: body.extractGraph !== false, // default true
          ollamaHost: OLLAMA_HOST,
          model: body.model || activeModel,
          maxChunkChars: Number(body.maxChunkChars) || 800
        });
        return new Response(JSON.stringify({ success: true, ...result }), { headers });
      } catch (e: any) {
        return new Response(JSON.stringify({ success: false, error: e.message }), { status: 500, headers });
      }
    }

    return new Response("Not Found", { status: 404, headers });
  }
});
