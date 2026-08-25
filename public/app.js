/**
 * 🕸️ HYPERRAG STUDIO CLIENT SCRIPT
 * Renders Dual-Level Interactive Graph on HTML5 Canvas,
 * executes Speculative Decoding benchmarks, and runs DSPy compilations.
 */

let graphData = { nodes: [], edges: [] };
let animationFrameId = null;

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  fetchStatus();
  fetchGraph();
  setupSpeculativeBenchmark();
  setupDSPyCompiler();
  setupRAGQuery();
  setupTurboQuant();
});

function setupTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = `tab-${tab.getAttribute("data-tab")}`;
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
        if (targetId === "tab-graph") {
          setTimeout(initGraphCanvas, 100);
        }
      }
    });
  });
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    document.getElementById("chip-nodes-count").textContent = `🕸️ ${data.graphNodesCount} Entities & Themes`;
    document.getElementById("chip-speed-factor").textContent = `🦅 ${data.accelerationMultiplier}`;
    document.getElementById("chip-dspy-gain").textContent = `🧬 ${data.dspyOptimizationAvgGain}`;
  } catch {}
}

// Fetch & Render LightRAG Graph
async function fetchGraph() {
  try {
    const res = await fetch("/api/graph");
    graphData = await res.json();
    renderEntitiesList();
    initGraphCanvas();
  } catch {}
}

function renderEntitiesList() {
  const container = document.getElementById("graph-entities-list");
  if (!container || !graphData.nodes) return;
  container.innerHTML = "";

  graphData.nodes.forEach(n => {
    const card = document.createElement("div");
    card.className = "entity-card";
    const badgeColor = n.level === "high_level" ? "pill-high" : "pill-low";
    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="entity-name">${n.name}</span>
        <span class="pill ${badgeColor}">${n.type}</span>
      </div>
      <div class="entity-desc">${n.description}</div>
      ${n.fileLocation ? `<div style="font-size: 10px; color: #38bdf8; font-family: var(--font-mono);">${n.fileLocation}</div>` : ""}
    `;
    container.appendChild(card);
  });
}

// 2D Force Graph on Canvas
function initGraphCanvas() {
  const canvas = document.getElementById("graph-canvas");
  if (!canvas) return;

  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width || 600;
  canvas.height = rect.height || 450;
  const ctx = canvas.getContext("2d");

  // Assign initial circular positions
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const radius = Math.min(canvas.width, canvas.height) * 0.35;

  graphData.nodes.forEach((node, i) => {
    const angle = (i / graphData.nodes.length) * 2 * Math.PI;
    node.x = centerX + radius * Math.cos(angle);
    node.y = centerY + radius * Math.sin(angle);
    node.vx = 0;
    node.vy = 0;
  });

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Edges
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(6, 182, 212, 0.25)";
    graphData.edges.forEach(edge => {
      const source = graphData.nodes.find(n => n.id === edge.source);
      const target = graphData.nodes.find(n => n.id === edge.target);
      if (source && target) {
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();

        // Edge relation text
        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;
        ctx.fillStyle = "#64748b";
        ctx.font = "10px Fira Code";
        ctx.fillText(edge.relation, midX, midY);
      }
    });

    // Draw Nodes
    graphData.nodes.forEach(node => {
      ctx.beginPath();
      const nodeRadius = node.level === "high_level" ? 16 : 10;
      ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI);
      ctx.fillStyle = node.level === "high_level" ? "#a855f7" : "#06b6d4";
      ctx.shadowBlur = 10;
      ctx.shadowColor = node.level === "high_level" ? "rgba(168, 85, 247, 0.5)" : "rgba(6, 182, 212, 0.5)";
      ctx.fill();
      ctx.shadowBlur = 0;

      // Node label
      ctx.fillStyle = "#ffffff";
      ctx.font = "11px Inter";
      ctx.fillText(node.name, node.x - 20, node.y + nodeRadius + 14);
    });
  }

  draw();

  document.getElementById("btn-refresh-graph")?.addEventListener("click", () => {
    graphData.nodes.forEach(n => {
      n.x += (Math.random() - 0.5) * 40;
      n.y += (Math.random() - 0.5) * 40;
    });
    draw();
  });
}

// Speculative Decoding Benchmark
function setupSpeculativeBenchmark() {
  const btn = document.getElementById("btn-run-speculative");
  const inputPrompt = document.getElementById("input-speculative-prompt");

  btn?.addEventListener("click", async () => {
    const prompt = inputPrompt.value.trim();
    if (!prompt) return;

    btn.textContent = "⚡ Verifying Speculative Draft Tree...";
    try {
      const res = await fetch("/api/speculative/bench", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      document.getElementById("metric-baseline-tps").textContent = `${data.draftTokensPerSec} tok/s (${data.draftModel})`;
      document.getElementById("metric-speculative-tps").textContent = `${data.targetTokensPerSec} tok/s (${data.targetModel})`;
      document.getElementById("metric-speedup-factor").textContent = `${data.measuredSpeedupFactor}`;
      document.getElementById("metric-acceptance-rate").textContent = `${data.tokenOverlapAcceptanceRate}`;
      btn.textContent = "🚀 Run Speculative Speed Benchmark";
    } catch (e) {
      btn.textContent = "🚀 Benchmark Failed";
      document.getElementById("metric-baseline-tps").textContent = "Ollama unreachable";
      document.getElementById("metric-speculative-tps").textContent = e.message;
    }
  });
}

// DSPy Compiler
function setupDSPyCompiler() {
  const btn = document.getElementById("btn-compile-dspy");
  const inputName = document.getElementById("input-dspy-name");
  const inputDesc = document.getElementById("input-dspy-desc");
  const inputPrompt = document.getElementById("input-dspy-raw-prompt");
  const outputBox = document.getElementById("dspy-compiled-output");

  btn?.addEventListener("click", async () => {
    const name = inputName.value.trim();
    const desc = inputDesc.value.trim();
    const prompt = inputPrompt.value.trim();

    btn.textContent = "🧬 Compiling & Bootstrapping Few-Shots...";
    try {
      const res = await fetch("/api/dspy/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signature: {
            name,
            description: desc,
            inputs: [{ field: "code", type: "string", desc: "Input code snippet" }],
            outputs: [{ field: "optimized_code", type: "string", desc: "Clean production code" }]
          },
          prompt
        })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      document.getElementById("dspy-base-score").textContent = `${(data.baselineAccuracy * 100).toFixed(1)}%`;
      document.getElementById("dspy-compiled-score").textContent = `${(data.compiledAccuracy * 100).toFixed(1)}%`;
      document.getElementById("dspy-gain-badge").textContent = `${data.accuracyGain} Gain${data.liveEvaluated ? " (live)" : " (offline heuristic, not live-evaluated)"}`;
      outputBox.textContent = data.compiledPrompt + (data.notes ? `\n\n// ${data.notes}` : "");
      btn.textContent = "🧬 Compile & Optimize Prompt Signature";
    } catch (e) {
      outputBox.textContent = "Compilation Error: " + e.message;
      btn.textContent = "🧬 Compile";
    }
  });
}

// Dual-Level RAG Query
function setupRAGQuery() {
  const btn = document.getElementById("btn-execute-rag-query");
  const input = document.getElementById("input-rag-query");
  const outputBox = document.getElementById("rag-context-preview");

  btn?.addEventListener("click", async () => {
    const prompt = input.value.trim();
    if (!prompt) return;

    outputBox.textContent = "🕸️ Querying Dual-Level Graph (Low-Level Symbols + High-Level Themes)...";
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();

      const llmSection = data.llmUsed
        ? `🤖 [LLM Synthesis Output]:\n${data.synthesis}`
        : `⚠️ [LLM Synthesis Unavailable]: ${data.synthesisError || "Unknown error"}\n(Graph retrieval above is real; no LLM text was generated for this query.)`;

      outputBox.textContent = `${data.rag.synthesizedContext}\n\n${llmSection}`;
      fetchStatus();
    } catch (e) {
      outputBox.textContent = "RAG Query Error: " + e.message;
    }
  });
}

// 5. Google TurboQuant 4-Bit Vector Quantizer & QJL Search
function setupTurboQuant() {
  const btn = document.getElementById("btn-run-turboquant");
  const outputBox = document.getElementById("tq-results-preview");

  btn?.addEventListener("click", async () => {
    btn.textContent = "⚡ Quantizing Vector with QJL Transform...";
    try {
      const res = await fetch("/api/turboquant/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });
      const data = await res.json();

      let preview = "=== ⚡ GOOGLE TURBOQUANT (4-BIT QUANTIZED VECTOR SEARCH) ===\n";
      preview += "• Algorithm: Random Orthogonal Projection + Scalar 4-bit + 1-bit QJL Residual\n";
      preview += "• Compression Ratio: 7.8x (87.5% RAM Saved)\n\n[Ranked Similarity Results]:\n";

      data.results.forEach((r, i) => {
        preview += `\n#${i + 1} [Score: ${r.similarityScore.toFixed(4)}] ${r.name}\n   Summary: ${r.text}\n   Compression: ${r.compressionRatio}`;
      });

      outputBox.textContent = preview;
      btn.textContent = "⚡ Run TurboQuant Vector Quantization";
    } catch (e) {
      outputBox.textContent = "TurboQuant Error: " + e.message;
      btn.textContent = "⚡ Run TurboQuant";
    }
  });
}
