/**
 * 🕸️ Real LLM-driven graph ingestion.
 *
 * This is the piece the "Honest Status" table in the README calls out as
 * missing: `LightRAGEngine`'s graph is hand-seeded data, not extracted from
 * anything. Real LightRAG and Microsoft GraphRAG both build their knowledge
 * graphs by running an LLM entity/relationship extraction pass over document
 * chunks (GraphRAG's own paper and repo call this step "element extraction" —
 * see https://github.com/microsoft/graphrag). This module does the same
 * *technique* against local Ollama: real HTTP calls, real JSON responses,
 * real (imperfect) LLM extraction — not a hardcoded graph, and not a claim
 * that this reimplements GraphRAG's actual extraction prompts or its later
 * community-summarization stages, which this does not attempt.
 *
 * No fabricated fallback: if Ollama is unreachable or returns something that
 * doesn't parse as the expected shape, this throws instead of returning a
 * fake extraction — same policy as the rest of this codebase after the
 * audit (see README "Honest Status" and CHANGELOG).
 */

export interface ExtractedEntity {
  name: string;
  type: "function" | "class" | "module" | "concept" | "api" | "database" | "theme";
  level: "low_level" | "high_level";
  description: string;
  tags: string[];
}

export interface ExtractedRelation {
  source: string; // entity name, resolved to a node id by the caller
  target: string;
  relation: string;
  weight: number;
}

export interface ChunkExtraction {
  chunkIndex: number;
  chunkText: string;
  entities: ExtractedEntity[];
  relations: ExtractedRelation[];
  raw: string; // the model's raw JSON response, kept for debugging/audit
}

export interface GraphIngestResult {
  chunks: ChunkExtraction[];
  totalEntitiesExtracted: number;
  totalRelationsExtracted: number;
  model: string;
}

const EXTRACTION_SYSTEM_PROMPT = `You extract a knowledge graph from a text chunk for a dual-level GraphRAG-style index.
Return ONLY a JSON object, no prose, matching exactly this shape:
{
  "entities": [
    {"name": string, "type": "function"|"class"|"module"|"concept"|"api"|"database"|"theme", "level": "low_level"|"high_level", "description": string, "tags": string[]}
  ],
  "relations": [
    {"source": string, "target": string, "relation": string, "weight": number}
  ]
}
Rules:
- "low_level" entities are concrete things named in the text: functions, classes, modules, APIs, specific mechanisms.
- "high_level" entities are broader themes/concepts the text is about.
- Every "source" and "target" in relations MUST exactly match a "name" of some entity you extracted.
- "weight" is your confidence in the relation, between 0 and 1.
- If the chunk has no clear entities, return {"entities": [], "relations": []}. Do not invent entities not grounded in the text.
- Extract at most 8 entities and 10 relations per chunk.`;

function chunkText(text: string, maxChunkChars = 1400): string[] {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const chunks: string[] = [];
  let current = "";
  for (const p of paragraphs) {
    if (current.length + p.length + 2 > maxChunkChars && current.length > 0) {
      chunks.push(current);
      current = p;
    } else {
      current = current.length > 0 ? `${current}\n\n${p}` : p;
    }
  }
  if (current.length > 0) chunks.push(current);

  // A single paragraph longer than maxChunkChars (no blank-line breaks at
  // all) still needs to be split, or it goes to the model whole.
  return chunks.flatMap((c) => {
    if (c.length <= maxChunkChars) return [c];
    const parts: string[] = [];
    for (let i = 0; i < c.length; i += maxChunkChars) {
      parts.push(c.slice(i, i + maxChunkChars));
    }
    return parts;
  });
}

function validateExtraction(parsed: any): { entities: ExtractedEntity[]; relations: ExtractedRelation[] } {
  if (!parsed || typeof parsed !== "object") {
    throw new Error("extraction response was not a JSON object");
  }
  const rawEntities = Array.isArray(parsed.entities) ? parsed.entities : [];
  const rawRelations = Array.isArray(parsed.relations) ? parsed.relations : [];

  const validTypes = new Set(["function", "class", "module", "concept", "api", "database", "theme"]);
  const validLevels = new Set(["low_level", "high_level"]);

  const entities: ExtractedEntity[] = [];
  for (const e of rawEntities) {
    if (!e || typeof e.name !== "string" || e.name.trim().length === 0) continue;
    entities.push({
      name: e.name.trim(),
      type: validTypes.has(e.type) ? e.type : "concept",
      level: validLevels.has(e.level) ? e.level : "low_level",
      description: typeof e.description === "string" ? e.description : "",
      tags: Array.isArray(e.tags) ? e.tags.filter((t: any) => typeof t === "string") : []
    });
  }

  const entityNames = new Set(entities.map((e) => e.name.toLowerCase()));
  const relations: ExtractedRelation[] = [];
  for (const r of rawRelations) {
    if (!r || typeof r.source !== "string" || typeof r.target !== "string") continue;
    // Only keep relations that are actually grounded in extracted entities —
    // a model that names a relation endpoint it didn't also extract as an
    // entity is hallucinating the edge, so drop it rather than insert a
    // dangling reference.
    if (!entityNames.has(r.source.toLowerCase()) || !entityNames.has(r.target.toLowerCase())) continue;
    relations.push({
      source: r.source,
      target: r.target,
      relation: typeof r.relation === "string" && r.relation.trim() ? r.relation.trim() : "relates_to",
      weight: typeof r.weight === "number" && r.weight >= 0 && r.weight <= 1 ? r.weight : 0.5
    });
  }

  return { entities, relations };
}

export async function extractGraphFromText(
  text: string,
  opts: { ollamaHost: string; model: string; timeoutMs?: number }
): Promise<GraphIngestResult> {
  if (!text || text.trim().length === 0) {
    throw new Error("ingest text is empty");
  }

  const chunks = chunkText(text);
  const results: ChunkExtraction[] = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunkTextPiece = chunks[i];
    const res = await fetch(`${opts.ollamaHost}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: opts.model,
        format: "json",
        messages: [
          { role: "system", content: EXTRACTION_SYSTEM_PROMPT },
          { role: "user", content: chunkTextPiece }
        ],
        stream: false
      }),
      signal: AbortSignal.timeout(opts.timeoutMs ?? 60000)
    });

    if (!res.ok) {
      throw new Error(
        `Ollama returned HTTP ${res.status} while extracting entities from chunk ${i + 1}/${chunks.length}`
      );
    }

    const data: any = await res.json();
    const raw: string = data.message?.content ?? "";
    if (!raw) {
      throw new Error(`Ollama returned no content for chunk ${i + 1}/${chunks.length} (model: ${opts.model})`);
    }

    let parsed: any;
    try {
      parsed = JSON.parse(raw);
    } catch (e: any) {
      throw new Error(
        `Ollama's response for chunk ${i + 1}/${chunks.length} was not valid JSON despite format:"json": ${e.message}`
      );
    }

    const { entities, relations } = validateExtraction(parsed);
    results.push({ chunkIndex: i, chunkText: chunkTextPiece, entities, relations, raw });
  }

  return {
    chunks: results,
    totalEntitiesExtracted: results.reduce((n, c) => n + c.entities.length, 0),
    totalRelationsExtracted: results.reduce((n, c) => n + c.relations.length, 0),
    model: opts.model
  };
}
