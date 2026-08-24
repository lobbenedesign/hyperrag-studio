/**
 * 🌲 REAL HIERARCHICAL NAVIGABLE SMALL WORLD (HNSW) VECTOR INDEX
 * Pure mathematical implementation of Malkov & Yashunin (2018):
 * "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
 */

import { RealVectorEmbedder } from "./real_vector_embedder";

export interface HNSWNode {
  id: string;
  text: string;
  vector: number[];
  level: number;
  metadata?: Record<string, any>;
}

export interface SearchResult {
  id: string;
  text: string;
  similarity: number; // 0.0 - 1.0 Cosine Similarity
  distance: number;   // 1.0 - similarity
  level: number;
  metadata?: Record<string, any>;
}

export class HNSWVectorIndex {
  private embedder: RealVectorEmbedder;
  private nodes: Map<string, HNSWNode> = new Map();
  // layers[level][nodeId] = Set<neighborNodeId>
  private layers: Map<number, Map<string, Set<string>>> = new Map();
  private entryPointId: string | null = null;
  private maxLevel: number = 0;

  // Hyperparameters
  private M: number = 16;              // Max connections per element per layer
  private M0: number = 32;             // Max connections for ground layer (level 0)
  private efConstruction: number = 64; // Size of dynamic candidate list during construction
  private efSearch: number = 32;       // Size of dynamic candidate list during search
  private mL: number = 1 / Math.log(16); // Normalization factor for level generation

  constructor(embedder?: RealVectorEmbedder) {
    this.embedder = embedder || new RealVectorEmbedder();
    this.layers.set(0, new Map());
  }

  /**
   * Generates random layer level for new node
   */
  private getRandomLevel(): number {
    const r = Math.random();
    const l = Math.floor(-Math.log(r > 0 ? r : 0.0001) * this.mL);
    return Math.min(l, 5); // Cap max layers to 5
  }

  /**
   * Inserts text chunk into HNSW Index
   */
  public async insert(id: string, text: string, metadata?: Record<string, any>): Promise<HNSWNode> {
    const vector = await this.embedder.getEmbedding(text);
    const level = this.getRandomLevel();

    const node: HNSWNode = { id, text, vector, level, metadata };
    this.nodes.set(id, node);

    // Initialize layer adjacency sets for this node
    for (let l = 0; l <= level; l++) {
      if (!this.layers.has(l)) this.layers.set(l, new Map());
      this.layers.get(l)!.set(id, new Set<string>());
    }

    if (!this.entryPointId) {
      this.entryPointId = id;
      this.maxLevel = level;
      return node;
    }

    let currObj = this.entryPointId;
    let currDist = this.computeDistance(vector, this.nodes.get(currObj)!.vector);

    // 1. Search from top layer down to level + 1 (greedy 1-NN)
    for (let l = this.maxLevel; l > level; l--) {
      let changed = true;
      while (changed) {
        changed = false;
        const neighbors = this.layers.get(l)?.get(currObj) || new Set();
        for (const neighborId of neighbors) {
          const neighborNode = this.nodes.get(neighborId);
          if (!neighborNode) continue;
          const dist = this.computeDistance(vector, neighborNode.vector);
          if (dist < currDist) {
            currDist = dist;
            currObj = neighborId;
            changed = true;
          }
        }
      }
    }

    // 2. Search and connect from min(maxLevel, level) down to 0
    let enterNodes = [currObj];
    for (let l = Math.min(this.maxLevel, level); l >= 0; l--) {
      const candidates = this.searchLayer(vector, enterNodes, this.efConstruction, l);
      const maxM = (l === 0 ? this.M0 : this.M);

      // Select top-M closest neighbors
      const neighbors = candidates.slice(0, maxM);
      const nodeNeighbors = this.layers.get(l)!.get(id)!;

      for (const n of neighbors) {
        nodeNeighbors.add(n.id);
        const peerNeighbors = this.layers.get(l)!.get(n.id);
        if (peerNeighbors) {
          peerNeighbors.add(id);
          // Prune peer connections if exceeding maxM
          if (peerNeighbors.size > maxM) {
            this.pruneConnections(n.id, maxM, l);
          }
        }
      }

      enterNodes = candidates.map(c => c.id);
    }

    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entryPointId = id;
    }

    return node;
  }

  /**
   * Greedy layer search for candidate nearest neighbors
   */
  private searchLayer(queryVec: number[], enterNodeIds: string[], ef: number, level: number): { id: string; dist: number }[] {
    const visited = new Set<string>(enterNodeIds);
    const candidates: { id: string; dist: number }[] = [];
    const results: { id: string; dist: number }[] = [];

    for (const id of enterNodeIds) {
      const node = this.nodes.get(id);
      if (!node) continue;
      const dist = this.computeDistance(queryVec, node.vector);
      candidates.push({ id, dist });
      results.push({ id, dist });
    }

    candidates.sort((a, b) => a.dist - b.dist);
    results.sort((a, b) => a.dist - b.dist);

    while (candidates.length > 0) {
      const current = candidates.shift()!;
      const worstResult = results[results.length - 1];

      if (current.dist > worstResult.dist && results.length >= ef) {
        break;
      }

      const neighbors = this.layers.get(level)?.get(current.id) || new Set();
      for (const neighborId of neighbors) {
        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          const neighborNode = this.nodes.get(neighborId);
          if (!neighborNode) continue;

          const dist = this.computeDistance(queryVec, neighborNode.vector);
          if (dist < worstResult.dist || results.length < ef) {
            candidates.push({ id: neighborId, dist });
            results.push({ id: neighborId, dist });
            candidates.sort((a, b) => a.dist - b.dist);
            results.sort((a, b) => a.dist - b.dist);

            if (results.length > ef) {
              results.pop();
            }
          }
        }
      }
    }

    return results;
  }

  /**
   * Prunes node connections to nearest M
   */
  private pruneConnections(nodeId: string, maxM: number, level: number) {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    const neighbors = Array.from(this.layers.get(level)?.get(nodeId) || []);
    if (neighbors.length <= maxM) return;

    const scored = neighbors.map(nId => {
      const peer = this.nodes.get(nId);
      const dist = peer ? this.computeDistance(node.vector, peer.vector) : 999;
      return { id: nId, dist };
    });

    scored.sort((a, b) => a.dist - b.dist);
    const prunedSet = new Set(scored.slice(0, maxM).map(s => s.id));
    this.layers.get(level)!.set(nodeId, prunedSet);
  }

  /**
   * Performs k-Nearest Neighbors Approximate Search on query text
   */
  public async search(queryText: string, k: number = 5): Promise<SearchResult[]> {
    if (!this.entryPointId || this.nodes.size === 0) return [];

    const queryVec = await this.embedder.getEmbedding(queryText);
    let currObj = this.entryPointId;
    let currDist = this.computeDistance(queryVec, this.nodes.get(currObj)!.vector);

    // 1. Traverse upper layers to layer 0
    for (let l = this.maxLevel; l > 0; l--) {
      let changed = true;
      while (changed) {
        changed = false;
        const neighbors = this.layers.get(l)?.get(currObj) || new Set();
        for (const neighborId of neighbors) {
          const neighborNode = this.nodes.get(neighborId);
          if (!neighborNode) continue;
          const dist = this.computeDistance(queryVec, neighborNode.vector);
          if (dist < currDist) {
            currDist = dist;
            currObj = neighborId;
            changed = true;
          }
        }
      }
    }

    // 2. Perform search on ground layer 0
    const candidates = this.searchLayer(queryVec, [currObj], Math.max(this.efSearch, k), 0);

    return candidates.slice(0, k).map(c => {
      const node = this.nodes.get(c.id)!;
      const sim = Number((1.0 - c.dist).toFixed(4));
      return {
        id: node.id,
        text: node.text,
        similarity: Math.max(0.0, Math.min(1.0, sim)),
        distance: Number(c.dist.toFixed(4)),
        level: node.level,
        metadata: node.metadata
      };
    });
  }

  /**
   * True Cosine Distance = 1.0 - CosineSimilarity
   */
  private computeDistance(vecA: number[], vecB: number[]): number {
    const sim = this.embedder.cosineSimilarity(vecA, vecB);
    return Math.max(0.0, 1.0 - sim);
  }

  public getStats() {
    return {
      totalIndexedDocuments: this.nodes.size,
      maxGraphLayer: this.maxLevel,
      layerDistributions: Array.from(this.layers.entries()).map(([lvl, map]) => ({
        layer: lvl,
        nodeCount: map.size
      })),
      indexType: "Hierarchical Navigable Small World (HNSW)",
      distanceMetric: "Cosine Distance (L2-Normalized)"
    };
  }
}
