#!/bin/bash
cd "$(dirname "$0")"
echo "🕸️ Starting HyperRAG Studio on http://localhost:3003..."
bun server.ts
