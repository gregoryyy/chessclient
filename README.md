# Checkle: Chess Client

Chess app to (1) test-drive advanced, fully client-side logic and (2) a very fast prototyping approach using LLM-driven insights on advanced tech concepts.

Idea: Both the Stockfish chess engine and a large language model (LLM) run entirely in the browser — no server calls, no API keys, no data sharing. For IP protection, the code is obfuscated.

This project demonstrates how far modern web technology (WebAssembly + WebGPU) can go for AI-driven interactive apps.

---

## Overview

This app combines three core components, all executed locally:

1. **Stockfish.js (WASM)** – a top-tier chess engine compiled to WebAssembly.  
   Provides real-time position evaluation, best-move search, and tactical verification.

   Link: https://github.com/nmrugg/stockfish.js

2. **WebLLM** – an open-source WebGPU-based runtime that executes quantized LLMs directly in the browser (e.g., Phi-3, Llama 3, Mistral).  
   Powers the natural-language “coach” that explains moves and suggests ideas.

   Link: https://webllm.mlc.ai/

3. **JavaScript frontend** – built with plain JS and Web Components (`chessboard-element` + `chess.js`), featuring:  
   - Playable board with drag-and-drop pieces  
   - Local opening and tactics library  
   - Position evaluation and principal variation (PV) display  
   - LLM-based commentary for move understanding  
   - Automatic blunder detection and exercise generation

   Uses npm packages chess.js, chessground and @mlc-ai/web-llm

Everything runs client-side — your browser acts as both engine and coach.

---

## Features

- Play against yourself or analyze positions freely  
- Stockfish in WASM – fast, no server dependency  
- Local LLM commentary using [WebLLM](https://webllm.mlc.ai)  
  - Explains move ideas, missed plans, and tactical motifs  
  - Summarizes critical positions in natural language  
- Real-time analysis  
  - Engine evaluation (centipawns or mate)
  - Principal variation (best line)
- Mini opening library (ECO-style identification)
- Tactics motif panel with standard patterns
- Blunder detection  
  - Detects major eval drops  
  - Stores recent blunders as training puzzles
- Exercise generation  
  - Automatically creates puzzles from your own mistakes  
  - Shows side-to-move, best line, and evaluation swing
- Offline, private, zero API usage

---

## Architecture

| Component | Technology | Role |
|------------|-------------|------|
| UI / Logic | Plain JS + `chess.js` + `chessboard-element` | Game rules, moves, and board rendering |
| Engine | `stockfish.wasm` / `stockfish.js` | Position evaluation and move search |
| AI Commentary | `@mlc-ai/web-llm` (WebGPU) | Natural language reasoning on top of engine data |
| Storage | IndexedDB (future) | Save games, exercises, cached models |
| Deployment | Static HTML + JS | Runs on any HTTPS host (or localhost) |

The engine and LLM communicate asynchronously via Web Workers.  
LLM prompts are constructed from engine data (FEN, evaluation deltas, principal variation).

---

## Example Prompt (for the LLM)

You are a concise chess coach.
Before FEN: r1bqkbnr/pppppppp/2n5/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 2 2
After FEN: r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2
Move played (SAN): e4
Engine eval before: 0.00, after: +0.25, delta: +0.25
Engine best move: e4
Engine PV: e4 e5 Nf3 Nc6
Explain the move in 2 sentences: What’s the idea behind e4?


Example output:

> White stakes a claim in the center and opens lines for the bishop and queen.  
> This is a standard principled move in most openings, creating early activity.

---

## Design Goals

- Zero backend – all intelligence local  
- Experimentation platform for:
  - In-browser reasoning
  - Human-in-the-loop pattern generation
  - Offline AI tutoring
- Composable architecture – easy to extend:
  - Replace LLM
  - Add online openings database
  - Add cloud sync or sharing later

---

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/gregoryyy/chessclient
   cd chessclient
   ```

2. Add Stockfish build (e.g. from official WASM builds):
   ```bash
   cp <js loader> ./stockfish-...js
   cp <wasm file> ./stockfish-...wasm
   ```
   There are five available engines, see https://github.com/nmrugg/stockfish.js. Currently, the lite single-threaded version is included.

3. Install dependencies:
   ```bash
   npm init -y
   # runs: npm i chess.js chessground @mlc-ai/web-llm
   mpm i
   ```

4. Run server
   ```bash
   npx vite
   # OR
   python3 -m http.server 8000
   ```

5. Open URL ```https://localhost:8000``` etc.


