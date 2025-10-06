# Checkle: Local Chess + LLM Coach

Checkle is an entirely client-side chess laboratory that couples a WebAssembly build of Stockfish with a browser-run large language model. The goal is to explore fast prototyping workflows where strategic insights, tutoring, and experimentation all happen without servers or API keys.

---

## Motivation

- **All-local experience** – move analysis, coaching, and evaluation run in the browser via WebAssembly and WebGPU.
- **Rapid iteration** – the app doubles as a sandbox for testing prompt engineering, PEFT adapters, and custom quantized models.
- **Privacy by default** – no network round-trips for play or analysis; your games stay on the device.

---

## Feature Highlights

- Play, analyze, or step through custom positions with an interactive board powered by `chess.js` and `chessground`.
- Embedded Stockfish (single-thread WASM build) supplies centipawn/mate evaluations, best-line search, and blunder detection.
- WebLLM-hosted models provide natural-language commentary, move explanations, and tactical hints.
- Mistake harvesting turns large evaluation swings into tactics exercises that you can replay immediately.
- Opening cues, ECO-style labels, and motif panels surface thematic ideas while you explore.

---

## System Architecture

| Layer | Tech | Responsibility |
| --- | --- | --- |
| UI & Game Logic | Vanilla JS, `chess.js`, `chessground`, custom Web Components | Move validation, board rendering, history tracking |
| Chess Engine | `stockfish.js` (WASM build) | Fast evaluation, search, PV generation |
| LLM Runtime | `@mlc-ai/web-llm` on WebGPU | Coach responses, natural language summarization |
| Data Flow | Web Workers & `postMessage` | Keeps engine/LLM work off the main thread |
| Bundling | Vite | Local dev server and production build |

Prompts sent to the LLM are composed from FEN snapshots, engine evaluations, and principal variations so the coach understands context for each move.

---

## Repository Layout

```
public/           Static assets, including model bundles
src/              Frontend source (board, UI glue, prompt construction)
llm/              Finetuning scripts, LoRA helpers, and docs (see llm/README.md)
node_modules/     Local dependencies (installed via npm)
```

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.11+ (needed for LoRA training scripts and MLC tooling)

### Install & Run

```bash
# clone the repo
git clone https://github.com/gregoryyy/chessclient
cd chessclient

# install npm dependencies
npm install

# start the dev server
npm run dev
```

The dev server launches Vite on `http://localhost:5173` (or the next free port). Open the URL in a WebGPU-capable browser (Chrome, Edge, or Safari Technology Preview).

### Production Build

```bash
npm run build        # produces dist/ assets
npm run preview      # serves the production bundle locally
```

---

## Stockfish Assets

The repository ships with a light WASM build. To swap or upgrade engines, download an alternative from [stockfish.js](https://github.com/nmrugg/stockfish.js) and place the loader/wasm pair in `public/engines`, then update the import path in `src/llm.ts`.

---

## Working with LLMs

WebLLM expects models compiled by the MLC toolchain. To add or replace models:

1. Finetune (optional) using the LoRA workflow detailed in `llm/README.md`.
2. Run `mlc_llm convert`, `mlc_llm quantize`, and `mlc_llm build-web` to emit the WebGPU bundle.
3. Host the generated files under `public/models/<model-id>` and register the model in `src/llm.ts`.

Models load lazily in the browser; progress is surfaced via the UI so users know when the coach is ready.

---

## Development Tips

- Keep multiple engines/LLM configs handy; switching between them helps compare coaching styles and latency.
- Use the browser devtools console to watch WebLLM download and compilation logs when debugging model initialization.
- `npm run lint` and `npm run build` are useful sanity checks before shipping changes.

---

## Roadmap Ideas

- Reinforcement-style self-play loops to generate dynamic training data.
- IndexedDB caching for models, games, and tactics history.
- Multiplayer or remote evaluation fallbacks for low-end devices.
- Expanded library of quantized models tuned for chess analysis.

---

## Acknowledgements

- [Stockfish.js](https://github.com/nmrugg/stockfish.js) for the WebAssembly engine.
- [WebLLM](https://webllm.mlc.ai/) and the MLC team for pushing WebGPU inference forward.
- Open-source chess tooling (`chess.js`, `chessground`, and community datasets) that make rapid experimentation possible.

