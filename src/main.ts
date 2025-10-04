import { Chess } from "chess.js";
import { Chessground } from "chessground";
import * as webllm from "@mlc-ai/web-llm";

// ---------- Config ----------
const STOCKFISH_URL = "/stockfish-17.1-lite-single-03e3232.js"; // served from public/
const LLM_MODEL = "Phi-3-mini-4k-instruct-q4f16_1";

// ---------- UI ----------
const boardRoot = document.getElementById("board") as HTMLElement;
const engineStatusEl = document.getElementById("engineStatus")!;
const llmStatusEl = document.getElementById("llmStatus")!;
const pvEl = document.getElementById("pv")!;
const evalEl = document.getElementById("eval")!;
const openEl = document.getElementById("openingName")!;
const themesEl = document.getElementById("themes")!;
const llmOutEl = document.getElementById("llmOut")!;
const logEl = document.getElementById("log")!;
const exerciseArea = document.getElementById("exerciseArea")!;

// ---------- Game/Engine/LLM state ----------
const game = new Chess();
let cg: any; // chessground instance
let engine: Worker | null = null;
let engineReady = false;
let lastAnalysis: { best?: string; pv?: string; eval?: string } = {};
let lastBlunder: { fen: string; playedSAN: string; delta: number; best?: string; pv?: string } | null = null;
let llmEngine: webllm.MLCEngineInterface | null = null;
let orientation: "white" | "black" = "white";

// ---------- Tiny opening & motif seed ----------
const OPENING_BOOK = [
  {
    name: "Italian Game", fens: [
      "rnbqkbnr/pppppppp/8/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -",
      "rnbqkbnr/pppp1ppp/8/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq -"
    ]
  },
  {
    name: "Sicilian Defence", fens: [
      "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    ]
  }
];
const TACTIC_THEMES = ["fork", "pin", "skewer", "discovered attack", "remove defender", "deflection"];

// ---------- Helpers ----------
function appendLog(msg: string) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}
function detectOpeningByFEN(fen: string): string {
  const key = fen.split(" ").slice(0, 4).join(" ");
  for (const o of OPENING_BOOK) if (o.fens.some(f => key.startsWith(f))) return o.name;
  return "—";
}
const SQUARES = (() => {
  const files = "abcdefgh".split(""), ranks = "12345678".split("");
  const out: string[] = [];
  for (const r of ranks) for (const f of files) out.push(f + r);
  return out;
})();
function toDests(ch: Chess) {
  // chessground expects a Map<from, string[] of to>
  const dests = new Map<string, string[]>();
  for (const s of SQUARES) {
    const moves = ch.moves({ square: s as any, verbose: true });
    if (moves.length) dests.set(s, moves.map(m => m.to));
  }
  return dests;
}
function setBoardFromGame() {
  cg.set({
    fen: game.fen(),
    movable: {
      color: game.turn() === "w" ? "white" : "black",
      dests: toDests(game),
      free: false
    }
  });
}

// ---------- Stockfish (WASM) ----------
function initEngine() {
  engine = new Worker(STOCKFISH_URL);
  engine.onmessage = (e: MessageEvent) => {
    const line = String((e.data as any)?.data ?? e.data ?? "");
    if (!line) return;

    if (line.startsWith("uciok")) {
      engineReady = true;
      engineStatusEl.textContent = "Engine: ready";
      appendLog("Stockfish ready");
    }
    if (line.startsWith("info")) {
      const scoreMatch = line.match(/\bscore (cp|mate) (-?\d+)/);
      const pvMatch = line.match(/\bpv (.+)$/);
      if (scoreMatch) {
        const kind = scoreMatch[1];
        const val = parseInt(scoreMatch[2], 10);
        evalEl.textContent = kind === "cp" ? (val / 100).toFixed(2) : (val > 0 ? `#${val}` : `#-${-val}`);
      }
      if (pvMatch) pvEl.textContent = pvMatch[1];
    }
    if (line.startsWith("bestmove")) {
      const uci = line.split(" ")[1];
      lastAnalysis = { best: uci, pv: pvEl.textContent, eval: evalEl.textContent };
    }
  };
  engine.postMessage("uci");
}
function analyzeCurrentPosition(ms = 500, multiPV = 1) {
  if (!engineReady || !engine) return;
  engine.postMessage("ucinewgame");
  engine.postMessage(`position fen ${game.fen()}`);
  engine.postMessage(`setoption name MultiPV value ${multiPV}`);
  engine.postMessage(`go movetime ${ms}`);
}

// ---------- WebLLM ----------
import { MLCEngine } from "@mlc-ai/web-llm";
type ProgressReport = { progress?: number; text?: string };
async function initLLM() {
  try {
    const llmEngine = new MLCEngine({
      initProgressCallback: (r: ProgressReport) => {
        const p = r.progress ?? 0;
        llmStatusEl.textContent = `LLM: ${Math.round(p * 100)}%`;
      },
    });
    await llmEngine.reload("Phi-3-mini-4k-instruct-q4f16_1");


    llmStatusEl.textContent = "LLM: ready";
    appendLog("WebLLM ready");
  } catch (e: any) {
    llmStatusEl.textContent = "LLM: failed";
    appendLog("WebLLM error: " + (e?.message ?? String(e)));
  }
}
async function coachComment(moveSAN: string, beforeFEN: string, afterFEN: string, evalBefore: number, evalAfter: number) {
  if (!llmEngine) return;
  const delta = evalAfter - evalBefore;
  const prompt = [
    "You are a concise chess coach. Explain the last move in two sentences.",
    `Before FEN: ${beforeFEN}`,
    `After FEN: ${afterFEN}`,
    `Move played (SAN): ${moveSAN}`,
    `Engine eval before: ${evalBefore}, after: ${evalAfter}, delta: ${delta.toFixed(2)}`,
    `Engine best move (UCI): ${lastAnalysis?.best ?? "-"}`,
    `Engine PV: ${lastAnalysis?.pv ?? "-"}`
  ].join("\n");
  const out = await llmEngine.chat.completions.create({
    messages: [
      { role: "system", content: "You are an expert chess coach providing clear, short explanations." },
      { role: "user", content: prompt }
    ],
    temperature: 0.6,
    max_tokens: 160
  });
  llmOutEl.textContent = out.choices?.[0]?.message?.content?.trim() || "(no output)";
}

// ---------- Blunder → exercise ----------
function maybeMarkBlunder(beforeEval: number, afterEval: number, playedSAN: string) {
  const drop = afterEval - beforeEval;
  if (drop <= -1.5) {
    lastBlunder = {
      fen: game.fen(),
      playedSAN,
      delta: drop,
      best: lastAnalysis?.best,
      pv: lastAnalysis?.pv
    };
    appendLog(`Blunder: Δ=${drop.toFixed(2)} SAN=${playedSAN} best=${lastBlunder.best ?? "?"}`);
  }
}
function renderExercise(ex: { type: string; side: string; fen: string; delta: number; best?: string; pv?: string }) {
  exerciseArea.innerHTML = `
    <div class="row">
      <span class="tag">${ex.type.toUpperCase()}</span>
      <span>Side to move: <strong>${ex.side}</strong></span>
      <span>Eval swing: ${ex.delta.toFixed(2)}</span>
    </div>
    <div id="exBoard" style="width:360px;height:360px"></div>
    <div class="row"><button id="showSolution">Show Engine Line</button></div>
    <div id="solution" class="mono"></div>
  `;
  // render static diagram
  const exRoot = document.getElementById("exBoard") as HTMLElement;
  Chessground(exRoot, { fen: ex.fen, viewOnly: true });
  (document.getElementById("showSolution") as HTMLButtonElement).onclick = () => {
    (document.getElementById("solution") as HTMLElement).textContent =
      `Best (UCI): ${ex.best ?? "?"}\nPV: ${ex.pv ?? "-"}`;
  };
}

// ---------- Move handling (chessground) ----------
function onMove(from: string, to: string) {
  const beforeFEN = game.fen(); // FEN before attempting the move
  const m = game.move({ from, to, promotion: "q" });
  if (!m) {
    // illegal: reset
    setBoardFromGame();
    return;
  }

  // trivial “themes” + opening name
  openEl.textContent = detectOpeningByFEN(beforeFEN);
  themesEl.innerHTML = "";
  TACTIC_THEMES.forEach(t => {
    const s = document.createElement("span");
    s.className = "tag";
    s.textContent = t;
    themesEl.appendChild(s);
  });

  // Analyze before
  game.undo();
  setBoardFromGame();
  analyzeCurrentPosition(350);
  // wait a moment for eval update
  setTimeout(() => {
    const evalBefore = parseFloat(evalEl.textContent) || 0;

    // redo the move, analyze after
    game.move({ from, to, promotion: "q" });
    setBoardFromGame();
    analyzeCurrentPosition(550);

    setTimeout(async () => {
      const evalAfter = parseFloat(evalEl.textContent) || 0;
      maybeMarkBlunder(evalBefore, evalAfter, m.san);
      await coachComment(m.san, beforeFEN, game.fen(), evalBefore, evalAfter);
    }, 600);
  }, 380);
}

// ---------- Controls ----------
(document.getElementById("newGame") as HTMLButtonElement).onclick = () => {
  game.reset();
  setBoardFromGame();
  pvEl.textContent = "—"; evalEl.textContent = "—"; llmOutEl.textContent = "—";
  openEl.textContent = "—"; themesEl.innerHTML = ""; lastBlunder = null;
};
(document.getElementById("flip") as HTMLButtonElement).onclick = () => {
  orientation = orientation === "white" ? "black" : "white";
  cg.set({ orientation });
};
(document.getElementById("takeback") as HTMLButtonElement).onclick = () => {
  game.undo();
  setBoardFromGame();
  analyzeCurrentPosition(350);
};
(document.getElementById("bestMove") as HTMLButtonElement).onclick = () => {
  analyzeCurrentPosition(800);
};
(document.getElementById("makeExercise") as HTMLButtonElement).onclick = () => {
  if (!lastBlunder) { exerciseArea.textContent = "No recent blunder to convert."; return; }
  const sideToMove = game.turn() === "w" ? "Black" : "White";
  renderExercise({ type: "tactic", side: sideToMove, ...lastBlunder });
};

// ---------- Boot ----------
(async function boot() {
  // init board
  cg = Chessground(boardRoot, {
    fen: game.fen(),
    orientation,
    movable: {
      color: "white",
      free: false,
      dests: toDests(game)
    },
    events: { move: onMove }
  });

  // engines
  initEngine();
  await initLLM();

  // set analysis mode once ready
  const wait = setInterval(() => {
    if (engineReady && engine) {
      clearInterval(wait);
      engine.postMessage("setoption name UCI_AnalyseMode value true");
    }
  }, 100);
})();

