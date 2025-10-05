import * as webllm from "@mlc-ai/web-llm";

export type LLMEngine = webllm.MLCEngineInterface;

/** Prefer these if present in this WebLLM build; otherwise pick the first available. */
const PREFERRED_MODELS = [
  "Llama-3.1-1B-chess-q4f16_1",
  "Llama-3.1-8B-Instruct-q4f16_1",
  "Llama-3-8B-Instruct-q4f16_1",
  "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
];

/** Return a model_id that actually exists in the current prebuilt config. */
function selectModel(): string {
  const list = (webllm.prebuiltAppConfig?.model_list ?? []).map(m => m.model_id);
  for (const id of PREFERRED_MODELS) if (list.includes(id)) return id;
  if (list.length === 0) throw new Error("No prebuilt models available in this WebLLM build.");
  return list[0];
}

/** Version-agnostic progress adapter: accepts number or {progress}. */
function progressAdapter(x: unknown): number {
  if (typeof x === "number") return x;
  if (x && typeof x === "object" && "progress" in (x as any)) return (x as any).progress ?? 0;
  return 0;
}

/**
 * Initialize WebLLM and return the engine. Updates the given status element and logs.
 * Never throws; returns null on failure (with UI messages already set).
 */
export async function initLLM(
  statusEl: HTMLElement,
  log: (msg: string) => void
): Promise<LLMEngine | null> {
  const modelId = selectModel();
  try {
    const engine = await webllm.CreateMLCEngine(modelId, {
      initProgressCallback: (r: unknown) => {
        const p = Math.round(progressAdapter(r) * 100);
        statusEl.textContent = `LLM: ${p}%`;
      },
      // logLevel: "INFO", // uncomment if you want verbose console logs
    });
    statusEl.textContent = `LLM: ready (${modelId})`;
    log(`WebLLM ready with ${modelId}`);
    return engine;
  } catch (e: any) {
    const available = (webllm.prebuiltAppConfig?.model_list ?? [])
      .map(m => m.model_id)
      .join(", ");
    statusEl.textContent = "LLM: failed";
    log(`WebLLM error: ${e?.message ?? String(e)}\nAvailable models: ${available || "(none)"}`);
    return null;
  }
}

/**
 * Ask the LLM to comment on a move/position.
 * Returns the model’s text (empty string on failure).
 */
export async function commentPosition(
  engine: LLMEngine,
  opts: {
    moveSAN: string;
    beforeFEN: string;
    afterFEN: string;
    evalBefore: number;
    evalAfter: number;
    bestMoveUCI?: string;
    pv?: string;
  }
): Promise<string> {
  const delta = opts.evalAfter - opts.evalBefore;
  const prompt = [
    "You are a concise chess coach. Explain the last move in two sentences.",
    `Before FEN: ${opts.beforeFEN}`,
    `After FEN: ${opts.afterFEN}`,
    `Move played (SAN): ${opts.moveSAN}`,
    `Engine eval before: ${opts.evalBefore}, after: ${opts.evalAfter}, delta: ${delta.toFixed(2)}`,
    `Engine best move (UCI): ${opts.bestMoveUCI ?? "-"}`,
    `Engine PV: ${opts.pv ?? "-"}`,
  ].join("\n");

  try {
    const out = await engine.chat.completions.create({
      messages: [
        { role: "system", content: "You are an expert chess coach providing clear, short explanations." },
        { role: "user", content: prompt },
      ],
      temperature: 0.6,
      max_tokens: 160,
    });
    return out.choices?.[0]?.message?.content?.trim() ?? "";
  } catch {
    return "";
  }
}
