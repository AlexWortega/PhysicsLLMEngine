import type { FrameSnapshot, SceneHeader, SimulateEvent } from "./types";
import {
  extractFirstFrameRaw,
  fmtFrame,
  fmtHeader,
  parseFrame,
  splitFirstFrame,
} from "./promptFormat";
import { ensureModel, streamCompletion, tokenize } from "./transformersEngine";

interface SimulateRequest {
  header: SceneHeader;
  initial_frames: FrameSnapshot[];
  n_frames: number;
  temperature: number;
}

export interface SimulateCallbacks {
  /** Fires every few tokens during a frame's generation — for UI liveness. */
  onTokenProgress?: (info: {
    step: number;
    total: number;
    tokens: number;
    text: string;
    piece: string;
  }) => void;
  /** Receives free-form verbose log lines for the UI log panel. */
  onLog?: (line: string) => void;
  /** Fires once per step with the EXACT prompt text fed to the model. */
  onPrompt?: (info: { step: number; total: number; prompt: string }) => void;
}

const N_CTX = 8192;
// One frame ≈ 70 chars × 22 obj / 4 chars-per-token ≈ 380 tok. We give a
// little headroom for the model to emit an explicit frame boundary. 600 is
// comfortably above all 30 scenario sizes (max 39 obj = ~480 tok).
const MAX_NEW = 600;
const HEADROOM = 128;

async function fitPrompt(
  wllama: Awaited<ReturnType<typeof ensureModel>>,
  prompt: string,
  onLog?: (s: string) => void,
): Promise<string> {
  // Drop oldest frames ONE AT A TIME until the prompt fits the budget. This
  // preserves as much of the prefix as possible across consecutive rollout
  // steps so llama.cpp's internal KV cache keeps hitting on shared tokens.
  // (Eager capContext(...,4) on every step would shift the prefix and force
  // re-tokenization of the entire body, wasting all the cache savings.)
  const budget = N_CTX - MAX_NEW - HEADROOM;
  let toks = await tokenize(wllama, prompt);
  onLog?.(`tokenize prompt: ${toks.length} tok (budget ${budget})`);
  if (toks.length <= budget) return prompt;

  const idx = prompt.indexOf("Frame ");
  if (idx < 0) return prompt;
  const head = prompt.slice(0, idx);
  const body = prompt.slice(idx);
  const frames = body.split(/(?=Frame \d+:)/).filter(Boolean);
  onLog?.(`overflow: ${toks.length}>${budget}, dropping from ${frames.length} frames`);
  let kept = frames;
  while (kept.length > 1) {
    kept = kept.slice(1);
    const candidate = head + kept.join("");
    toks = await tokenize(wllama, candidate);
    if (toks.length <= budget) {
      onLog?.(`fitPrompt: kept ${kept.length} frames (${toks.length} tok)`);
      return candidate;
    }
  }
  onLog?.(`fitPrompt: catastrophic — body fully dropped`);
  return head + "Predict next frame:";
}

export async function* simulateStream(
  payload: SimulateRequest,
  signal?: AbortSignal,
  cb?: SimulateCallbacks,
): AsyncGenerator<SimulateEvent, void, void> {
  const nFrames = Math.max(2, Math.min(60, payload.n_frames));

  const log = (s: string) => cb?.onLog?.(s);
  log(`simulateStream: ${nFrames} frames, n_obj from header`);

  yield { type: "status", msg: "Loading model…" };
  log("ensureModel: checking cache");

  let lastPct = -1;
  const wllama = await ensureModel(({ loaded, total }) => {
    if (!total) return;
    const pct = Math.round((loaded / total) * 100);
    if (pct !== lastPct && pct % 5 === 0) {
      lastPct = pct;
      log(`download: ${pct}% (${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB)`);
    }
  }).catch((err) => {
    log(`model load FAILED: ${String(err)}`);
    throw new Error(`model load failed: ${String(err)}`);
  });
  log("model loaded");

  if (signal?.aborted) return;

  const header = payload.header;
  const initial = payload.initial_frames;
  const nObj =
    header.object_count ?? header.objects?.length ?? initial[0]?.objects.length ?? 0;

  const rolled: Array<Record<number, FrameSnapshot["objects"][number]>> =
    initial.map((fr) => Object.fromEntries(fr.objects.map((o) => [o.id, o])));
  let ctx = fmtHeader(header) + initial.map(fmtFrame).join("");
  let lastIdx = initial.length > 0 ? initial[initial.length - 1].frame : 0;

  yield { type: "ready", n_objects: nObj, n_frames: nFrames };

  const t0 = performance.now();
  let totalTok = 0;

  for (let step = 0; step < nFrames; step++) {
    if (signal?.aborted) return;
    log(`--- step ${step + 1}/${nFrames} ---`);

    const tStep = performance.now();
    let prompt = ctx + "Predict next frame:";
    prompt = await fitPrompt(wllama, prompt, log);
    log(`prompt ready: ${prompt.length} chars`);
    cb?.onPrompt?.({ step: step + 1, total: nFrames, prompt });

    let text: string;
    let stopReason = "max_tokens";
    let tokenCount = 0;
    try {
      text = await streamCompletion(wllama, {
        prompt,
        maxTokens: MAX_NEW,
        temperature: Math.max(payload.temperature, 0.0),
        topP: 0.95,
        signal,
        onTokenInterval: 1,
        onTokenProgress: ({ tokens, text, piece }) => {
          tokenCount = tokens;
          cb?.onTokenProgress?.({
            step: step + 1,
            total: nFrames,
            tokens,
            text,
            piece,
          });
        },
        shouldStop: (full) => {
          const matches = full.match(/Frame\s+\d+:/g);
          if (matches && matches.length >= 2) {
            stopReason = "frame-boundary";
            return true;
          }
          if (full.includes("Predict next frame:")) {
            stopReason = "predict-suffix";
            return true;
          }
          return false;
        },
      });
    } catch (err) {
      log(`generation ERROR: ${String(err)}`);
      yield { type: "error", message: String(err) };
      return;
    }

    const tGen = performance.now() - tStep;
    log(`generated ${tokenCount} tok in ${tGen.toFixed(0)}ms (${(tokenCount / (tGen / 1000)).toFixed(1)} t/s) · stop=${stopReason}`);

    if (step < 2) {
      // eslint-disable-next-line no-console
      console.log("[physics-llm] step", step + 1, "raw model output:\n" + text);
    }

    if (signal?.aborted) return;

    const first = splitFirstFrame(text);
    let newObjs = parseFrame(first, nObj);

    const hasFrameHeader = /Frame\s+\d+:/.test(text);
    log(
      `parsed ${newObjs.length}/${nObj} obj` +
        (hasFrameHeader ? "" : " · DRIFT (no Frame header)"),
    );

    // Compute mean motion vs prev frame as a "shaking detector"
    const prev = rolled[rolled.length - 1];
    if (newObjs.length > 0) {
      let sumD = 0,
        cnt = 0;
      for (const o of newObjs) {
        const p = prev[o.id];
        if (p) {
          const dx = o.position.x - p.position.x;
          const dy = o.position.y - p.position.y;
          sumD += Math.sqrt(dx * dx + dy * dy);
          cnt += 1;
        }
      }
      if (cnt > 0) {
        const mean = sumD / cnt;
        log(`mean Δpos = ${mean.toFixed(3)} px${mean < 0.05 ? " · ⚠ frozen" : ""}`);
      }
    }

    if (newObjs.length === 0) {
      log(`parse FAILED — no obj_X matches; holding previous frame`);
      newObjs = Object.values(prev);
    } else if (newObjs.length < nObj) {
      log(`partial parse: ${newObjs.length}/${nObj} — filling missing from prev`);
      const haveIds = new Set(newObjs.map((o) => o.id));
      for (const id of Object.keys(prev).map(Number)) {
        if (!haveIds.has(id)) newObjs.push(prev[id]);
      }
    }
    const objMap = Object.fromEntries(newObjs.map((o) => [o.id, o]));
    rolled.push(objMap);
    lastIdx += 1;

    const elapsed = (performance.now() - t0) / 1000;
    totalTok += Math.max(1, Math.floor(first.length / 4));

    // Round-trip into context. If the model emitted a "Frame N:" header,
    // splice that block as-is. If it drifted and dropped the header (common
    // after a few autoregressive steps on small scenes), reformat with our
    // own header so next step's prompt remains in-distribution.
    let rawAppend = extractFirstFrameRaw(text);
    const hasHeader = /Frame\s+\d+:/.test(rawAppend);
    if (!hasHeader || rawAppend.trim().length === 0) {
      rawAppend = fmtFrame({
        frame: lastIdx,
        description: `Frame ${lastIdx}: All objects are in motion.`,
        objects: newObjs,
      });
    }
    // Append-only: keep prompt prefix stable so llama.cpp re-uses KV cache.
    // fitPrompt() at the next step's start handles overflow by dropping
    // oldest frames lazily, only when budget is actually hit.
    ctx = ctx + rawAppend;

    yield {
      type: "frame",
      step: step + 1,
      total: nFrames,
      frame_idx: lastIdx,
      objects: newObjs,
      elapsed: Number(elapsed.toFixed(2)),
      tps: Number((totalTok / Math.max(elapsed, 0.001)).toFixed(1)),
      raw: first.trim().slice(0, 600),
    };
  }

  yield { type: "done", elapsed: Number(((performance.now() - t0) / 1000).toFixed(2)) };
}

export async function prepareModel(
  onProgress?: (pct: number) => void,
): Promise<void> {
  await ensureModel(({ loaded, total }) => {
    if (!total) return;
    onProgress?.(Math.round((loaded / total) * 100));
  });
}

export async function fetchScenarios(): Promise<
  Array<{ name: string; header: SceneHeader; initial_frames: FrameSnapshot[] }>
> {
  const resp = await fetch("/api/scenarios");
  if (!resp.ok) throw new Error(`scenarios failed: ${resp.status}`);
  return resp.json();
}
