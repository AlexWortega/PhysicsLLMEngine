import { Wllama, type AssetsPathConfig } from "@wllama/wllama";
import singleThreadWasmUrl from "@wllama/wllama/esm/single-thread/wllama.wasm?url";
import multiThreadWasmUrl from "@wllama/wllama/esm/multi-thread/wllama.wasm?url";

const MODEL_URL =
  "https://huggingface.co/AlexWortega/lfm2-scenarios-GGUF/resolve/main/lfm2-scenarios-Q4_K_M.gguf";

const N_CTX = 8192;

const CONFIG_PATHS: AssetsPathConfig = {
  "single-thread/wllama.wasm": singleThreadWasmUrl,
  "multi-thread/wllama.wasm": multiThreadWasmUrl,
};

let _wllama: Wllama | null = null;
let _loadingPromise: Promise<Wllama> | null = null;
export let LAST_LOAD_INFO: {
  isolated: boolean;
  sab: boolean;
  hwc: number;
  nThreads: number;
} | null = null;

export interface LoadProgress {
  loaded: number;
  total: number;
}

export async function ensureModel(
  onProgress?: (p: LoadProgress) => void,
): Promise<Wllama> {
  if (_wllama) return _wllama;
  if (_loadingPromise) return _loadingPromise;

  _loadingPromise = (async () => {
    const isolated =
      typeof crossOriginIsolated !== "undefined" ? crossOriginIsolated : false;
    const hwc = navigator.hardwareConcurrency ?? 1;
    const sab = typeof SharedArrayBuffer !== "undefined";
    // eslint-disable-next-line no-console
    console.warn(
      `[wllama] crossOriginIsolated=${isolated}, SharedArrayBuffer=${sab}, hardwareConcurrency=${hwc}`,
    );
    if (!isolated || !sab) {
      // eslint-disable-next-line no-console
      console.error(
        "[wllama] ⚠ multi-thread DISABLED (page is not cross-origin isolated). " +
          "Inference will fall back to single-thread WASM and be ~10x slower. " +
          "Open the Space in a NEW TAB instead of inside the iframe.",
      );
    }
    const w = new Wllama(CONFIG_PATHS, {
      logger: {
        debug: () => {},
        log: () => {},
        warn: console.warn,
        error: console.error,
      },
    });
    const nThreads = isolated && sab ? Math.min(hwc, 8) : 1;
    LAST_LOAD_INFO = { isolated, sab, hwc, nThreads };
    await w.loadModelFromUrl(MODEL_URL, {
      n_ctx: N_CTX,
      n_threads: nThreads,
      progressCallback: ({ loaded, total }) =>
        onProgress?.({ loaded, total }),
    });
    // eslint-disable-next-line no-console
    console.warn(`[wllama] model loaded with n_threads=${nThreads}`);
    _wllama = w;
    return w;
  })();

  try {
    return await _loadingPromise;
  } finally {
    _loadingPromise = null;
  }
}

export interface CompletionStreamOptions {
  prompt: string;
  maxTokens: number;
  temperature: number;
  topP?: number;
  signal?: AbortSignal;
  /** Return true to stop generation early (e.g. saw "next frame" boundary). */
  shouldStop: (fullText: string) => boolean;
  /** Fired roughly every onTokenInterval tokens during generation. */
  onTokenProgress?: (info: {
    tokens: number;
    text: string;
    piece: string;
  }) => void;
  onTokenInterval?: number;
}

/** Run a streaming completion; resolves with the full generated text. */
export async function streamCompletion(
  w: Wllama,
  opts: CompletionStreamOptions,
): Promise<string> {
  let fullText = "";
  const ctrl = new AbortController();
  if (opts.signal) {
    if (opts.signal.aborted) ctrl.abort();
    else opts.signal.addEventListener("abort", () => ctrl.abort());
  }

  let nTokens = 0;
  let prevLen = 0;
  const onTokenInterval = Math.max(1, opts.onTokenInterval ?? 4);
  try {
    const result = await w.createCompletion(opts.prompt, {
      nPredict: opts.maxTokens,
      sampling: {
        temp: opts.temperature,
        top_p: opts.topP ?? 0.95,
      },
      abortSignal: ctrl.signal,
      onNewToken: (_token, _piece, currentText) => {
        const piece = currentText.slice(prevLen);
        prevLen = currentText.length;
        fullText = currentText;
        nTokens += 1;
        if (opts.onTokenProgress && nTokens % onTokenInterval === 0) {
          opts.onTokenProgress({ tokens: nTokens, text: currentText, piece });
        }
        if (opts.shouldStop(currentText)) ctrl.abort();
      },
    });
    return typeof result === "string" ? result : fullText;
  } catch (err) {
    if ((err as Error)?.name === "WllamaAbortError") return fullText;
    throw err;
  }
}

export async function tokenize(w: Wllama, text: string): Promise<number[]> {
  const enc = await w.tokenize(text);
  return Array.from(enc);
}
