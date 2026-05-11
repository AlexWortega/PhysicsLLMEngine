/**
 * WebGPU-backed inference using @huggingface/transformers (ONNX runtime web).
 * Loads our fine-tuned LFM2-350M from `AlexWortega/lfm2-scenarios-ONNX`.
 *
 * The ONNX export has TWO known issues we work around at load time:
 *   - `model_q4f16.onnx` has Cast nodes around gqa_attention_bias that fail
 *     at WebGPU session-create. Confirmed on 3.7.x and 4.x EPs.
 *   - `model_q4.onnx` works in 3.7.x but regresses on the v4 C++ WebGPU EP for
 *     LFM2's hybrid conv+attention architecture (transformers.js #1599).
 *
 * Strategy: walk a priority list of (device, dtype) combos, log each failure,
 * and use the first session that creates successfully. If everything fails we
 * surface the full attempt log instead of hanging the boot overlay.
 */
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  env,
} from "@huggingface/transformers";

env.allowRemoteModels = true;
env.allowLocalModels = false;

const MODEL_ID = "AlexWortega/lfm2-scenarios-ONNX";

type Device = "webgpu" | "wasm";
type Dtype = "q4" | "q4f16" | "fp16";

interface LoadAttempt {
  device: Device;
  dtype: Dtype;
}

let _tokenizer: PreTrainedTokenizer | null = null;
let _model: PreTrainedModel | null = null;
let _loadingPromise: Promise<void> | null = null;

export let LAST_LOAD_INFO: {
  device: Device;
  dtype: Dtype;
  webgpuAvailable: boolean;
  isMobile: boolean;
  attempts: Array<{ device: Device; dtype: Dtype; error?: string }>;
} | null = null;

export interface LoadProgress {
  loaded: number;
  total: number;
  file?: string;
}

async function detectWebGPU(): Promise<boolean> {
  if (typeof navigator === "undefined") return false;
  // @ts-expect-error - navigator.gpu is the WebGPU entry point
  const gpu = navigator.gpu;
  if (!gpu) return false;
  try {
    const adapter = await gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

export function isMobileDevice(): boolean {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent || "";
  // iPad on iOS 13+ reports as Macintosh; check touch points to catch it.
  const iPadOS =
    /Mac/.test(ua) && typeof navigator.maxTouchPoints === "number" && navigator.maxTouchPoints > 1;
  return /Android|iPhone|iPod|Mobile|Tablet|Opera Mini|IEMobile/i.test(ua) || iPadOS;
}

function buildAttemptOrder(webgpuAvailable: boolean, mobile: boolean): LoadAttempt[] {
  // Mobile: WebGPU is absent on iOS Safari and chronically OOMs a 350M model on
  // most Android phones. Go straight to WASM — slow but stable.
  if (mobile || !webgpuAvailable) {
    return [{ device: "wasm", dtype: "q4" }];
  }
  return [
    // 4-bit weights + fp32 activations. Most compatible across EPs/GPUs.
    { device: "webgpu", dtype: "q4" },
    // Smaller VRAM footprint. Works on most modern discrete GPUs.
    { device: "webgpu", dtype: "fp16" },
    // Last resort — keeps the page usable instead of dying.
    { device: "wasm", dtype: "q4" },
  ];
}

export async function ensureModel(
  onProgress?: (p: LoadProgress) => void,
): Promise<{ model: PreTrainedModel; tokenizer: PreTrainedTokenizer }> {
  if (_model && _tokenizer) return { model: _model, tokenizer: _tokenizer };
  if (_loadingPromise) {
    await _loadingPromise;
    return { model: _model!, tokenizer: _tokenizer! };
  }

  _loadingPromise = (async () => {
    const webgpuAvailable = await detectWebGPU();
    const mobile = isMobileDevice();
    const attempts = buildAttemptOrder(webgpuAvailable, mobile);
    const attemptLog: Array<{ device: Device; dtype: Dtype; error?: string }> = [];

    _tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      progress_callback: (p) => {
        if (p && typeof p === "object" && "loaded" in p && "total" in p) {
          onProgress?.({
            loaded: Number((p as { loaded: number }).loaded),
            total: Number((p as { total: number }).total),
            file: (p as { file?: string }).file,
          });
        }
      },
    });

    let lastError: unknown = null;
    for (const attempt of attempts) {
      try {
        console.warn(
          `[tx.js] try device=${attempt.device} dtype=${attempt.dtype}`,
        );
        _model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
          device: attempt.device,
          dtype: attempt.dtype,
          progress_callback: (p) => {
            if (p && typeof p === "object" && "loaded" in p && "total" in p) {
              onProgress?.({
                loaded: Number((p as { loaded: number }).loaded),
                total: Number((p as { total: number }).total),
                file: (p as { file?: string }).file,
              });
            }
          },
        });
        attemptLog.push({ device: attempt.device, dtype: attempt.dtype });
        LAST_LOAD_INFO = {
          device: attempt.device,
          dtype: attempt.dtype,
          webgpuAvailable,
          isMobile: mobile,
          attempts: attemptLog,
        };
        console.warn(
          `[tx.js] LOADED · device=${attempt.device} dtype=${attempt.dtype}`,
        );
        return;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        attemptLog.push({
          device: attempt.device,
          dtype: attempt.dtype,
          error: msg,
        });
        console.error(
          `[tx.js] FAIL · device=${attempt.device} dtype=${attempt.dtype}: ${msg}`,
        );
        lastError = err;
        _model = null;
      }
    }
    LAST_LOAD_INFO = {
      device: attempts[attempts.length - 1].device,
      dtype: attempts[attempts.length - 1].dtype,
      webgpuAvailable,
      isMobile: mobile,
      attempts: attemptLog,
    };
    const summary = attemptLog
      .map((a) => `${a.device}+${a.dtype}: ${a.error ?? "ok"}`)
      .join(" | ");
    throw new Error(`All load attempts failed. ${summary}`, { cause: lastError });
  })();

  try {
    await _loadingPromise;
  } finally {
    _loadingPromise = null;
  }
  return { model: _model!, tokenizer: _tokenizer! };
}

export interface CompletionStreamOptions {
  prompt: string;
  maxTokens: number;
  temperature: number;
  topP?: number;
  signal?: AbortSignal;
  shouldStop: (fullText: string) => boolean;
  onTokenProgress?: (info: {
    tokens: number;
    text: string;
    piece: string;
  }) => void;
  onTokenInterval?: number;
}

/** Stream a completion. Returns the full generated text once finished/stopped. */
export async function streamCompletion(
  bundle: { model: PreTrainedModel; tokenizer: PreTrainedTokenizer },
  opts: CompletionStreamOptions,
): Promise<string> {
  const { model, tokenizer } = bundle;
  let fullText = "";
  let nTokens = 0;
  const interval = Math.max(1, opts.onTokenInterval ?? 1);
  let stopped = false;

  const inputs = await tokenizer(opts.prompt);

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function: (chunk: string) => {
      if (stopped) return;
      fullText += chunk;
      nTokens += 1;
      if (opts.onTokenProgress && nTokens % interval === 0) {
        opts.onTokenProgress({ tokens: nTokens, text: fullText, piece: chunk });
      }
      if (opts.shouldStop(fullText)) stopped = true;
    },
  });

  try {
    const greedy = opts.temperature < 0.001;
    await model.generate({
      ...(inputs as Record<string, unknown>),
      max_new_tokens: opts.maxTokens,
      do_sample: !greedy,
      ...(greedy
        ? {}
        : { temperature: opts.temperature, top_p: opts.topP ?? 0.95 }),
      streamer,
    } as unknown as Parameters<typeof model.generate>[0]);
  } catch (err) {
    console.error("[tx.js] generate error:", err);
    throw err;
  }
  return fullText;
}

export async function tokenize(
  bundle: { tokenizer: PreTrainedTokenizer },
  text: string,
): Promise<number[]> {
  const ids = bundle.tokenizer.encode(text);
  return Array.from(ids ?? []);
}
