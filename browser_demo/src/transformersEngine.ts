/**
 * WebGPU-backed inference using @huggingface/transformers (ONNX runtime web).
 * Loads our fine-tuned LFM2-350M from `AlexWortega/lfm2-scenarios-ONNX`.
 *
 * Public API mirrors wllamaEngine so streamClient can swap engines via a flag.
 */
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  env,
} from "@huggingface/transformers";

// Allow the library to fetch models cross-origin from HF CDN
env.allowRemoteModels = true;
env.allowLocalModels = false;

const MODEL_ID = "AlexWortega/lfm2-scenarios-ONNX";

let _tokenizer: PreTrainedTokenizer | null = null;
let _model: PreTrainedModel | null = null;
let _loadingPromise: Promise<void> | null = null;
export let LAST_LOAD_INFO: {
  device: string;
  dtype: string;
  webgpuAvailable: boolean;
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
    const device = webgpuAvailable ? "webgpu" : "wasm";
    // q4 (4-bit weights, fp32 activations) — works on both WebGPU and WASM.
    // q4f16 attempted but our patched ONNX has gqa_attention_bias/Cast nodes
    // that error out at session-create time when converted to fp16, even
    // with op_block_list. Would need a clean re-export from base LFM2 to fix.
    const dtype = "q4";
    LAST_LOAD_INFO = { device, dtype, webgpuAvailable };
    // eslint-disable-next-line no-console
    console.warn(`[tx.js] device=${device} dtype=${dtype} webgpu=${webgpuAvailable}`);

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

    _model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      device,
      dtype,
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
    // eslint-disable-next-line no-console
    console.warn(`[tx.js] model loaded`);
  })();

  await _loadingPromise;
  _loadingPromise = null;
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
  // eslint-disable-next-line no-console
  console.warn("[tx.js] inputs keys:", Object.keys(inputs ?? {}));
  // eslint-disable-next-line no-console
  console.warn("[tx.js] input_ids dims:", (inputs as { input_ids?: { dims?: number[] } })?.input_ids?.dims);

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
      // Greedy is meaningfully faster (no sampling loop). Default UI temp=0
      // hits this path. Sampling kicks in only if user sets temperature > 0.
      do_sample: !greedy,
      ...(greedy
        ? {}
        : { temperature: opts.temperature, top_p: opts.topP ?? 0.95 }),
      streamer,
    } as unknown as Parameters<typeof model.generate>[0]);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("[tx.js] generate error:", err);
    throw err;
  }
  return fullText;
}

export async function tokenize(
  bundle: { tokenizer: PreTrainedTokenizer },
  text: string,
): Promise<number[]> {
  // tokenizer.encode returns plain number[] of token ids — unambiguous.
  const ids = bundle.tokenizer.encode(text);
  return Array.from(ids ?? []);
}
