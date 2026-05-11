import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, type ExtraObject, type ExtraWall, type Tool } from "./Canvas";
import { simulateStream, fetchScenarios, prepareModel } from "./streamClient";
import { LAST_LOAD_INFO, isMobileDevice } from "./transformersEngine";
import type {
  FrameSnapshot,
  ScenarioBundle,
  SceneHeader,
  Status,
} from "./types";

const CANVAS_W = 720;
const CANVAS_H = 560;
const MOBILE_BREAKPOINT = 900;

function useViewportSize() {
  const [size, setSize] = useState<{ w: number; h: number; mobile: boolean }>(
    () => {
      if (typeof window === "undefined") {
        return { w: CANVAS_W, h: CANVAS_H, mobile: false };
      }
      return {
        w: window.innerWidth,
        h: window.innerHeight,
        mobile: window.innerWidth < MOBILE_BREAKPOINT || isMobileDevice(),
      };
    },
  );
  useEffect(() => {
    function onResize() {
      setSize({
        w: window.innerWidth,
        h: window.innerHeight,
        mobile: window.innerWidth < MOBILE_BREAKPOINT || isMobileDevice(),
      });
    }
    window.addEventListener("resize", onResize);
    window.addEventListener("orientationchange", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("orientationchange", onResize);
    };
  }, []);
  return size;
}

interface LogLine {
  t: number;
  text: string;
}

export function App() {
  const viewport = useViewportSize();
  const isMobile = viewport.mobile;
  // On mobile/narrow viewports the canvas takes the full width minus padding,
  // and height is bounded to a 4:3-ish aspect of the available width so the
  // controls beneath stay visible without scrolling.
  const canvasW = isMobile
    ? Math.max(280, Math.min(viewport.w - 24, 540))
    : CANVAS_W;
  const canvasH = isMobile
    ? Math.max(220, Math.round(canvasW * (CANVAS_H / CANVAS_W)))
    : CANVAS_H;
  const [scenarios, setScenarios] = useState<ScenarioBundle[]>([]);
  const [selectedName, setSelectedName] = useState<string>("");
  const [tool, setTool] = useState<Tool>("view");
  const [mobilePanel, setMobilePanel] = useState<"controls" | "context" | null>(
    null,
  );
  const [extras, setExtras] = useState<{
    objects: ExtraObject[];
    walls: ExtraWall[];
  }>({ objects: [], walls: [] });
  const [nFrames, setNFrames] = useState(20);
  const [temperature, setTemperature] = useState(0);
  const [status, setStatus] = useState<Status>("idle");
  const [statusMsg, setStatusMsg] = useState<string>("Loading scenarios…");
  const [predictedFrames, setPredictedFrames] = useState<FrameSnapshot[]>([]);
  const [playIdx, setPlayIdx] = useState<number | null>(null);
  const [tps, setTps] = useState<number>(0);
  const [elapsed, setElapsed] = useState<number>(0);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [modelReady, setModelReady] = useState(false);
  const [downloadPct, setDownloadPct] = useState<number | null>(null);
  const [liveTokens, setLiveTokens] = useState<number>(0);
  const [livePieces, setLivePieces] = useState<Array<{ id: number; text: string }>>([]);
  const [livePrompt, setLivePrompt] = useState<string>("");
  const [promptStep, setPromptStep] = useState<{ step: number; total: number } | null>(null);
  const [bootStarted, setBootStarted] = useState(false);
  const liveTextRef = useRef<HTMLDivElement | null>(null);
  const promptRef = useRef<HTMLDivElement | null>(null);
  const pieceIdRef = useRef(0);
  const piecesBufferRef = useRef<Array<{ id: number; text: string }>>([]);
  const tokensBufferRef = useRef(0);
  const lastTokenFlushRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  // Fetch scenarios on mount
  useEffect(() => {
    fetchScenarios()
      .then((items) => {
        const bundles = items as unknown as ScenarioBundle[];
        setScenarios(bundles);
        // Pick a light scenario as default — angry_birds (alphabetically first)
        // has ~39 bodies + 36 static blocks and noticeably lags the canvas
        // before any prediction has been generated.
        const lightDefaults = ["billiards", "pong", "pendulum", "seesaw", "ramp_roll"];
        const pick =
          bundles.find((b) => lightDefaults.includes(b.name))?.name ??
          bundles[0]?.name ??
          "";
        if (pick) setSelectedName(pick);
        setStatusMsg(`${bundles.length} scenarios loaded`);
      })
      .catch((e) => {
        setStatus("error");
        setStatusMsg(`Failed to load scenarios: ${String(e)}`);
      });
  }, []);

  const selected = useMemo(
    () => scenarios.find((s) => s.name === selectedName) ?? null,
    [scenarios, selectedName],
  );
  const header: SceneHeader | null = selected?.header ?? null;
  const initialFrames: FrameSnapshot[] = selected?.initial_frames ?? [];

  // Combine initial frames + predicted to drive the playback
  const allFrames = useMemo(
    () => [...initialFrames, ...predictedFrames],
    [initialFrames, predictedFrames],
  );

  // Keep refs up-to-date so the playback interval can read the LATEST counts
  // without restarting on every state change.
  const totalFramesRef = useRef(0);
  useEffect(() => {
    totalFramesRef.current = initialFrames.length + predictedFrames.length;
  }, [initialFrames.length, predictedFrames.length]);

  // Single playback loop — always running once we have ≥1 predicted frame.
  useEffect(() => {
    if (predictedFrames.length === 0) {
      setPlayIdx(initialFrames.length > 0 ? initialFrames.length - 1 : null);
      return;
    }
    let i = 0;
    setPlayIdx(0);
    const id = window.setInterval(() => {
      const total = totalFramesRef.current;
      if (total <= 0) return;
      i = (i + 1) % total;
      setPlayIdx(i);
    }, 1000 / 24);
    return () => window.clearInterval(id);
  }, [predictedFrames.length === 0, initialFrames.length]);

  function pushLog(text: string) {
    setLogs((prev) => [...prev.slice(-300), { t: Date.now(), text }]);
  }

  // Auto-scroll the live token panel to the bottom as text grows.
  useEffect(() => {
    if (liveTextRef.current) {
      liveTextRef.current.scrollTop = liveTextRef.current.scrollHeight;
    }
  }, [livePieces.length]);

  // Pin the prompt panel to the bottom (where the freshly added frame and the
  // "Predict next frame:" suffix live).
  useEffect(() => {
    if (promptRef.current) {
      promptRef.current.scrollTop = promptRef.current.scrollHeight;
    }
  }, [livePrompt]);

  async function runSimulate() {
    if (!header) return;
    setStatus("generating");
    // Keep prior predictedFrames so the canvas keeps animating the previous
    // run while the new generation streams in; new frames are appended.
    setLogs([]);
    setTps(0);
    setElapsed(0);
    setLivePieces([]);

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      // Model boot now happens via the START overlay before Simulate is clickable;
      // this remains as a fallback in case modelReady is somehow false.
      if (!modelReady) {
        await startBoot();
      }
      setStatusMsg("Generating…");
      // Build the request. Speed wins:
      //   1) Use ONLY the first seed frame (the initial-state snapshot) —
      //      the other 3 are nearly-static evolution and just bloat the
      //      prompt by ~3x without giving the model new info. The model
      //      was trained with min_context_frames=1, so 1 frame is in-dist.
      //   2) Patch extras into that single frame so the model sees a
      //      consistent N+E-object world from step 1.
      const mergedHeader = mergeExtras(header, extras);
      const seedFrame = initialFrames[0];
      const patchedInitialFrames: FrameSnapshot[] = seedFrame
        ? [
            {
              ...seedFrame,
              objects: [
                ...seedFrame.objects,
                ...extras.objects.map((o) => ({
                  id: o.id,
                  position: o.position,
                  velocity: o.velocity ?? { x: 0, y: 0 },
                  angle: 0,
                  angular_velocity: 0,
                  type: o.kind === "box" ? "rectangle" : "circle",
                  radius: o.radius,
                  width: o.width,
                  height: o.height,
                })),
              ],
            },
          ]
        : initialFrames;
      for await (const ev of simulateStream(
        {
          header: mergedHeader,
          initial_frames: patchedInitialFrames,
          n_frames: nFrames,
          temperature,
        },
        ctrl.signal,
        {
          onTokenProgress: ({ tokens, piece }) => {
            // Throttle React updates: only push to UI ~10 times per second
            // regardless of token rate. Avoids re-rendering the whole right
            // panel on every single token (which lags the canvas badly).
            tokensBufferRef.current += 1;
            piecesBufferRef.current.push({
              id: ++pieceIdRef.current,
              text: piece ?? "",
            });
            const now = performance.now();
            if (now - lastTokenFlushRef.current > 100) {
              lastTokenFlushRef.current = now;
              setLiveTokens(tokens);
              const flush = piecesBufferRef.current.splice(0);
              setLivePieces((prev) => {
                const next = [...prev, ...flush];
                // Cap to last 120 pieces to keep the DOM cheap.
                return next.length > 120 ? next.slice(-120) : next;
              });
            }
          },
          onLog: (line) => pushLog(line),
          onPrompt: ({ step, total, prompt }) => {
            // Only show the LAST 2KB of the prompt — the model's actual
            // attention window context. Avoids 5KB+ React diffing per step.
            const tail = prompt.length > 2000 ? prompt.slice(-2000) : prompt;
            setLivePrompt(tail);
            setPromptStep({ step, total });
          },
        },
      )) {
        if (ev.type === "status") {
          setStatusMsg(ev.msg);
          pushLog(ev.msg);
        } else if (ev.type === "ready") {
          setStatusMsg(`Generating ${ev.n_frames} frames · ${ev.n_objects} objects`);
          pushLog(`ready · ${ev.n_objects} objects`);
        } else if (ev.type === "frame") {
          setPredictedFrames((prev) => [
            ...prev,
            { frame: ev.frame_idx, objects: ev.objects },
          ]);
          setTps(ev.tps);
          setElapsed(ev.elapsed);
          setLiveTokens(0);
          setLivePieces([]);
          setStatusMsg(`frame ${ev.step}/${ev.total} · ${ev.tps.toFixed(1)} t/s`);
          pushLog(
            `frame ${ev.step}/${ev.total} · ${ev.objects.length} objs · ${ev.tps.toFixed(1)}t/s`,
          );
        } else if (ev.type === "done") {
          setStatusMsg(`Done in ${ev.elapsed.toFixed(1)}s`);
          pushLog(`done · ${ev.elapsed.toFixed(1)}s`);
          setStatus("ready");
        } else if (ev.type === "error") {
          setStatus("error");
          setStatusMsg(`error: ${ev.message}`);
        }
      }
      setStatus("ready");
    } catch (err: unknown) {
      if ((err as DOMException)?.name === "AbortError") {
        setStatusMsg("Cancelled");
      } else {
        setStatus("error");
        setStatusMsg(`Error: ${String(err)}`);
      }
    } finally {
      abortRef.current = null;
    }
  }

  function cancel() {
    abortRef.current?.abort();
    setStatus("ready");
    setStatusMsg("Cancelled");
  }

  function clearExtras() {
    setExtras({ objects: [], walls: [] });
  }

  async function startBoot() {
    if (bootStarted) return;
    setBootStarted(true);
    setStatusMsg("Downloading LFM2-350M (≈460 MB q4)…");
    setDownloadPct(0);
    try {
      await prepareModel((pct) => {
        setDownloadPct(pct);
        setStatusMsg(`Downloading model · ${pct}%`);
      });
      setDownloadPct(null);
      setModelReady(true);
      setStatusMsg("Model ready · pick a scenario and Simulate");
    } catch (err) {
      setStatus("error");
      setStatusMsg(`Model load failed: ${String(err)}`);
      setBootStarted(false);
    }
  }

  const liveFrame: FrameSnapshot | null =
    playIdx !== null && allFrames[playIdx] ? allFrames[playIdx] : null;
  const trailFrames = useMemo(
    () => (playIdx !== null ? allFrames.slice(0, playIdx) : allFrames),
    [allFrames, playIdx],
  );

  const tools: Array<{ id: Tool; label: string; icon: string }> = [
    { id: "view", label: "View", icon: "👁" },
    { id: "circle", label: "Circle", icon: "●" },
    { id: "box", label: "Box", icon: "■" },
    { id: "wall", label: "Wall", icon: "╱" },
    { id: "delete", label: "Delete", icon: "✕" },
  ];

  return (
    <div className={`app ${isMobile ? "is-mobile" : ""}`}>
      <header className="topbar">
        <span className="title">Physics LLM</span>
        <span
          className={`badge ${LAST_LOAD_INFO?.device === "webgpu" ? "good" : "warn"}`}
          title={
            LAST_LOAD_INFO
              ? `device=${LAST_LOAD_INFO.device} · dtype=${LAST_LOAD_INFO.dtype} · webgpu=${LAST_LOAD_INFO.webgpuAvailable}` +
                (LAST_LOAD_INFO.attempts && LAST_LOAD_INFO.attempts.length > 1
                  ? `\n\nAttempts:\n${LAST_LOAD_INFO.attempts
                      .map(
                        (a) =>
                          `· ${a.device}+${a.dtype}: ${a.error ? "FAIL — " + a.error : "OK"}`,
                      )
                      .join("\n")}`
                  : "")
              : "model not loaded yet"
          }
        >
          LFM2-350M ·{" "}
          {LAST_LOAD_INFO
            ? `${LAST_LOAD_INFO.device.toUpperCase()} · ${LAST_LOAD_INFO.dtype}`
            : "loading…"}
        </span>
        {LAST_LOAD_INFO && !LAST_LOAD_INFO.webgpuAvailable && !isMobile ? (
          <span
            className="warn-banner"
            title="WebGPU not available — falling back to WASM (slower). Use Chrome/Edge or enable chrome://flags/#enable-unsafe-webgpu."
          >
            ⚠ WASM fallback (no WebGPU)
          </span>
        ) : null}
        <div className="spacer" />
        {isMobile ? (
          <button
            className="mobile-drawer-btn"
            onClick={() =>
              setMobilePanel(mobilePanel === "controls" ? null : "controls")
            }
            aria-label="Open controls"
          >
            ☰
          </button>
        ) : null}
        {downloadPct !== null || (status === "generating" && !modelReady) ? (
          <div
            className="download-bar"
            title={`Downloading model · ${downloadPct ?? 0}%`}
          >
            <div className="fill" style={{ width: `${downloadPct ?? 0}%` }} />
            <span>model {downloadPct ?? 0}%</span>
          </div>
        ) : status === "generating" && modelReady ? (
          <div
            className="download-bar"
            title={`Generating ${predictedFrames.length}/${nFrames}`}
          >
            <div
              className="fill gen"
              style={{
                width: `${Math.max(
                  4,
                  Math.round((predictedFrames.length / Math.max(nFrames, 1)) * 100),
                )}%`,
              }}
            />
            <span>
              gen {predictedFrames.length}/{nFrames}
              {liveTokens > 0 ? ` · +${liveTokens}t` : ""}
              {tps > 0 ? ` · ${tps.toFixed(1)} t/s` : ""}
            </span>
          </div>
        ) : null}
        <span className={`status ${status}`}>{statusMsg}</span>
      </header>

      <aside
        className={`sidebar${isMobile ? (mobilePanel === "controls" ? " mobile-open" : " mobile-hidden") : ""}`}
      >
        {isMobile ? (
          <button
            className="drawer-close"
            onClick={() => setMobilePanel(null)}
            aria-label="Close panel"
          >
            ✕
          </button>
        ) : null}
        <section>
          <h3>Scenario</h3>
          <select
            value={selectedName}
            onChange={(e) => {
              setSelectedName(e.target.value);
              setPredictedFrames([]);
            }}
          >
            {scenarios.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
          {header ? (
            <p className="scene-hint">{(header.description ?? "").slice(0, 120)}</p>
          ) : null}
        </section>

        <section>
          <h3>Tool</h3>
          <div className="tool-group">
            {tools.map((t) => (
              <button
                key={t.id}
                className={`tool-btn ${tool === t.id ? "active" : ""}`}
                onClick={() => setTool(t.id)}
              >
                <span className="icon">{t.icon}</span>
                {t.label}
              </button>
            ))}
          </div>
          <p className="scene-hint">
            {tool === "view"
              ? "Pick a tool to add objects to the scene before simulating."
              : tool === "circle" || tool === "box"
              ? "Click + drag — drag length sets initial velocity."
              : tool === "wall"
              ? "Drag to draw a static segment."
              : "Click an extra object to remove it."}
          </p>
          <button className="secondary" onClick={clearExtras}>
            Clear extras ({extras.objects.length + extras.walls.length})
          </button>
        </section>

        <section>
          <h3>Generation</h3>
          <label className="field">
            <span>Frames ({nFrames})</span>
            <input
              type="range"
              min={4}
              max={50}
              value={nFrames}
              onChange={(e) => setNFrames(Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>Temperature ({temperature.toFixed(2)})</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
            />
          </label>
          {status === "generating" ? (
            <button className="primary danger" onClick={cancel}>
              Cancel
            </button>
          ) : (
            <button
              className="primary"
              onClick={runSimulate}
              disabled={!header || status === "loading"}
            >
              ▶ Simulate
            </button>
          )}
        </section>

        <section>
          <h3>Stats</h3>
          <div className="stats">
            <div>
              <span>tok/s</span>
              <strong>{tps.toFixed(1)}</strong>
            </div>
            <div>
              <span>elapsed</span>
              <strong>{elapsed.toFixed(1)}s</strong>
            </div>
            <div>
              <span>frames</span>
              <strong>{predictedFrames.length}</strong>
            </div>
          </div>
        </section>
      </aside>

      <main className="canvas-wrap">
        <div className="canvas-card" style={{ width: canvasW, height: canvasH }}>
          <Canvas
            width={canvasW}
            height={canvasH}
            header={header}
            liveFrame={liveFrame}
            extras={extras}
            setExtras={setExtras}
            tool={tool}
            trail={trailFrames}
          />
          {playIdx !== null && allFrames.length > 1 ? (
            <div className="playback-hud">
              <span className="hud-dot" />
              frame {playIdx + 1} / {allFrames.length}
              {playIdx >= initialFrames.length ? (
                <span className="hud-tag">predicted</span>
              ) : (
                <span className="hud-tag seed">seed</span>
              )}
              {status === "generating" ? (
                <span className="hud-tag gen">generating…</span>
              ) : null}
            </div>
          ) : null}
          {!modelReady ? (
            <div className="boot-overlay">
              <div className="boot-card">
                <div className="boot-title">Physics LLM</div>
                <div className="boot-sub">
                  LFM2-350M fine-tuned on 2D rigid-body physics. Runs entirely in
                  your browser via{" "}
                  <strong>{isMobile ? "WASM" : "WebGPU"}</strong>
                  {isMobile ? " (mobile fallback)" : " (with WASM fallback)"}.
                </div>
                {isMobile ? (
                  <div className="boot-note" style={{ color: "var(--warn)" }}>
                    ⚠ On phones the model runs on CPU/WASM and may take a few
                    minutes per frame. Try a desktop with WebGPU for full speed.
                  </div>
                ) : null}
                {bootStarted && downloadPct !== null ? (
                  <>
                    <div className="boot-bar">
                      <div className="boot-fill" style={{ width: `${downloadPct}%` }} />
                    </div>
                    <div className="boot-progress">
                      Downloading model · {downloadPct}%
                    </div>
                  </>
                ) : bootStarted ? (
                  <div className="boot-progress">Initialising…</div>
                ) : (
                  <button className="boot-button" onClick={startBoot}>
                    ▶ START · download model (≈460 MB)
                  </button>
                )}
                <div className="boot-note">
                  First run downloads ≈460 MB. Cached afterwards.
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </main>

      <aside
        className={`right-panel${isMobile ? (mobilePanel === "context" ? " mobile-open" : " mobile-hidden") : ""}`}
      >
        {isMobile ? (
          <button
            className="drawer-close"
            onClick={() => setMobilePanel(null)}
            aria-label="Close panel"
          >
            ✕
          </button>
        ) : null}
        <h3>
          Context
          {promptStep ? (
            <span className="muted-suffix">
              {" "}· step {promptStep.step}/{promptStep.total} · {livePrompt.length} chars
            </span>
          ) : null}
        </h3>
        <div className="prompt-stream" ref={promptRef}>
          {livePrompt ? (
            <ContextHighlighted prompt={livePrompt} />
          ) : (
            <span className="ts-empty">
              The full prompt fed to the model will appear here as it is built each step.
            </span>
          )}
        </div>
        <h3>
          Tokens (live){liveTokens > 0 ? <span className="muted-suffix"> · {liveTokens}</span> : null}
        </h3>
        <div className="token-stream" ref={liveTextRef}>
          {livePieces.length === 0 ? (
            <span className="ts-empty">
              {status === "generating"
                ? "…waiting for first token"
                : "Tokens will stream here while the model writes a frame."}
            </span>
          ) : (
            <>
              {livePieces.map((p) => (
                <span key={p.id} className="tok-fresh">
                  {p.text}
                </span>
              ))}
              {status === "generating" ? <span className="caret">▍</span> : null}
            </>
          )}
        </div>
        <h3>Log</h3>
        <div className="log">
          {logs.length === 0
            ? "Pick a scenario and hit Simulate."
            : logs.map((l, i) => (
                <div key={i}>
                  <span className="t">{new Date(l.t).toLocaleTimeString()}</span>
                  <span>{l.text}</span>
                </div>
              ))}
        </div>
        <p className="scene-hint">
          Model:{" "}
          <a
            href="https://huggingface.co/AlexWortega/lfm2-scenarios-GGUF"
            target="_blank"
            rel="noreferrer"
          >
            lfm2-scenarios-GGUF
          </a>
          <br />
          Dataset:{" "}
          <a
            href="https://huggingface.co/datasets/AlexWortega/physics-scenarios-packed"
            target="_blank"
            rel="noreferrer"
          >
            physics-scenarios-packed
          </a>
        </p>
      </aside>

      {isMobile && modelReady ? (
        <div className="mobile-actionbar">
          <button
            className="mobile-action"
            onClick={() =>
              setMobilePanel(mobilePanel === "controls" ? null : "controls")
            }
          >
            ⚙ Controls
          </button>
          {status === "generating" ? (
            <button className="mobile-action danger" onClick={cancel}>
              ■ Cancel
            </button>
          ) : (
            <button
              className="mobile-action primary"
              onClick={runSimulate}
              disabled={!header}
            >
              ▶ Simulate
            </button>
          )}
          <button
            className="mobile-action"
            onClick={() =>
              setMobilePanel(mobilePanel === "context" ? null : "context")
            }
          >
            ☷ Context
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** Render the prompt with semantic highlighting:
 *  - header (everything before the first "Frame N:")
 *  - each frame block (alternating muted / brighter)
 *  - last frame (highlighted as the "freshest" context)
 *  - "Predict next frame:" suffix (bold accent)
 */
const ContextHighlighted = memo(function ContextHighlighted({
  prompt,
}: {
  prompt: string;
}) {
  // Strip any trailing "Predict next frame:" suffix and render it separately.
  const SUFFIX = "Predict next frame:";
  let body = prompt;
  let suffix = "";
  if (body.endsWith(SUFFIX)) {
    body = body.slice(0, body.length - SUFFIX.length);
    suffix = SUFFIX;
  }
  // Show only the last ~5000 chars to keep the DOM cheap on long rollouts.
  const MAX = 5000;
  let truncated = false;
  if (body.length > MAX) {
    body = body.slice(-MAX);
    truncated = true;
  }
  // Find frame boundaries
  const matches = [...body.matchAll(/Frame\s+\d+:/g)];
  const segments: Array<{ kind: "head" | "frame"; text: string; idx: number }> = [];
  if (matches.length === 0) {
    segments.push({ kind: "head", text: body, idx: 0 });
  } else {
    const first = matches[0].index ?? 0;
    if (first > 0) segments.push({ kind: "head", text: body.slice(0, first), idx: 0 });
    for (let i = 0; i < matches.length; i++) {
      const start = matches[i].index ?? 0;
      const end = i + 1 < matches.length ? matches[i + 1].index ?? body.length : body.length;
      segments.push({ kind: "frame", text: body.slice(start, end), idx: i });
    }
  }
  const lastFrameIdx = segments.length - 1;
  return (
    <>
      {truncated ? (
        <span className="ctx-trunc">…(truncated head)…{"\n"}</span>
      ) : null}
      {segments.map((seg, i) => {
        if (seg.kind === "head") {
          return <span key={i} className="ctx-head">{seg.text}</span>;
        }
        const isLast = i === lastFrameIdx;
        const cls = isLast ? "ctx-frame ctx-frame-last" : i % 2 === 0 ? "ctx-frame ctx-frame-a" : "ctx-frame ctx-frame-b";
        return <span key={i} className={cls}>{seg.text}</span>;
      })}
      {suffix ? <span className="ctx-suffix">{suffix}</span> : null}
    </>
  );
});

function mergeExtras(
  header: SceneHeader,
  extras: { objects: ExtraObject[]; walls: ExtraWall[] },
): SceneHeader {
  if (extras.objects.length === 0 && extras.walls.length === 0) return header;
  const next: SceneHeader = JSON.parse(JSON.stringify(header));
  for (const o of extras.objects) {
    next.objects.push({
      id: o.id,
      type: o.kind === "box" ? "rectangle" : "circle",
      position: o.position,
      velocity: o.velocity,
      radius: o.radius,
      width: o.width,
      height: o.height,
      material: { mass: o.mass, friction: 0.5, elasticity: 0.5 },
    });
  }
  next.object_count = next.objects.length;
  next.static_geometry = next.static_geometry ?? [];
  for (const w of extras.walls) {
    next.static_geometry.push({ type: "segment", p1: w.p1, p2: w.p2 });
  }
  return next;
}
