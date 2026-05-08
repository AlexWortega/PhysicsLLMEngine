import { useMemo, useRef, useState } from "react";
import { Stage, Layer, Circle, Rect, Line, Group, Text } from "react-konva";
import type { KonvaEventObject } from "konva/lib/Node";
import type { SceneHeader, FrameSnapshot } from "./types";

export type Tool = "view" | "circle" | "box" | "wall" | "delete";

interface ExtraObject {
  id: number;
  kind: "circle" | "box";
  position: { x: number; y: number };
  velocity: { x: number; y: number };
  radius?: number;
  width?: number;
  height?: number;
  mass: number;
}

interface ExtraWall {
  id: number;
  p1: { x: number; y: number };
  p2: { x: number; y: number };
}

interface Props {
  width: number;
  height: number;
  header: SceneHeader | null;
  liveFrame: FrameSnapshot | null;
  extras: { objects: ExtraObject[]; walls: ExtraWall[] };
  setExtras: (
    next:
      | { objects: ExtraObject[]; walls: ExtraWall[] }
      | ((prev: { objects: ExtraObject[]; walls: ExtraWall[] }) => {
          objects: ExtraObject[];
          walls: ExtraWall[];
        }),
  ) => void;
  tool: Tool;
  trail: FrameSnapshot[];
}

const PALETTE = [
  "#5eead4",
  "#8b5cf6",
  "#f59e0b",
  "#ef4444",
  "#3b82f6",
  "#22c55e",
  "#ec4899",
  "#fbbf24",
  "#06b6d4",
  "#a78bfa",
];

export type { ExtraObject, ExtraWall };

export function Canvas({
  width,
  height,
  header,
  liveFrame,
  extras,
  setExtras,
  tool,
  trail,
}: Props) {
  const [draftStart, setDraftStart] = useState<{ x: number; y: number } | null>(
    null,
  );
  const [draftEnd, setDraftEnd] = useState<{ x: number; y: number } | null>(null);
  const stageRef = useRef<any>(null);

  // Compute world bounds from header objects + static geometry, then build a
  // transform from world coords to canvas pixels (Y is flipped because
  // Pymunk uses bottom-up while Konva uses top-down).
  const transform = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    if (header) {
      for (const o of header.objects) {
        xs.push(o.position.x);
        ys.push(o.position.y);
      }
      for (const sg of header.static_geometry ?? []) {
        if (sg.type === "segment") {
          xs.push(sg.p1.x, sg.p2.x);
          ys.push(sg.p1.y, sg.p2.y);
        } else if (sg.type === "circle") {
          xs.push(sg.center.x);
          ys.push(sg.center.y);
        }
      }
    }
    for (const fr of trail) {
      for (const o of fr.objects) {
        xs.push(o.position.x);
        ys.push(o.position.y);
      }
    }
    if (xs.length === 0) {
      xs.push(0, 800);
      ys.push(0, 600);
    }
    const pad = 60;
    const xmin = Math.min(...xs) - pad;
    const xmax = Math.max(...xs) + pad;
    const ymin = Math.min(...ys) - pad;
    const ymax = Math.max(...ys) + pad;
    const span = Math.max(xmax - xmin, ymax - ymin, 200);
    const cx = (xmin + xmax) / 2;
    const cy = (ymin + ymax) / 2;
    const scale = Math.min(width, height) / span;
    const ox = width / 2 - cx * scale;
    const oy = height / 2 + cy * scale; // Y flipped
    return {
      toX: (x: number) => x * scale + ox,
      toY: (y: number) => -y * scale + oy,
      scale,
      span,
    };
  }, [header, trail, width, height]);

  function pointer(e: KonvaEventObject<MouseEvent | TouchEvent>) {
    const stage = e.target.getStage();
    if (!stage) return null;
    const p = stage.getPointerPosition();
    return p ? { x: p.x, y: p.y } : null;
  }

  function pixelToWorld(px: { x: number; y: number }) {
    const { scale } = transform;
    return {
      x: (px.x - (transform.toX(0))) / scale,
      y: -(px.y - transform.toY(0)) / scale,
    };
  }

  const nextId = useMemo(() => {
    return (
      Math.max(
        0,
        ...extras.objects.map((o) => o.id),
        ...(header?.objects.map((o) => o.id) ?? []),
      ) + 1
    );
  }, [extras, header]);

  function handleDown(e: KonvaEventObject<MouseEvent | TouchEvent>) {
    if (tool === "view") return;
    const p = pointer(e);
    if (!p) return;
    setDraftStart(p);
    setDraftEnd(p);
    if (tool === "delete") {
      const w = pixelToWorld(p);
      setExtras((prev) => ({
        ...prev,
        objects: prev.objects.filter(
          (o) =>
            (o.position.x - w.x) ** 2 + (o.position.y - w.y) ** 2 >
            ((o.radius ?? Math.max(o.width ?? 0, o.height ?? 0) / 2) +
              5) **
              2,
        ),
      }));
    }
  }

  function handleMove(e: KonvaEventObject<MouseEvent | TouchEvent>) {
    if (!draftStart) return;
    const p = pointer(e);
    if (p) setDraftEnd(p);
  }

  function handleUp(e: KonvaEventObject<MouseEvent | TouchEvent>) {
    if (!draftStart) return;
    const p = pointer(e);
    if (!p) {
      setDraftStart(null);
      setDraftEnd(null);
      return;
    }
    const startWorld = pixelToWorld(draftStart);
    const endWorld = pixelToWorld(p);
    if (tool === "circle" || tool === "box") {
      const obj: ExtraObject = {
        id: nextId,
        kind: tool,
        position: startWorld,
        velocity: {
          x: (endWorld.x - startWorld.x) * 4,
          y: (endWorld.y - startWorld.y) * 4,
        },
        radius: tool === "circle" ? 18 : undefined,
        width: tool === "box" ? 36 : undefined,
        height: tool === "box" ? 36 : undefined,
        mass: 1.0,
      };
      setExtras((prev) => ({ ...prev, objects: [...prev.objects, obj] }));
    } else if (tool === "wall") {
      const dx = endWorld.x - startWorld.x;
      const dy = endWorld.y - startWorld.y;
      if (dx * dx + dy * dy > 64) {
        setExtras((prev) => ({
          ...prev,
          walls: [...prev.walls, { id: nextId, p1: startWorld, p2: endWorld }],
        }));
      }
    }
    setDraftStart(null);
    setDraftEnd(null);
  }

  // Build draw artifacts from current state.
  const liveById = useMemo(() => {
    const m = new Map<number, FrameSnapshot["objects"][number]>();
    if (liveFrame) for (const o of liveFrame.objects) m.set(o.id, o);
    return m;
  }, [liveFrame]);

  return (
    <Stage
      ref={stageRef}
      width={width}
      height={height}
      onMouseDown={handleDown}
      onMouseMove={handleMove}
      onMouseUp={handleUp}
      onTouchStart={handleDown}
      onTouchMove={handleMove}
      onTouchEnd={handleUp}
    >
      {/* grid */}
      <Layer listening={false}>
        {Array.from({ length: 32 }, (_, i) => (
          <Line
            key={`v-${i}`}
            points={[i * (width / 32), 0, i * (width / 32), height]}
            stroke="#1a1f2e"
            strokeWidth={1}
          />
        ))}
        {Array.from({ length: 24 }, (_, i) => (
          <Line
            key={`h-${i}`}
            points={[0, i * (height / 24), width, i * (height / 24)]}
            stroke="#1a1f2e"
            strokeWidth={1}
          />
        ))}
      </Layer>

      {/* static geometry */}
      <Layer listening={false}>
        {header?.static_geometry?.map((sg, i) => {
          if (sg.type === "segment") {
            return (
              <Line
                key={`sg-${i}`}
                points={[
                  transform.toX(sg.p1.x),
                  transform.toY(sg.p1.y),
                  transform.toX(sg.p2.x),
                  transform.toY(sg.p2.y),
                ]}
                stroke="#94a3b8"
                strokeWidth={3}
                lineCap="round"
              />
            );
          }
          if (sg.type === "circle") {
            return (
              <Circle
                key={`sg-${i}`}
                x={transform.toX(sg.center.x)}
                y={transform.toY(sg.center.y)}
                radius={sg.radius * transform.scale}
                fill="#475569"
                stroke="#94a3b8"
              />
            );
          }
          return null;
        })}
        {extras.walls.map((w) => (
          <Line
            key={`ew-${w.id}`}
            points={[
              transform.toX(w.p1.x),
              transform.toY(w.p1.y),
              transform.toX(w.p2.x),
              transform.toY(w.p2.y),
            ]}
            stroke="#fbbf24"
            strokeWidth={3}
            dash={[6, 4]}
            lineCap="round"
          />
        ))}
      </Layer>

      {/* trail */}
      <Layer listening={false}>
        {trail.slice(-10).map((fr, i) =>
          fr.objects.map((o) => {
            const a = (i + 1) / 12;
            return (
              <Circle
                key={`t-${i}-${o.id}`}
                x={transform.toX(o.position.x)}
                y={transform.toY(o.position.y)}
                radius={3}
                fill={PALETTE[o.id % PALETTE.length]}
                opacity={a * 0.45}
              />
            );
          }),
        )}
      </Layer>

      {/* objects (header + extras), at live position when available */}
      <Layer>
        {header?.objects.map((meta) => {
          const live = liveById.get(meta.id);
          const x = (live ?? meta).position.x;
          const y = (live ?? meta).position.y;
          const angle = (live?.angle ?? 0) * (180 / Math.PI);
          const color = PALETTE[meta.id % PALETTE.length];
          if (meta.type === "circle") {
            return (
              <Circle
                key={`o-${meta.id}`}
                x={transform.toX(x)}
                y={transform.toY(y)}
                radius={(meta.radius ?? 12) * transform.scale}
                fill={color}
                stroke="#0b0d12"
                strokeWidth={1.5}
                rotation={angle}
              />
            );
          }
          const w = (meta.width ?? 30) * transform.scale;
          const h = (meta.height ?? 30) * transform.scale;
          return (
            <Group
              key={`o-${meta.id}`}
              x={transform.toX(x)}
              y={transform.toY(y)}
              rotation={angle}
            >
              <Rect
                x={-w / 2}
                y={-h / 2}
                width={w}
                height={h}
                fill={color}
                stroke="#0b0d12"
                strokeWidth={1.5}
                cornerRadius={3}
              />
            </Group>
          );
        })}
        {extras.objects.map((o) => {
          const color = PALETTE[o.id % PALETTE.length];
          if (o.kind === "circle") {
            return (
              <Group key={`e-${o.id}`}>
                <Circle
                  x={transform.toX(o.position.x)}
                  y={transform.toY(o.position.y)}
                  radius={(o.radius ?? 18) * transform.scale}
                  fill={color}
                  stroke="#0b0d12"
                  strokeWidth={1.5}
                  dash={[4, 4]}
                />
                <Line
                  points={[
                    transform.toX(o.position.x),
                    transform.toY(o.position.y),
                    transform.toX(o.position.x + o.velocity.x / 4),
                    transform.toY(o.position.y + o.velocity.y / 4),
                  ]}
                  stroke="#fbbf24"
                  strokeWidth={2}
                />
              </Group>
            );
          }
          const w = (o.width ?? 36) * transform.scale;
          const h = (o.height ?? 36) * transform.scale;
          return (
            <Group key={`e-${o.id}`}>
              <Rect
                x={transform.toX(o.position.x) - w / 2}
                y={transform.toY(o.position.y) - h / 2}
                width={w}
                height={h}
                fill={color}
                stroke="#0b0d12"
                strokeWidth={1.5}
                dash={[4, 4]}
                cornerRadius={3}
              />
              <Line
                points={[
                  transform.toX(o.position.x),
                  transform.toY(o.position.y),
                  transform.toX(o.position.x + o.velocity.x / 4),
                  transform.toY(o.position.y + o.velocity.y / 4),
                ]}
                stroke="#fbbf24"
                strokeWidth={2}
              />
            </Group>
          );
        })}
      </Layer>

      {/* draft preview */}
      <Layer listening={false}>
        {draftStart && draftEnd && (tool === "circle" || tool === "box") ? (
          <Group>
            {tool === "circle" ? (
              <Circle
                x={draftStart.x}
                y={draftStart.y}
                radius={18 * transform.scale}
                stroke="#5eead4"
                dash={[3, 3]}
                strokeWidth={2}
              />
            ) : (
              <Rect
                x={draftStart.x - 18 * transform.scale}
                y={draftStart.y - 18 * transform.scale}
                width={36 * transform.scale}
                height={36 * transform.scale}
                stroke="#5eead4"
                dash={[3, 3]}
                strokeWidth={2}
              />
            )}
            <Line
              points={[draftStart.x, draftStart.y, draftEnd.x, draftEnd.y]}
              stroke="#fbbf24"
              strokeWidth={2}
            />
          </Group>
        ) : null}
        {draftStart && draftEnd && tool === "wall" ? (
          <Line
            points={[draftStart.x, draftStart.y, draftEnd.x, draftEnd.y]}
            stroke="#fbbf24"
            strokeWidth={3}
            dash={[4, 4]}
          />
        ) : null}
      </Layer>

      {/* footer text */}
      {header ? (
        <Layer listening={false}>
          <Text
            x={10}
            y={height - 22}
            text={(header.description ?? "").slice(0, 80)}
            fontSize={12}
            fill="#64748b"
          />
        </Layer>
      ) : null}
    </Stage>
  );
}
