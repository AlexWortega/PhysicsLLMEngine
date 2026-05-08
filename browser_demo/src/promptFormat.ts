import type { FrameSnapshot, SceneHeader } from "./types";

const fmt = (n: number, d: number) => n.toFixed(d);

export function fmtHeader(h: SceneHeader): string {
  const lines: string[] = [`Scene: ${h.description ?? ""}`];
  const g = h.gravity ?? { x: 0, y: 0 };
  lines.push(`Gravity: (${g.x ?? 0}, ${g.y ?? 0})`);
  lines.push(`Timestep: ${fmt(h.timestep ?? 0.01667, 5)}`);
  if (h.scenario_type) lines.push(`Type: ${h.scenario_type}`);
  if (h.difficulty != null) lines.push(`Difficulty: ${h.difficulty}`);
  const stat = h.static_geometry ?? [];
  if (stat.length > 0) {
    const parts: string[] = [];
    for (const sg of stat) {
      if (sg.type === "segment") {
        parts.push(
          `seg (${Math.round(sg.p1.x)},${Math.round(sg.p1.y)})-(${Math.round(sg.p2.x)},${Math.round(sg.p2.y)})`,
        );
      } else if (sg.type === "circle") {
        parts.push(
          `peg (${Math.round(sg.center.x)},${Math.round(sg.center.y)}) r=${Math.round(sg.radius)}`,
        );
      }
    }
    if (parts.length > 0) lines.push(`Static: ${parts.join("; ")}`);
  }
  const constr = h.constraints ?? [];
  if (constr.length > 0) {
    const parts = constr.map((c) => `${c.type} ${c.body_a}->${c.body_b}`);
    lines.push(`Constraints: ${parts.join("; ")}`);
  }
  lines.push("");
  return lines.join("\n");
}

export function fmtFrame(fr: FrameSnapshot): string {
  const lines: string[] = [`Frame ${fr.frame}: ${fr.description ?? ""}`];
  for (const o of fr.objects) {
    const p = o.position;
    const v = o.velocity ?? { x: 0, y: 0 };
    const a = o.angle ?? 0;
    const av = o.angular_velocity ?? 0;
    let s = `  obj_${o.id}: pos=(${fmt(p.x, 4)}, ${fmt(p.y, 4)}), vel=(${fmt(v.x, 4)}, ${fmt(v.y, 4)})`;
    if (Math.abs(a) > 0.001 || Math.abs(av) > 0.001) {
      s += `, a=${fmt(a, 4)}, av=${fmt(av, 4)}`;
    }
    lines.push(s);
  }
  lines.push("");
  return lines.join("\n");
}

const OBJ_RE =
  /obj_(\d+):\s*pos=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\),\s*vel=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)(?:,\s*a=(-?[\d.]+),\s*av=(-?[\d.]+))?/g;

export function parseFrame(text: string, nObj: number): FrameSnapshot["objects"] {
  const out: Record<number, FrameSnapshot["objects"][number]> = {};
  for (const m of text.matchAll(OBJ_RE)) {
    const i = Number(m[1]);
    if (i < nObj) {
      out[i] = {
        id: i,
        position: { x: Number(m[2]), y: Number(m[3]) },
        velocity: { x: Number(m[4]), y: Number(m[5]) },
        angle: m[6] ? Number(m[6]) : 0,
        angular_velocity: m[7] ? Number(m[7]) : 0,
      };
    }
  }
  return Object.values(out);
}

export function splitFirstFrame(text: string): string {
  // Frame boundaries in the trained format are single \n (each frame text
  // ends with a trailing empty line, so concatenated frames are separated by
  // one \n, not two). Find the first "Frame N:" and slice up to the second.
  const re = /Frame\s+\d+:/g;
  const matches = [...text.matchAll(re)];
  if (matches.length === 0) return text;
  const firstEnd = (matches[0].index ?? 0) + matches[0][0].length;
  const secondStart = matches.length > 1 ? (matches[1].index ?? text.length) : text.length;
  return text.slice(firstEnd, secondStart);
}

/** Returns the substring containing exactly the first emitted "Frame N: ...\n  obj_…" block,
 *  including the leading "Frame N:" header — for round-tripping back into context. */
export function extractFirstFrameRaw(text: string): string {
  const re = /Frame\s+\d+:/g;
  const matches = [...text.matchAll(re)];
  if (matches.length === 0) return text;
  const start = matches[0].index ?? 0;
  const end = matches.length > 1 ? (matches[1].index ?? text.length) : text.length;
  return text.slice(start, end).replace(/\s+$/, "") + "\n";
}

export function capContext(ctx: string, keep = 3): string {
  // Split header from frames; keep header + last `keep` frames.
  const idx = ctx.indexOf("Frame ");
  if (idx < 0) return ctx;
  const head = ctx.slice(0, idx);
  const body = ctx.slice(idx);
  const chunks = body.split(/(?=Frame \d+:)/).filter(Boolean);
  if (chunks.length <= keep) return ctx;
  return head + chunks.slice(-keep).join("");
}
