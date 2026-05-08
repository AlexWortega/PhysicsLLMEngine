export interface ObjectMeta {
  id: number;
  type: "circle" | "rectangle" | "box";
  position: { x: number; y: number };
  velocity?: { x: number; y: number };
  radius?: number;
  width?: number;
  height?: number;
  material?: { mass: number; friction: number; elasticity: number };
}

export interface SceneHeader {
  type: "scene_header";
  seed?: number;
  description?: string;
  scenario_type?: string;
  difficulty?: number;
  object_count?: number;
  gravity: { x: number; y: number };
  timestep: number;
  objects: ObjectMeta[];
  static_geometry?: Array<
    | {
        type: "segment";
        p1: { x: number; y: number };
        p2: { x: number; y: number };
      }
    | { type: "circle"; center: { x: number; y: number }; radius: number }
  >;
  constraints?: Array<{ type: string; body_a: number; body_b: number }>;
}

export interface FrameSnapshot {
  frame: number;
  description?: string;
  objects: Array<{
    id: number;
    position: { x: number; y: number };
    velocity?: { x: number; y: number };
    angle?: number;
    angular_velocity?: number;
    type?: string;
    width?: number;
    height?: number;
    radius?: number;
  }>;
}

export interface ScenarioBundle {
  name: string;
  header: SceneHeader;
  initial_frames: FrameSnapshot[];
}

export type SimulateEvent =
  | { type: "status"; msg: string }
  | { type: "ready"; n_objects: number; n_frames: number }
  | {
      type: "frame";
      step: number;
      total: number;
      frame_idx: number;
      objects: FrameSnapshot["objects"];
      elapsed: number;
      tps: number;
      raw: string;
    }
  | { type: "done"; elapsed: number }
  | { type: "error"; message: string }
  | { type: "progress"; step: number; total: number; tokens: number };

export type Status = "idle" | "loading" | "ready" | "generating" | "error";
