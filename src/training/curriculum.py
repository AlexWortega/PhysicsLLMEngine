"""
Curriculum learning module for physics training.

Sorts scenes by complexity (object count) for curriculum-based training.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time


def _read_scene_header(args):
    """Read header from a single JSONL file (for multiprocessing)."""
    jsonl_path, complexity_metric = args
    try:
        with open(jsonl_path, 'r') as f:
            header = json.loads(f.readline())
        if header.get("type") != "scene_header":
            return None
        return {
            "path": str(jsonl_path),
            "complexity": header.get(complexity_metric, 0),
            "seed": header.get("seed", 0),
            "object_count": header.get("object_count", 0),
        }
    except (json.JSONDecodeError, IOError):
        return None


class CurriculumDataset:
    """Dataset that sorts scenes by complexity for curriculum learning."""

    def __init__(self, data_dir: str, complexity_metric: str = "object_count"):
        """
        Initialize curriculum dataset.

        Args:
            data_dir: Directory containing scene JSONL files
            complexity_metric: Field to use for sorting (default: object_count)
        """
        self.data_dir = Path(data_dir)
        self.complexity_metric = complexity_metric
        self.scenes: List[Dict[str, Any]] = []
        self._scan_scenes()

    def _scan_scenes(self) -> None:
        """Scan all scenes and extract complexity metrics using parallel I/O."""
        start = time.time()
        all_files = list(self.data_dir.rglob("*.jsonl"))
        n_files = len(all_files)
        print(f"Scanning {n_files} scene files...", flush=True)

        if n_files > 1000:
            from multiprocessing import Pool
            import os
            n_workers = min(os.cpu_count() or 4, 16)
            args = [(str(f), self.complexity_metric) for f in all_files]
            with Pool(n_workers) as pool:
                results = pool.map(_read_scene_header, args, chunksize=500)
            self.scenes = [r for r in results if r is not None]
        else:
            self.scenes = []
            for jsonl_file in all_files:
                result = _read_scene_header((str(jsonl_file), self.complexity_metric))
                if result is not None:
                    self.scenes.append(result)

        # Sort by complexity ascending
        self.scenes.sort(key=lambda x: x["complexity"])
        elapsed = time.time() - start
        print(f"Scanned {len(self.scenes)} scenes in {elapsed:.1f}s", flush=True)

    def get_curriculum_stages(self, num_stages: int = 5) -> List[List[str]]:
        """
        Split scenes into curriculum stages.

        Stage 1: Simplest scenes (lowest object count)
        Stage N: Most complex scenes (highest object count)

        Args:
            num_stages: Number of curriculum stages

        Returns:
            List of lists, each containing scene paths for that stage
        """
        if not self.scenes:
            return [[] for _ in range(num_stages)]

        stage_size = len(self.scenes) // num_stages
        stages = []

        for i in range(num_stages):
            start = i * stage_size
            if i == num_stages - 1:
                # Last stage gets any remainder
                end = len(self.scenes)
            else:
                end = (i + 1) * stage_size

            stage_paths = [s["path"] for s in self.scenes[start:end]]
            stages.append(stage_paths)

        return stages

    def get_stage_info(self, num_stages: int = 5) -> List[Dict[str, Any]]:
        """
        Get information about each curriculum stage.

        Args:
            num_stages: Number of curriculum stages

        Returns:
            List of dicts with stage info (num_scenes, complexity_range)
        """
        if not self.scenes:
            return [{"stage": i + 1, "num_scenes": 0, "complexity_min": 0, "complexity_max": 0}
                    for i in range(num_stages)]

        stage_size = len(self.scenes) // num_stages
        stage_info = []

        for i in range(num_stages):
            start = i * stage_size
            end = len(self.scenes) if i == num_stages - 1 else (i + 1) * stage_size
            stage_scenes = self.scenes[start:end]

            complexities = [s["complexity"] for s in stage_scenes]
            stage_info.append({
                "stage": i + 1,
                "num_scenes": len(stage_scenes),
                "complexity_min": min(complexities) if complexities else 0,
                "complexity_max": max(complexities) if complexities else 0,
            })

        return stage_info

    def __len__(self) -> int:
        return len(self.scenes)


def get_curriculum_stages(
    data_dir: str,
    num_stages: int = 5,
    complexity_metric: str = "object_count"
) -> List[List[str]]:
    """
    Convenience function to get curriculum stages.

    Args:
        data_dir: Directory containing scene JSONL files
        num_stages: Number of curriculum stages
        complexity_metric: Field to use for sorting

    Returns:
        List of lists, each containing scene paths for that stage
    """
    dataset = CurriculumDataset(data_dir, complexity_metric)
    return dataset.get_curriculum_stages(num_stages)
