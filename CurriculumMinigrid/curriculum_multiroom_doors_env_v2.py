"""
Procedural MiniGrid room-chain benchmark with two doors per room.

Design goals:
- 5x5 rooms (3x3 internal space)
- no corridors/tunnels between rooms
- each stage room has exactly two doors:
  - one target door defined by a family codebook over wall faces
  - one distractor door sampled procedurally per reset
- target door colors are configurable per family and per stage
- distractor door colors are sampled reproducibly from the env seed
- deeper tasks are formed by extending the target-room chain

Wall-face encoding for target door sequences:
- `N`: north wall center
- `E`: east wall center
- `S`: south wall center
- `W`: west wall center

Example:
- family direction sequence `EENN` means the target path expands east, east,
  north, north across the four-stage family.
- family color sequence `yellow` means every target door in the family is yellow.
- family color sequence `(yellow, red, blue, green)` would vary target door
  colors by stage.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gym.envs.registration import register

from gym_minigrid.minigrid import COLOR_NAMES, Door, Goal, Grid, MiniGridEnv, MissionSpace, Wall


@dataclass(frozen=True)
class StagePlan:
    stage_idx: int
    room_cell: Tuple[int, int]
    next_cell: Tuple[int, int]
    distractor_cell: Tuple[int, int]
    target_dir: str
    distractor_dir: str
    target_color: str
    distractor_color: str


class CurriculumMultiRoomDoorsEnvV2(MiniGridEnv):
    ROOM_SIZE = 5
    ROOM_STEP = ROOM_SIZE - 1
    ROOM_CENTER = 2
    PADDING = 2

    DIRECTIONS: Tuple[str, ...] = ("N", "E", "S", "W")
    DIR_TO_DELTA: Dict[str, Tuple[int, int]] = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0),
    }
    DIR_TO_DOOR_OFFSET: Dict[str, Tuple[int, int]] = {
        "N": (ROOM_CENTER, 0),
        "E": (ROOM_SIZE - 1, ROOM_CENTER),
        "S": (ROOM_CENTER, ROOM_SIZE - 1),
        "W": (0, ROOM_CENTER),
    }

    DEFAULT_DIRECTION_SEQUENCES: Dict[str, str] = {
        "A": "EENN",
        "B": "SSWW",
        "C": "NNEE",
    }
    DEFAULT_COLOR_SEQUENCES: Dict[str, str] = {
        "A": "yellow",
        "B": "blue",
        "C": "green",
    }

    def __init__(
        self,
        family_id: str = "A",
        depth: int = 1,
        direction_sequence: Optional[Iterable[str]] = None,
        color_sequence: Optional[Iterable[str]] = None,
        max_depth: int = 4,
        max_steps_per_stage: int = 20,
        mission_text: Optional[str] = None,
        **kwargs,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.family_id = family_id
        self.depth = depth
        self.max_depth = max(max_depth, depth)
        self.canvas_radius = self.max_depth + 1

        raw_direction_sequence = direction_sequence
        if raw_direction_sequence is None:
            raw_direction_sequence = self.DEFAULT_DIRECTION_SEQUENCES.get(
                family_id, "E" * self.max_depth
            )
        raw_color_sequence = color_sequence
        if raw_color_sequence is None:
            raw_color_sequence = self.DEFAULT_COLOR_SEQUENCES.get(family_id, "yellow")

        self.family_direction_sequence = self._normalize_direction_sequence(
            raw_direction_sequence, self.max_depth
        )
        self.family_color_sequence = self._normalize_color_sequence(
            raw_color_sequence, self.max_depth
        )
        self.target_directions = self.family_direction_sequence[: self.depth]
        self.target_colors = self.family_color_sequence[: self.depth]

        self.chain_cells = self._build_target_chain(self.target_directions)
        self.chain_cell_set = set(self.chain_cells)
        if len(self.chain_cell_set) != len(self.chain_cells):
            raise ValueError(
                f"Target direction sequence for family {family_id} self-intersects within depth={depth}: {self.target_directions}"
            )

        canvas_span = 2 * self.canvas_radius
        canvas_size = 2 * self.PADDING + self.ROOM_SIZE + canvas_span * self.ROOM_STEP
        self.goal_pos = None
        self.latest_stage_plans: List[dict] = []

        if mission_text is None:
            unique_colors = sorted(set(self.target_colors))
            if len(unique_colors) == 1:
                mission_text = f"open the {unique_colors[0]} doors and reach the goal"
            else:
                mission_text = "open the target doors in order and reach the goal"
        self._mission_text = mission_text

        mission_space = MissionSpace(mission_func=lambda: self._mission_text)
        super().__init__(
            mission_space=mission_space,
            width=canvas_size,
            height=canvas_size,
            max_steps=(depth + 1) * max_steps_per_stage,
            **kwargs,
        )

    def seed(self, seed=None):
        self.reset(seed=seed)
        return [seed]

    def _normalize_direction_sequence(
        self, direction_sequence: Iterable[str], required_length: int
    ) -> Tuple[str, ...]:
        if isinstance(direction_sequence, str):
            clean = direction_sequence.replace(",", " ").split()
            if len(clean) == 1 and len(clean[0]) > 1:
                directions = [ch.upper() for ch in clean[0]]
            else:
                directions = [token.upper() for token in clean]
        else:
            directions = [str(token).upper() for token in direction_sequence]

        if len(directions) == 1:
            directions = directions * required_length
        if len(directions) < required_length:
            raise ValueError(
                f"direction_sequence must have length >= {required_length}; got {directions}"
            )
        if any(direction not in self.DIRECTIONS for direction in directions[:required_length]):
            raise ValueError(
                f"direction_sequence must only contain {self.DIRECTIONS}; got {directions}"
            )
        return tuple(directions[:required_length])

    def _normalize_color_sequence(
        self, color_sequence: Iterable[str], required_length: int
    ) -> Tuple[str, ...]:
        if isinstance(color_sequence, str):
            clean = color_sequence.replace(",", " ").split()
            colors = [token.lower() for token in clean] if clean else [color_sequence.lower()]
        else:
            colors = [str(token).lower() for token in color_sequence]

        if len(colors) == 1:
            colors = colors * required_length
        if len(colors) < required_length:
            raise ValueError(
                f"color_sequence must have length >= {required_length}; got {colors}"
            )
        if any(color not in COLOR_NAMES for color in colors[:required_length]):
            raise ValueError(
                f"color_sequence must only contain valid MiniGrid colors {tuple(COLOR_NAMES)}; got {colors}"
            )
        return tuple(colors[:required_length])

    def _build_target_chain(self, directions: Sequence[str]) -> List[Tuple[int, int]]:
        cells = [(0, 0)]
        x, y = 0, 0
        for direction in directions:
            dx, dy = self.DIR_TO_DELTA[direction]
            x += dx
            y += dy
            cells.append((x, y))
        return cells

    def _fill_walls(self):
        self.grid = Grid(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                self.grid.set(x, y, Wall())

    def _cell_to_top(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        return (
            self.PADDING + (cell[0] + self.canvas_radius) * self.ROOM_STEP,
            self.PADDING + (cell[1] + self.canvas_radius) * self.ROOM_STEP,
        )

    def _carve_room(self, cell: Tuple[int, int]):
        top_x, top_y = self._cell_to_top(cell)
        for x in range(top_x + 1, top_x + self.ROOM_SIZE - 1):
            for y in range(top_y + 1, top_y + self.ROOM_SIZE - 1):
                self.grid.set(x, y, None)

    def _door_pos(self, cell: Tuple[int, int], direction: str) -> Tuple[int, int]:
        top_x, top_y = self._cell_to_top(cell)
        offset_x, offset_y = self.DIR_TO_DOOR_OFFSET[direction]
        return top_x + offset_x, top_y + offset_y

    def _center_pos(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        top_x, top_y = self._cell_to_top(cell)
        return top_x + self.ROOM_CENTER, top_y + self.ROOM_CENTER

    def _randomized_directions(self) -> List[str]:
        remaining = list(self.DIRECTIONS)
        ordered: List[str] = []
        while remaining:
            idx = self._rand_int(0, len(remaining))
            ordered.append(remaining.pop(idx))
        return ordered

    def _sample_distractor_room_assignments(self) -> List[Tuple[str, Tuple[int, int]]]:
        def backtrack(stage_idx: int, used_cells: set) -> Optional[List[Tuple[str, Tuple[int, int]]]]:
            if stage_idx == self.depth:
                return []

            room_cell = self.chain_cells[stage_idx]
            target_dir = self.target_directions[stage_idx]
            candidates: List[Tuple[str, Tuple[int, int]]] = []
            for distractor_dir in self._randomized_directions():
                if distractor_dir == target_dir:
                    continue
                dx, dy = self.DIR_TO_DELTA[distractor_dir]
                distractor_cell = (room_cell[0] + dx, room_cell[1] + dy)
                if distractor_cell in self.chain_cell_set or distractor_cell in used_cells:
                    continue
                candidates.append((distractor_dir, distractor_cell))

            for distractor_dir, distractor_cell in candidates:
                used_cells.add(distractor_cell)
                suffix = backtrack(stage_idx + 1, used_cells)
                if suffix is not None:
                    return [(distractor_dir, distractor_cell)] + suffix
                used_cells.remove(distractor_cell)
            return None

        assignments = backtrack(0, set())
        if assignments is None:
            raise ValueError(
                f"Could not place distractor rooms without overlap for family {self.family_id} and target sequence {self.target_directions}."
            )
        return assignments

    def _sample_distractor_color(self, target_color: str) -> str:
        distractor_colors = [color for color in COLOR_NAMES if color != target_color]
        return distractor_colors[self._rand_int(0, len(distractor_colors))]

    def _sample_stage_plans(self) -> List[StagePlan]:
        distractor_assignments = self._sample_distractor_room_assignments()
        stage_plans: List[StagePlan] = []
        for stage_idx in range(self.depth):
            distractor_dir, distractor_cell = distractor_assignments[stage_idx]
            stage_plans.append(
                StagePlan(
                    stage_idx=stage_idx,
                    room_cell=self.chain_cells[stage_idx],
                    next_cell=self.chain_cells[stage_idx + 1],
                    distractor_cell=distractor_cell,
                    target_dir=self.target_directions[stage_idx],
                    distractor_dir=distractor_dir,
                    target_color=self.target_colors[stage_idx],
                    distractor_color=self._sample_distractor_color(self.target_colors[stage_idx]),
                )
            )
        return stage_plans

    def _place_door(self, cell: Tuple[int, int], direction: str, color: str):
        door_x, door_y = self._door_pos(cell, direction)
        self.grid.set(door_x, door_y, Door(color, is_locked=False))

    def _gen_grid(self, width: int, height: int):
        del width, height
        self._fill_walls()

        stage_plans = self._sample_stage_plans()
        self.latest_stage_plans = [
            {
                "stage_idx": plan.stage_idx,
                "room_cell": tuple(plan.room_cell),
                "next_cell": tuple(plan.next_cell),
                "distractor_cell": tuple(plan.distractor_cell),
                "target_dir": plan.target_dir,
                "distractor_dir": plan.distractor_dir,
                "target_color": plan.target_color,
                "distractor_color": plan.distractor_color,
            }
            for plan in stage_plans
        ]

        all_room_cells = set(self.chain_cells)
        all_room_cells.update(plan.distractor_cell for plan in stage_plans)
        for room_cell in all_room_cells:
            self._carve_room(room_cell)

        for plan in stage_plans:
            self._place_door(plan.room_cell, plan.target_dir, plan.target_color)
            self._place_door(plan.room_cell, plan.distractor_dir, plan.distractor_color)

        goal_cell = self.chain_cells[-1]
        goal_x, goal_y = self._center_pos(goal_cell)
        self.grid.set(goal_x, goal_y, Goal())
        self.goal_pos = (goal_x, goal_y)

        start_cell = self.chain_cells[0]
        self.agent_pos = self._center_pos(start_cell)
        self.agent_dir = 0
        self.carrying = None
        self.mission = self._mission_text

    @property
    def task_name(self) -> str:
        return f"v2-{self.family_id}{self.depth}"

    @property
    def task_metadata(self) -> dict:
        return {
            "family_id": self.family_id,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "family_direction_sequence": list(self.family_direction_sequence),
            "target_directions": list(self.target_directions),
            "family_color_sequence": list(self.family_color_sequence),
            "target_colors": list(self.target_colors),
            "chain_cells": [tuple(cell) for cell in self.chain_cells],
            "latest_stage_plans": self.latest_stage_plans,
        }


def register_curriculum_multiroom_doors_envs_v2(
    max_depth: int = 4,
    families: Tuple[str, ...] = ("A", "B", "C"),
    family_direction_sequences: Optional[Dict[str, Iterable[str]]] = None,
    family_color_sequences: Optional[Dict[str, Iterable[str]]] = None,
    entry_point: str = "CurriculumMinigrid.curriculum_multiroom_doors_env_v2:CurriculumMultiRoomDoorsEnvV2",
):
    family_direction_sequences = family_direction_sequences or {}
    family_color_sequences = family_color_sequences or {}

    for family_id in families:
        for depth in range(1, max_depth + 1):
            kwargs = {
                "family_id": family_id,
                "depth": depth,
                "max_depth": max_depth,
            }
            if family_id in family_direction_sequences:
                kwargs["direction_sequence"] = family_direction_sequences[family_id]
            if family_id in family_color_sequences:
                kwargs["color_sequence"] = family_color_sequences[family_id]
            register(
                id=f"CurriculumMultiRoomDoorsV2-{family_id}{depth}-v0",
                entry_point=entry_point,
                kwargs=kwargs,
            )


register_curriculum_multiroom_doors_envs_v2(
    max_depth=4,
    family_direction_sequences={
        "A": "ENEN",
        "B": "SWSW",
        "C": "WNWN",
    },
    family_color_sequences={
        "A": "yellow",
        "B": "blue",
        "C": "green",
    },
)
