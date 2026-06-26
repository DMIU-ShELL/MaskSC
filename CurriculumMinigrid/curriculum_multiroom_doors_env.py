"""
Hand-crafted MiniGrid branch-selection benchmark with unlocked colored doors.

Goal of this benchmark:
- repeated room motif
- fixed family rule across depth
- no keys
- one sparse goal at the end
- selective reuse should matter because family identity determines which
  door color is the correct branch at every stage.

Families:
- A: always choose yellow
- B: always choose blue
- C: always choose green

Difficulties:
- easy:   target color vs neutral distractor
- medium: target color vs one competing family color
- hard:   target color vs both competing family colors

Each stage is a 5x5 room with candidate doors on the east wall. The correct
branch continues to the next stage; distractor branches are short dead-end
corridors. By default the correct door row changes with depth so the agent
cannot solve by always choosing the same wall position.

You can optionally override that row pattern with `binary_sequence` for the
two-door variants, either by direct construction or through env registration:
- `1` -> top/left candidate door
- `0` -> bottom/right candidate door

Example:
- `binary_sequence=\"1111\"` means every stage uses the top/left door
- `binary_sequence=\"1010\"` alternates top/left then bottom/right
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gym.envs.registration import register

from gym_minigrid.minigrid import Door, Goal, Grid, MiniGridEnv, MissionSpace, Wall


@dataclass(frozen=True)
class VariantSpec:
    candidate_rows: Tuple[int, ...]
    correct_row_cycle: Tuple[int, ...]


@dataclass(frozen=True)
class HallPlacement:
    top: Tuple[int, int]
    correct_row: int


class CurriculumMultiRoomDoorsEnv(MiniGridEnv):
    FAMILY_TARGET_COLORS: Dict[str, str] = {
        "A": "yellow",
        "B": "blue",
        "C": "green",
    }

    VARIANT_SPECS: Dict[str, VariantSpec] = {
        "easy": VariantSpec(candidate_rows=(1, 3), correct_row_cycle=(1, 3)),
        "medium": VariantSpec(candidate_rows=(1, 3), correct_row_cycle=(3, 1)),
        "hard": VariantSpec(candidate_rows=(1, 2, 3), correct_row_cycle=(1, 3, 2)),
    }

    ROOM_SIZE = 5
    ENTRY_ROW = 2
    CORRIDOR_LENGTH = 2
    GOAL_LOCAL_POS = (3, ENTRY_ROW)
    BASE_X = 2
    BASE_Y = 2

    def __init__(
        self,
        family_id: str = "A",
        depth: int = 1,
        variant: str = "easy",
        binary_sequence: Optional[Iterable[int]] = None,
        max_depth: int = 4,
        max_steps_per_stage: int = 20,
        mission_text: Optional[str] = None,
        **kwargs,
    ):
        if family_id not in self.FAMILY_TARGET_COLORS:
            raise ValueError(f"Unknown family_id={family_id}")
        if variant not in self.VARIANT_SPECS:
            raise ValueError(f"Unknown variant={variant}")
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.family_id = family_id
        self.depth = depth
        self.variant = variant
        self.max_depth = max(max_depth, depth)
        self.target_color = self.FAMILY_TARGET_COLORS[family_id]
        self.variant_spec = self.VARIANT_SPECS[variant]
        self.binary_sequence = self._normalize_binary_sequence(binary_sequence)

        layout = self._plan_layout(depth)
        translated_layout, goal_room_top, width, height = self._translate_layout(layout)
        self.layout = translated_layout
        self.goal_room_top = goal_room_top
        self.size = width
        self.goal_pos = None

        if mission_text is None:
            mission_text = (
                f"open the {self.target_color} door in each room and reach the goal"
            )
        self._mission_text = mission_text

        mission_space = MissionSpace(mission_func=lambda: self._mission_text)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=(depth + 1) * max_steps_per_stage,
            **kwargs,
        )

    def seed(self, seed=None):
        self.reset(seed=seed)
        return [seed]

    def _normalize_binary_sequence(
        self, binary_sequence: Optional[Iterable[int]]
    ) -> Optional[Tuple[int, ...]]:
        if binary_sequence is None:
            return None

        if len(self.variant_spec.candidate_rows) != 2:
            raise ValueError(
                "binary_sequence is only supported for two-door variants"
            )

        if isinstance(binary_sequence, int):
            bits = tuple(int(ch) for ch in str(binary_sequence))
        elif isinstance(binary_sequence, str):
            bits = tuple(int(ch) for ch in binary_sequence.strip())
        else:
            bits = tuple(int(bit) for bit in binary_sequence)

        if len(bits) < self.depth:
            raise ValueError("binary_sequence length must be >= depth")
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("binary_sequence must contain only 0/1 values")
        return bits

    def _plan_layout(self, depth: int) -> List[HallPlacement]:
        placements: List[HallPlacement] = []
        top_x, top_y = 0, 0
        for stage_idx in range(depth):
            correct_row = self._correct_row_for_stage(stage_idx)
            placements.append(HallPlacement(top=(top_x, top_y), correct_row=correct_row))
            top_x += self.ROOM_SIZE + self.CORRIDOR_LENGTH
            top_y += correct_row - self.ENTRY_ROW
        return placements

    def _translate_layout(
        self, layout: Sequence[HallPlacement]
    ) -> Tuple[List[HallPlacement], Tuple[int, int], int, int]:
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0

        for placement in layout:
            room_x, room_y = placement.top
            max_x = max(max_x, room_x + self.ROOM_SIZE - 1)
            min_y = min(min_y, room_y)
            max_y = max(max_y, room_y + self.ROOM_SIZE - 1)
            for row in self.variant_spec.candidate_rows:
                corridor_y = room_y + row
                max_x = max(max_x, room_x + self.ROOM_SIZE + self.CORRIDOR_LENGTH - 1)
                min_y = min(min_y, corridor_y)
                max_y = max(max_y, corridor_y)

        last_room = layout[-1]
        goal_room_top = (
            last_room.top[0] + self.ROOM_SIZE + self.CORRIDOR_LENGTH,
            last_room.top[1] + last_room.correct_row - self.ENTRY_ROW,
        )
        max_x = max(max_x, goal_room_top[0] + self.ROOM_SIZE - 1)
        min_y = min(min_y, goal_room_top[1])
        max_y = max(max_y, goal_room_top[1] + self.ROOM_SIZE - 1)

        shift_x = self.BASE_X - min_x
        shift_y = self.BASE_Y - min_y

        translated_layout = [
            HallPlacement(
                top=(placement.top[0] + shift_x, placement.top[1] + shift_y),
                correct_row=placement.correct_row,
            )
            for placement in layout
        ]
        translated_goal_room_top = (goal_room_top[0] + shift_x, goal_room_top[1] + shift_y)
        width = max_x + shift_x + 3
        height = max_y + shift_y + 3
        return translated_layout, translated_goal_room_top, width, height

    def _correct_row_for_stage(self, stage_idx: int) -> int:
        if self.binary_sequence is not None:
            top_row = min(self.variant_spec.candidate_rows)
            bottom_row = max(self.variant_spec.candidate_rows)
            return top_row if self.binary_sequence[stage_idx] == 1 else bottom_row
        cycle = self.variant_spec.correct_row_cycle
        return cycle[stage_idx % len(cycle)]

    def _candidate_colors_for_stage(self, stage_idx: int) -> List[str]:
        target = self.target_color
        if self.variant == "easy":
            return [target, "grey"]

        competing = [
            color
            for fam, color in self.FAMILY_TARGET_COLORS.items()
            if fam != self.family_id
        ]
        if stage_idx % 2 == 1:
            competing = list(reversed(competing))

        if self.variant == "medium":
            return [target, competing[0]]
        if self.variant == "hard":
            return [target] + competing
        raise ValueError(f"Unhandled variant={self.variant}")

    def _door_row_to_color(self, stage_idx: int, correct_row: int) -> Dict[int, str]:
        door_rows = list(self.variant_spec.candidate_rows)
        colors = self._candidate_colors_for_stage(stage_idx)
        if len(door_rows) != len(colors):
            raise ValueError("door/color specification mismatch")

        mapping = {correct_row: colors[0]}
        distractor_rows = [row for row in door_rows if row != correct_row]
        for row, color in zip(distractor_rows, colors[1:]):
            mapping[row] = color
        return mapping

    def _fill_walls(self):
        self.grid = Grid(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                self.grid.set(x, y, Wall())

    def _carve_room(self, top: Tuple[int, int]):
        top_x, top_y = top
        for x in range(top_x + 1, top_x + self.ROOM_SIZE - 1):
            for y in range(top_y + 1, top_y + self.ROOM_SIZE - 1):
                self.grid.set(x, y, None)

    def _carve_horizontal_corridor(self, start_x: int, y: int, length: int):
        for x in range(start_x, start_x + length):
            self.grid.set(x, y, None)

    def _door_pos(self, top: Tuple[int, int], local_row: int) -> Tuple[int, int]:
        return top[0] + self.ROOM_SIZE - 1, top[1] + local_row

    def _set_room_entry_opening(self, top: Tuple[int, int]):
        self.grid.set(top[0], top[1] + self.ENTRY_ROW, None)

    def _goal_pos_from_room(self, top: Tuple[int, int]) -> Tuple[int, int]:
        return top[0] + self.GOAL_LOCAL_POS[0], top[1] + self.GOAL_LOCAL_POS[1]

    def _gen_grid(self, width: int, height: int):
        del width, height
        self._fill_walls()

        for stage_idx, placement in enumerate(self.layout):
            room_top = placement.top
            correct_row = placement.correct_row
            self._carve_room(room_top)
            if stage_idx > 0:
                self._set_room_entry_opening(room_top)

            door_row_to_color = self._door_row_to_color(stage_idx, correct_row)
            for door_row, color in door_row_to_color.items():
                door_x, door_y = self._door_pos(room_top, door_row)
                self.grid.set(door_x, door_y, Door(color, is_locked=False))
                self._carve_horizontal_corridor(
                    room_top[0] + self.ROOM_SIZE,
                    door_y,
                    self.CORRIDOR_LENGTH,
                )

        self._carve_room(self.goal_room_top)
        self._set_room_entry_opening(self.goal_room_top)
        goal_x, goal_y = self._goal_pos_from_room(self.goal_room_top)
        self.grid.set(goal_x, goal_y, Goal())
        self.goal_pos = (goal_x, goal_y)

        start_top = self.layout[0].top
        self.agent_pos = (start_top[0] + 1, start_top[1] + self.ENTRY_ROW)
        self.agent_dir = 0
        self.carrying = None
        self.mission = self._mission_text

    @property
    def task_name(self) -> str:
        return f"{self.variant}-{self.family_id}{self.depth}"

    @property
    def task_metadata(self) -> dict:
        return {
            "variant": self.variant,
            "family_id": self.family_id,
            "target_color": self.target_color,
            "depth": self.depth,
            "candidate_rows": list(self.variant_spec.candidate_rows),
            "binary_sequence": list(self.binary_sequence) if self.binary_sequence is not None else None,
            "correct_rows": [placement.correct_row for placement in self.layout],
            "goal_room_top": tuple(self.goal_room_top),
        }


def register_curriculum_multiroom_doors_envs(
    max_depth: int = 4,
    families: Tuple[str, ...] = ("A", "B", "C"),
    variants: Tuple[str, ...] = ("easy", "medium", "hard"),
    family_binary_sequences: Optional[Dict[str, Iterable[int]]] = None,
    entry_point: str = "CurriculumMinigrid.curriculum_multiroom_doors_env:CurriculumMultiRoomDoorsEnv",
):
    if family_binary_sequences is None:
        family_binary_sequences = {}

    for variant in variants:
        for family_id in families:
            for depth in range(1, max_depth + 1):
                kwargs = {
                    "family_id": family_id,
                    "depth": depth,
                    "variant": variant,
                    "max_depth": max_depth,
                }
                if family_id in family_binary_sequences and len(
                    CurriculumMultiRoomDoorsEnv.VARIANT_SPECS[variant].candidate_rows
                ) == 2:
                    kwargs["binary_sequence"] = family_binary_sequences[family_id]
                register(
                    id=f"CurriculumMultiRoomDoors{variant.capitalize()}-{family_id}{depth}-v0",
                    entry_point=entry_point,
                    kwargs=kwargs,
                )

register_curriculum_multiroom_doors_envs(
    max_depth=4,
    family_binary_sequences={
        "A": "1011",
        "B": "0110",
        "C": "1101"
    },
)
