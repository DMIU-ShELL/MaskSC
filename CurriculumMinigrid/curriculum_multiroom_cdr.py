"""
MiniGrid Checkpoint Door Routing benchmark.

This environment is designed to test selective reuse of prior policies in a
continual RL setting without requiring long-horizon memory or state-conditioned
skill stitching.

Core design:
- a single 5x5 checkpoint room is reused across stages
- each decision room contains 3 doors at the center of the north/east/south walls
- door colors (yellow/blue/green) are randomly permuted across those faces each stage
- the correct door color depends on the task family and the stage index
- traversing the correct door advances the same episode to the next decision room
- traversing a wrong door terminates the episode with zero reward
- after the final correct door, the agent is moved into a simple goal room

This makes task A(k) an exact prefix of A(k+1), but each deeper task adds one
new local decision under sparse reward.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gym.envs.registration import register

from gym_minigrid.minigrid import Ball, Box, Door, Goal, Grid, MiniGridEnv, MissionSpace, Wall


@dataclass(frozen=True)
class StageContext:
    stage_idx: int
    target_color: str
    correct_dir: str
    door_color_by_dir: Dict[str, str]
    family_cue_color: str
    stage_cue_pos: Tuple[int, int]


class CurriculumMultiRoomCDREnv(MiniGridEnv):
    ROOM_SIZE = 5
    ROOM_CENTER = 2

    CANDIDATE_DIRS: Tuple[str, ...] = ("N", "E", "S")
    DOOR_COLORS: Tuple[str, ...] = ("yellow", "blue", "green")
    DIR_TO_DOOR_POS: Dict[str, Tuple[int, int]] = {
        "N": (ROOM_CENTER, 0),
        "E": (ROOM_SIZE - 1, ROOM_CENTER),
        "S": (ROOM_CENTER, ROOM_SIZE - 1),
    }
    STAGE_CUE_POSITIONS: Tuple[Tuple[int, int], ...] = (
        (1, 1),
        (3, 1),
        (3, 3),
        (1, 3),
    )
    FAMILY_CUE_POS = (1, 2)
    DECISION_SPAWN_POS = (2, 2)
    GOAL_ROOM_SPAWN_POS = (3, 2)
    GOAL_POS = (2, 2)

    DEFAULT_FAMILY_TARGET_COLOR_CODEBOOKS: Dict[str, Tuple[str, ...]] = {
        "A": ("yellow", "blue", "green", "yellow"),
        "B": ("blue", "green", "yellow", "blue"),
        "C": ("green", "yellow", "blue", "green"),
    }
    DEFAULT_FAMILY_CUE_COLORS: Dict[str, str] = {
        "A": "red",
        "B": "blue",
        "C": "green",
    }

    def __init__(
        self,
        family_id: str = "A",
        depth: int = 1,
        family_target_color_codebooks: Optional[Dict[str, Iterable[str]]] = None,
        family_cue_colors: Optional[Dict[str, str]] = None,
        max_depth: int = 4,
        max_steps_per_stage: int = 12,
        mission_text: Optional[str] = None,
        **kwargs,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")

        if family_target_color_codebooks is None:
            family_target_color_codebooks = self.DEFAULT_FAMILY_TARGET_COLOR_CODEBOOKS
        if family_cue_colors is None:
            family_cue_colors = self.DEFAULT_FAMILY_CUE_COLORS
        if family_id not in family_target_color_codebooks:
            raise ValueError(f"Unknown family_id={family_id}")
        if family_id not in family_cue_colors:
            raise ValueError(f"Missing cue color for family_id={family_id}")

        self.family_id = family_id
        self.depth = depth
        self.max_depth = max(max_depth, depth)
        self.family_cue_colors = {k: str(v).lower() for k, v in family_cue_colors.items()}
        self.family_cue_color = self.family_cue_colors[family_id]
        self.family_target_color_codebooks = {
            key: self._normalize_color_sequence(value, self.max_depth)
            for key, value in family_target_color_codebooks.items()
        }
        self.target_color_codebook = self.family_target_color_codebooks[family_id][: self.depth]

        self.current_stage = 0
        self.in_goal_room = False
        self.stage_contexts: List[StageContext] = []
        self.current_door_pos_to_dir: Dict[Tuple[int, int], str] = {}
        self.goal_pos = None

        if mission_text is None:
            mission_text = "follow the checkpoint cues and reach the goal"
        self._mission_text = mission_text

        mission_space = MissionSpace(mission_func=lambda: self._mission_text)
        super().__init__(
            mission_space=mission_space,
            width=self.ROOM_SIZE,
            height=self.ROOM_SIZE,
            max_steps=(self.depth + 1) * max_steps_per_stage,
            **kwargs,
        )

    def seed(self, seed=None):
        self.reset(seed=seed)
        return [seed]

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
                f"color sequence must have length >= {required_length}; got {colors}"
            )
        if any(color not in self.DOOR_COLORS for color in colors[:required_length]):
            raise ValueError(
                f"target colors must be chosen from {self.DOOR_COLORS}; got {colors}"
            )
        return tuple(colors[:required_length])

    def _sample_stage_contexts(self) -> List[StageContext]:
        contexts: List[StageContext] = []
        for stage_idx, target_color in enumerate(self.target_color_codebook):
            shuffled_colors = list(self.DOOR_COLORS)
            for idx in range(len(shuffled_colors) - 1, 0, -1):
                swap_idx = self._rand_int(0, idx + 1)
                shuffled_colors[idx], shuffled_colors[swap_idx] = (
                    shuffled_colors[swap_idx],
                    shuffled_colors[idx],
                )
            door_color_by_dir = {
                direction: color
                for direction, color in zip(self.CANDIDATE_DIRS, shuffled_colors)
            }
            correct_dir = next(
                direction
                for direction, color in door_color_by_dir.items()
                if color == target_color
            )
            contexts.append(
                StageContext(
                    stage_idx=stage_idx,
                    target_color=target_color,
                    correct_dir=correct_dir,
                    door_color_by_dir=door_color_by_dir,
                    family_cue_color=self.family_cue_color,
                    stage_cue_pos=self.STAGE_CUE_POSITIONS[stage_idx % len(self.STAGE_CUE_POSITIONS)],
                )
            )
        return contexts

    def _reset_grid(self):
        self.grid = Grid(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                self.grid.set(x, y, Wall())
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                self.grid.set(x, y, None)

    def _load_decision_room(self, stage_idx: int):
        context = self.stage_contexts[stage_idx]
        self._reset_grid()
        self.current_door_pos_to_dir = {}

        for direction, pos in self.DIR_TO_DOOR_POS.items():
            color = context.door_color_by_dir[direction]
            self.grid.set(pos[0], pos[1], Door(color, is_locked=False))
            self.current_door_pos_to_dir[pos] = direction

        self.grid.set(self.FAMILY_CUE_POS[0], self.FAMILY_CUE_POS[1], Ball(context.family_cue_color))
        self.grid.set(context.stage_cue_pos[0], context.stage_cue_pos[1], Box("grey"))

        self.agent_pos = self.DECISION_SPAWN_POS
        self.agent_dir = 0  # east
        self.carrying = None
        self.goal_pos = None
        self.in_goal_room = False
        self.current_stage = stage_idx
        self.mission = self._mission_text

    def _load_goal_room(self):
        self._reset_grid()
        goal_x, goal_y = self.GOAL_POS
        self.grid.set(goal_x, goal_y, Goal())
        self.grid.set(self.FAMILY_CUE_POS[0], self.FAMILY_CUE_POS[1], Ball(self.family_cue_color))
        self.agent_pos = self.GOAL_ROOM_SPAWN_POS
        self.agent_dir = 2  # west
        self.carrying = None
        self.goal_pos = self.GOAL_POS
        self.current_door_pos_to_dir = {}
        self.in_goal_room = True
        self.current_stage = self.depth
        self.mission = self._mission_text

    def _advance_after_correct_door(self):
        next_stage = self.current_stage + 1
        if next_stage >= self.depth:
            self._load_goal_room()
        else:
            self._load_decision_room(next_stage)

    def _gen_grid(self, width: int, height: int):
        del width, height
        self.stage_contexts = self._sample_stage_contexts()
        self._load_decision_room(0)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = dict(info)

        if not terminated and not truncated and not self.in_goal_room:
            current_pos = tuple(self.agent_pos)
            if action == self.actions.forward and current_pos in self.current_door_pos_to_dir:
                chosen_dir = self.current_door_pos_to_dir[current_pos]
                correct_dir = self.stage_contexts[self.current_stage].correct_dir
                if chosen_dir == correct_dir:
                    self._advance_after_correct_door()
                    obs = self.gen_obs()
                else:
                    terminated = True
                    reward = 0

        info.update(
            {
                "family_id": self.family_id,
                "depth": self.depth,
                "current_stage": self.current_stage,
                "in_goal_room": self.in_goal_room,
            }
        )
        return obs, reward, terminated, truncated, info

    @property
    def task_name(self) -> str:
        return f"cdr-{self.family_id}{self.depth}"

    @property
    def task_metadata(self) -> dict:
        return {
            "family_id": self.family_id,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "family_cue_color": self.family_cue_color,
            "target_color_codebook": list(self.target_color_codebook),
            "stage_contexts": [
                {
                    "stage_idx": context.stage_idx,
                    "target_color": context.target_color,
                    "correct_dir": context.correct_dir,
                    "door_color_by_dir": dict(context.door_color_by_dir),
                    "family_cue_color": context.family_cue_color,
                    "stage_cue_pos": tuple(context.stage_cue_pos),
                }
                for context in self.stage_contexts
            ],
            "current_stage": self.current_stage,
            "in_goal_room": self.in_goal_room,
        }


def register_curriculum_multiroom_cdr_envs(
    max_depth: int = 4,
    families: Tuple[str, ...] = ("A", "B", "C"),
    family_target_color_codebooks: Optional[Dict[str, Iterable[str]]] = None,
    family_cue_colors: Optional[Dict[str, str]] = None,
    entry_point: str = "CurriculumMinigrid.curriculum_multiroom_cdr:CurriculumMultiRoomCDREnv",
):
    if family_target_color_codebooks is None:
        family_target_color_codebooks = CurriculumMultiRoomCDREnv.DEFAULT_FAMILY_TARGET_COLOR_CODEBOOKS
    if family_cue_colors is None:
        family_cue_colors = CurriculumMultiRoomCDREnv.DEFAULT_FAMILY_CUE_COLORS

    for family_id in families:
        for depth in range(1, max_depth + 1):
            register(
                id=f"CurriculumMultiRoomCDR-{family_id}{depth}-v0",
                entry_point=entry_point,
                kwargs={
                    "family_id": family_id,
                    "depth": depth,
                    "max_depth": max_depth,
                    "family_target_color_codebooks": family_target_color_codebooks,
                    "family_cue_colors": family_cue_colors,
                },
            )


register_curriculum_multiroom_cdr_envs(max_depth=4)
