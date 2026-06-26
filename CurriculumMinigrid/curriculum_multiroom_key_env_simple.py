"""
Hand-crafted MiniGrid door-key curriculum with exact room archetypes.

This variant is intended to remove the geometry-mismatch confound from the
procedural `curriculum_multiroom_key_env.py` benchmark. Each color corresponds
to a fixed room template, and multi-room tasks are built by chaining those same
templates in a deterministic left-to-right layout.

The key property is:
  - The one-room task for a color is the exact local subproblem used inside
    deeper multi-room tasks.

This makes the benchmark better suited for studying whether selective reuse
works when the reusable prior is clearly relevant, without also demanding strong
procedural generalization.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from gym.envs.registration import register

from gym_minigrid.minigrid import Door, Goal, Grid, Key, MiniGridEnv, MissionSpace, Wall


@dataclass(frozen=True)
class RoomArchetype:
    key_pos: Tuple[int, int]
    blockers: Tuple[Tuple[int, int], ...] = ()


class AutoConsumeDoor(Door):
    """
    Locked door variant that consumes the matching carried key on successful
    unlock. This avoids the need for an explicit drop action between rooms.
    """

    def toggle(self, env, pos):
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                env.carrying = None
                return True
            return False
        return super().toggle(env, pos)


class CurriculumMultiRoomKeySimpleEnv(MiniGridEnv):
    DEFAULT_FAMILY_SEQUENCES: Dict[str, Tuple[str, ...]] = {
        "A": ("yellow", "red", "blue", "green", "purple", "grey"),
        "B": ("blue", "yellow", "green", "purple", "red", "grey"),
        "C": ("green", "purple", "yellow", "red", "blue", "grey"),
        "D": ("red", "blue", "purple", "yellow", "green", "grey"),
        "E": ("purple",),
        "F": ("grey",),
    }

    DEFAULT_ROOM_ARCHETYPES: Dict[str, RoomArchetype] = {
        "yellow": RoomArchetype(key_pos=(2, 3), blockers=((3, 1),)),
        "red": RoomArchetype(key_pos=(3, 2), blockers=((3, 3),)),
        "blue": RoomArchetype(key_pos=(3, 3), blockers=((1, 1), (1, 3))),
        "green": RoomArchetype(key_pos=(2, 1), blockers=((3, 3),)),
        "purple": RoomArchetype(key_pos=(1, 3), blockers=((2, 2),)),
        "grey": RoomArchetype(key_pos=(3, 1), blockers=((2, 3),)),
    }

    ROOM_WIDTH = 5
    ROOM_HEIGHT = 5
    DOOR_ROW = 2
    AGENT_START_LOCAL = (1, DOOR_ROW)
    GOAL_LOCAL = (2, DOOR_ROW)

    def __init__(
        self,
        family_id: str = "A",
        depth: int = 1,
        family_sequences: Optional[Dict[str, Iterable[str]]] = None,
        room_archetypes: Optional[Dict[str, RoomArchetype]] = None,
        canvas_depth: Optional[int] = None,
        max_steps_per_room: int = 30,
        consume_key_on_unlock: bool = True,
        **kwargs,
    ):
        assert depth >= 1, "depth must be >= 1"

        if family_sequences is None:
            family_sequences = self.DEFAULT_FAMILY_SEQUENCES
        if room_archetypes is None:
            room_archetypes = self.DEFAULT_ROOM_ARCHETYPES

        if family_id not in family_sequences:
            raise ValueError(f"Unknown family_id={family_id}")

        sequence = tuple(family_sequences[family_id])
        if depth > len(sequence):
            raise ValueError(
                f"depth={depth} exceeds sequence length={len(sequence)} for family {family_id}"
            )
        if canvas_depth is None:
            canvas_depth = depth
        if canvas_depth < depth:
            raise ValueError("canvas_depth must be >= depth")

        self.family_id = family_id
        self.depth = depth
        self.canvas_depth = canvas_depth
        self.family_sequences = {k: tuple(v) for k, v in family_sequences.items()}
        self.color_prefix = sequence[:depth]
        self.room_archetypes = dict(room_archetypes)
        self.consume_key_on_unlock = consume_key_on_unlock

        for color in self.color_prefix:
            if color not in self.room_archetypes:
                raise ValueError(f"Missing room archetype for color={color}")

        self.num_rooms = depth + 1  # stage rooms + final goal room
        self.canvas_num_rooms = self.canvas_depth + 1
        width = 2 + self.canvas_num_rooms * (self.ROOM_WIDTH - 1)
        height = self.ROOM_HEIGHT + 2

        mission_space = MissionSpace(
            mission_func=lambda: "collect keys, unlock doors, and reach the goal"
        )

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=self.num_rooms * max_steps_per_room,
            **kwargs,
        )

    def seed(self, seed=None):
        self.reset(seed=seed)
        return [seed]

    def _room_top_left(self, room_idx: int) -> Tuple[int, int]:
        return (1 + room_idx * (self.ROOM_WIDTH - 1), 1)

    def _to_global(self, room_idx: int, local_pos: Tuple[int, int]) -> Tuple[int, int]:
        top_x, top_y = self._room_top_left(room_idx)
        return top_x + local_pos[0], top_y + local_pos[1]

    def _draw_room_shell(self, room_idx: int):
        top_x, top_y = self._room_top_left(room_idx)
        wall = Wall()

        for dx in range(self.ROOM_WIDTH):
            self.grid.set(top_x + dx, top_y, wall)
            self.grid.set(top_x + dx, top_y + self.ROOM_HEIGHT - 1, wall)

        for dy in range(self.ROOM_HEIGHT):
            self.grid.set(top_x, top_y + dy, wall)
            self.grid.set(top_x + self.ROOM_WIDTH - 1, top_y + dy, wall)

    def _place_stage_room_contents(self, room_idx: int, color: str):
        archetype = self.room_archetypes[color]

        for blocker_local in archetype.blockers:
            bx, by = self._to_global(room_idx, blocker_local)
            self.grid.set(bx, by, Wall())

        key_x, key_y = self._to_global(room_idx, archetype.key_pos)
        self.grid.set(key_x, key_y, Key(color))

    def _place_transition_doors(self):
        door_cls = AutoConsumeDoor if self.consume_key_on_unlock else Door

        for room_idx, color in enumerate(self.color_prefix, start=1):
            door_x, door_y = self._to_global(room_idx, (0, self.DOOR_ROW))
            self.grid.set(door_x, door_y, door_cls(color, is_locked=True))

    def _place_agent(self):
        agent_x, agent_y = self._to_global(0, self.AGENT_START_LOCAL)
        self.agent_pos = (agent_x, agent_y)
        self.agent_dir = 0  # face right/east
        self.carrying = None

    def _place_goal(self):
        goal_x, goal_y = self._to_global(self.depth, self.GOAL_LOCAL)
        self.grid.set(goal_x, goal_y, Goal())
        self.goal_pos = (goal_x, goal_y)

    def _gen_grid(self, width: int, height: int):
        del width, height
        self.grid = Grid(self.width, self.height)

        for room_idx in range(self.num_rooms):
            self._draw_room_shell(room_idx)

        for room_idx, color in enumerate(self.color_prefix):
            self._place_stage_room_contents(room_idx, color)

        self._place_transition_doors()
        self._place_goal()
        self._place_agent()
        self.mission = "collect keys, unlock doors, and reach the goal"

    @property
    def task_name(self) -> str:
        return f"{self.family_id}{self.depth}"

    @property
    def task_metadata(self) -> dict:
        return {
            "family_id": self.family_id,
            "depth": self.depth,
            "canvas_depth": self.canvas_depth,
            "color_prefix": list(self.color_prefix),
            "num_rooms": self.num_rooms,
            "canvas_num_rooms": self.canvas_num_rooms,
            "consume_key_on_unlock": self.consume_key_on_unlock,
            "room_archetypes": {
                color: {
                    "key_pos": tuple(spec.key_pos),
                    "blockers": [tuple(pos) for pos in spec.blockers],
                }
                for color, spec in self.room_archetypes.items()
            },
        }


def make_default_family_sequences():
    return {
        "A": ("yellow", "red", "blue", "green", "purple", "grey"),
        "B": ("blue", "yellow", "green", "purple", "red", "grey"),
        "C": ("green", "purple", "yellow", "red", "blue", "grey"),
        "D": ("red", "blue", "purple", "yellow", "green", "grey"),
        "E": ("purple",),
        "F": ("grey",),
    }


def register_curriculum_multiroom_key_simple_envs(
    max_depth: int = 6,
    families: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F"),
    entry_point: str = "CurriculumMinigrid.curriculum_multiroom_key_env_simple:CurriculumMultiRoomKeySimpleEnv",
):
    family_sequences = make_default_family_sequences()

    for fam in families:
        sequence = family_sequences[fam]
        for depth in range(1, min(max_depth, len(sequence)) + 1):
            register(
                id=f"CurriculumMultiRoomKeySimple-{fam}{depth}-v0",
                entry_point=entry_point,
                kwargs={
                    "family_id": fam,
                    "depth": depth,
                    "family_sequences": family_sequences,
                    "canvas_depth": max_depth,
                    "consume_key_on_unlock": True,
                    "max_steps_per_room": 30,
                },
            )


register_curriculum_multiroom_key_simple_envs(max_depth=6)
