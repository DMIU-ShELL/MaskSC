from gym.envs.registration import register, registry

from gym_minigrid.minigrid import (
    COLOR_NAMES,
    Door,
    Goal,
    Grid,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    Wall,
    WorldObj,
)

from .curriculumMultiRoomEnv import (
    CurriculumMultiRoomEnv as _BaseCurriculumMultiRoomEnv,
    _normalize_door_colors,
)


__all__ = [
    "Wall2",
    "CurriculumMultiRoomWallObjectEnv",
    "register_curriculum_wall_object",
]


def _lookup_idx(mapping, idx):
    return mapping[idx]


def _set_idx(mapping, idx, value):
    if isinstance(mapping, dict):
        mapping[idx] = value
        return

    while len(mapping) <= idx:
        mapping.append(None)
    mapping[idx] = value


def _ensure_wall2_registered():
    if "wall2" not in OBJECT_TO_IDX:
        wall2_idx = max(OBJECT_TO_IDX.values()) + 1
        OBJECT_TO_IDX["wall2"] = wall2_idx
        _set_idx(IDX_TO_OBJECT, wall2_idx, "wall2")


class Wall2(Wall):
    """Wall-like blocking object with a distinct MiniGrid object id."""

    def __init__(self, color="grey"):
        super().__init__(color)
        self.type = "wall2"


def _patch_worldobj_decode_for_wall2():
    if getattr(WorldObj, "_wall2_decode_patched", False):
        return

    original_decode = WorldObj.decode

    @staticmethod
    def decode(type_idx, color_idx, state):
        if _lookup_idx(IDX_TO_OBJECT, type_idx) == "wall2":
            return Wall2(_lookup_idx(IDX_TO_COLOR, color_idx))
        return original_decode(type_idx, color_idx, state)

    WorldObj.decode = decode
    WorldObj._wall2_decode_patched = True


_ensure_wall2_registered()
_patch_worldobj_decode_for_wall2()


def _normalize_color_sequence(colors, default_colors):
    if colors is None:
        colors = list(default_colors)
    elif isinstance(colors, str):
        clean = colors.replace(",", " ").split()
        colors = [token.lower() for token in clean] if clean else [colors.lower()]
    else:
        colors = [str(color).lower() for color in colors]

    if not colors:
        raise ValueError("wall_colors must contain at least one MiniGrid color")

    invalid_colors = [color for color in colors if color not in COLOR_NAMES]
    if invalid_colors:
        raise ValueError(
            f"wall_colors must only contain valid MiniGrid colors "
            f"{tuple(COLOR_NAMES)}; got {colors}"
        )

    return tuple(colors)


def _normalize_wall_object(wall_object):
    if wall_object is None:
        return "wall"

    wall_object = str(wall_object).lower()
    if wall_object not in ("wall", "wall2"):
        raise ValueError("wall_object must be either 'wall' or 'wall2'")
    return wall_object


def _normalize_task_set(task_set):
    if not isinstance(task_set, dict):
        raise ValueError("task sets must be dictionaries")

    name = task_set.get("name")
    if not name:
        raise ValueError("task sets must include a non-empty `name`")

    wall_object = _normalize_wall_object(task_set.get("wall_object"))
    wall_colors = _normalize_color_sequence(
        task_set.get("wall_colors", task_set.get("wall_color")),
        ("grey",),
    )

    return {
        "name": str(name),
        "door_colors": _normalize_door_colors(task_set.get("door_colors")),
        "wall_object": wall_object,
        "wall_colors": wall_colors,
    }


def _env_is_registered(env_id):
    try:
        return env_id in registry.env_specs
    except AttributeError:
        return env_id in registry


class CurriculumMultiRoomWallObjectEnv(_BaseCurriculumMultiRoomEnv):
    """Original CurriculumMultiRoomEnv fork with `wall`/`wall2` room walls."""

    def __init__(
        self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        targetNumRooms=2,
        layoutNumRooms=None,
        door_colors=None,
        wall_object="wall",
        wall_colors=None,
        **kwargs,
    ):
        wall_object = kwargs.pop("wall_object", wall_object)
        wall_colors = kwargs.pop("wall_colors", wall_colors)

        self.wall_object = _normalize_wall_object(wall_object)
        self.wall_colors = _normalize_color_sequence(wall_colors, ("grey",))

        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=maxRoomSize,
            targetNumRooms=targetNumRooms,
            layoutNumRooms=layoutNumRooms,
            door_colors=door_colors,
            **kwargs,
        )

    def _make_wall(self, color):
        if self.wall_object == "wall2":
            return Wall2(color)
        return Wall(color)

    def _gen_grid(self, width, height):
        roomList = self._generate_room_chain(width, height, self.layoutNumRooms)
        roomList = roomList[: self.targetNumRooms]

        assert len(roomList) > 0
        self.rooms = roomList

        self.grid = Grid(width, height)

        for idx, room in enumerate(roomList):
            topX, topY = room.top
            sizeX, sizeY = room.size
            wall_color = self.wall_colors[idx % len(self.wall_colors)]

            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, self._make_wall(wall_color))
                self.grid.set(topX + i, topY + sizeY - 1, self._make_wall(wall_color))

            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, self._make_wall(wall_color))
                self.grid.set(topX + sizeX - 1, topY + j, self._make_wall(wall_color))

            if idx > 0:
                door_color = self.door_colors[(idx - 1) % len(self.door_colors)]
                entry_door = Door(door_color)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], entry_door)

                prev_room = roomList[idx - 1]
                prev_room.exitDoorPos = room.entryDoorPos

        self.place_agent(roomList[0].top, roomList[0].size)
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "traverse the rooms to get to the goal"


def register_curriculum_wall_object(
    min_target_rooms=2,
    max_target_rooms=5,
    min_num_rooms=2,
    max_num_rooms=5,
    max_room_size=5,
    layout_num_rooms=5,
    task_sets=None,
    entry_point=(
        "CurriculumMinigrid.curriculumMultiRoomEnvWallObject:"
        "CurriculumMultiRoomWallObjectEnv"
    ),
):
    if layout_num_rooms is None:
        layout_num_rooms = max(max_num_rooms, max_target_rooms)

    if task_sets is None:
        task_sets = [
            {
                "name": "Wall",
                "door_colors": ["yellow"],
                "wall_object": "wall",
                "wall_colors": ["grey"],
            },
            {
                "name": "Wall2",
                "door_colors": ["blue"],
                "wall_object": "wall2",
                "wall_colors": ["purple"],
            },
        ]
    if not task_sets:
        raise ValueError("task_sets must contain at least one task set")

    normalized_task_sets = [_normalize_task_set(task_set) for task_set in task_sets]

    seen_env_ids = set()
    for task_set in normalized_task_sets:
        env_prefix = f"CurriculumMultiRoomWallObjectEnv-{task_set['name']}"

        for target_num_rooms in range(min_target_rooms, max_target_rooms + 1):
            env_id = f"{env_prefix}-R{target_num_rooms}-v0"
            if env_id in seen_env_ids:
                raise ValueError(f"duplicate env id generated for curriculum: {env_id}")
            seen_env_ids.add(env_id)

            if _env_is_registered(env_id):
                continue

            register(
                id=env_id,
                entry_point=entry_point,
                kwargs={
                    "minNumRooms": min_num_rooms,
                    "maxNumRooms": max_num_rooms,
                    "maxRoomSize": max_room_size,
                    "targetNumRooms": target_num_rooms,
                    "layoutNumRooms": layout_num_rooms,
                    "door_colors": task_set["door_colors"],
                    "wall_object": task_set["wall_object"],
                    "wall_colors": task_set["wall_colors"],
                },
            )


register_curriculum_wall_object()
