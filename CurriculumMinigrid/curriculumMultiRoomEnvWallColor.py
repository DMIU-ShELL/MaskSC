from gym.envs.registration import register, registry

from gym_minigrid.minigrid import COLOR_NAMES, Door, Goal, Grid, Wall

from .curriculumMultiRoomEnv import (
    CurriculumMultiRoomEnv as _BaseCurriculumMultiRoomEnv,
    _normalize_door_colors,
)


__all__ = ["CurriculumMultiRoomWallColorEnv", "register_curriculum_wall_color"]


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


def _default_task_set_name(door_colors, wall_colors):
    if door_colors == wall_colors:
        return "".join(color.capitalize() for color in wall_colors)
    door_name = "".join(color.capitalize() for color in door_colors)
    wall_name = "".join(color.capitalize() for color in wall_colors)
    return f"{wall_name}Walls{door_name}Doors"


def _normalize_task_set(task_set, task_set_idx):
    if isinstance(task_set, dict):
        raw_door_colors = task_set.get("door_colors", task_set.get("door_color"))
        raw_wall_colors = task_set.get("wall_colors", task_set.get("wall_color"))

        if raw_door_colors is None and raw_wall_colors is None:
            raise ValueError(
                "task set dictionaries must include `door_colors` or `wall_colors`"
            )
        if raw_door_colors is None:
            raw_door_colors = raw_wall_colors
        if raw_wall_colors is None:
            raw_wall_colors = raw_door_colors

        door_colors = _normalize_door_colors(raw_door_colors)
        wall_colors = _normalize_color_sequence(raw_wall_colors, door_colors)
        name = task_set.get("name")
        legacy_ids = task_set.get("legacy_ids", False)
    else:
        door_colors = _normalize_door_colors(task_set)
        wall_colors = _normalize_color_sequence(task_set, door_colors)
        name = None
        legacy_ids = task_set_idx == 0

    if legacy_ids and name:
        raise ValueError("task sets cannot specify both `name` and `legacy_ids=True`")

    if legacy_ids:
        name = ""
    elif name is None:
        name = _default_task_set_name(door_colors, wall_colors)

    return {
        "name": name,
        "door_colors": door_colors,
        "wall_colors": wall_colors,
    }


def _env_is_registered(env_id):
    try:
        return env_id in registry.env_specs
    except AttributeError:
        return env_id in registry


class CurriculumMultiRoomWallColorEnv(_BaseCurriculumMultiRoomEnv):
    """Original CurriculumMultiRoomEnv fork with configurable wall colors."""

    def __init__(
        self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        targetNumRooms=2,
        layoutNumRooms=None,
        door_colors=None,
        wall_colors=None,
        **kwargs,
    ):
        wall_colors = kwargs.pop("wall_colors", wall_colors)
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
                self.grid.set(topX + i, topY, Wall(wall_color))
                self.grid.set(topX + i, topY + sizeY - 1, Wall(wall_color))

            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, Wall(wall_color))
                self.grid.set(topX + sizeX - 1, topY + j, Wall(wall_color))

            if idx > 0:
                door_color = self.door_colors[(idx - 1) % len(self.door_colors)]
                entry_door = Door(door_color)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], entry_door)

                prev_room = roomList[idx - 1]
                prev_room.exitDoorPos = room.entryDoorPos

        self.place_agent(roomList[0].top, roomList[0].size)
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "traverse the rooms to get to the goal"


def register_curriculum_wall_color(
    min_target_rooms=2,
    max_target_rooms=11,
    min_num_rooms=2,
    max_num_rooms=10,
    max_room_size=5,
    layout_num_rooms=None,
    task_sets=None,
    entry_point=(
        "CurriculumMinigrid.curriculumMultiRoomEnvWallColor:"
        "CurriculumMultiRoomWallColorEnv"
    ),
):
    if layout_num_rooms is None:
        layout_num_rooms = max(max_num_rooms, max_target_rooms)

    if task_sets is None:
        task_sets = [
            {"name": "Yellow", "door_colors": ["yellow"], "wall_colors": ["yellow"]},
            {"name": "Blue", "door_colors": ["blue"], "wall_colors": ["blue"]},
        ]
    if not task_sets:
        raise ValueError("task_sets must contain at least one task set")

    normalized_task_sets = [
        _normalize_task_set(task_set, task_set_idx)
        for task_set_idx, task_set in enumerate(task_sets)
    ]

    seen_env_ids = set()
    for task_set in normalized_task_sets:
        env_prefix = "CurriculumMultiRoomWallColorEnv"
        if task_set["name"]:
            env_prefix = f"{env_prefix}-{task_set['name']}"

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
                    "wall_colors": task_set["wall_colors"],
                },
            )


register_curriculum_wall_color()
