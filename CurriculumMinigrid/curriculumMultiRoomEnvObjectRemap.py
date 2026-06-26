from gym.envs.registration import register, registry

from gym_minigrid.minigrid import (
    COLOR_NAMES,
    COLOR_TO_IDX,
    Door,
    Floor,
    Goal,
    Grid,
    OBJECT_TO_IDX,
    Wall,
)

from .curriculumMultiRoomEnv import (
    CurriculumMultiRoomEnv as _BaseCurriculumMultiRoomEnv,
    _normalize_door_colors,
)


__all__ = [
    "OBJECT_REMAP_PROFILES",
    "RemappedWall",
    "RemappedDoor",
    "RemappedFloor",
    "CurriculumMultiRoomObjectRemapEnv",
    "register_curriculum_object_remap",
]


ALLOWED_WALL_DOOR_OBS_OBJECTS = ("wall", "door", "key", "ball", "box", "lava")
ALLOWED_FLOOR_OBS_OBJECTS = ("empty", "floor", "key", "ball", "box", "lava")

# The defaults below are a small symbolic codebook: each profile gets a distinct
# wall, door, and passable-floor observation tuple while preserving dynamics.
OBJECT_REMAP_PROFILES = {
    "Identity": {
        "wall_observed_type": "wall",
        "wall_colors": ["grey"],
        "door_observed_type": "door",
        "door_observed_colors": None,
        "floor_observed_type": "empty",
        "floor_colors": ["blue"],
    },
    "IdentitySwap": {
        "wall_observed_type": "door",
        "wall_colors": ["blue"],
        "door_observed_type": "wall",
        "door_observed_colors": ["purple"],
        "floor_observed_type": "ball",
        "floor_colors": ["green"],
    },
    "BoxKey": {
        "wall_observed_type": "box",
        "wall_colors": ["red"],
        "door_observed_type": "key",
        "door_observed_colors": ["blue"],
        "floor_observed_type": "ball",
        "floor_colors": ["yellow"],
    },
    "KeyBoxSwap": {
        "wall_observed_type": "key",
        "wall_colors": ["yellow"],
        "door_observed_type": "box",
        "door_observed_colors": ["green"],
        "floor_observed_type": "floor",
        "floor_colors": ["grey"],
    },
    "BallLava": {
        "wall_observed_type": "ball",
        "wall_colors": ["purple"],
        "door_observed_type": "lava",
        "door_observed_colors": ["red"],
        "floor_observed_type": "box",
        "floor_colors": ["purple"],
    },
    "LavaBall": {
        "wall_observed_type": "lava",
        "wall_colors": ["green"],
        "door_observed_type": "ball",
        "door_observed_colors": ["blue"],
        "floor_observed_type": "key",
        "floor_colors": ["red"],
    },
}


def _env_is_registered(env_id):
    try:
        return env_id in registry.env_specs
    except AttributeError:
        return env_id in registry


def _normalize_color_sequence(colors, default_colors, field_name):
    if colors is None:
        colors = list(default_colors)
    elif isinstance(colors, str):
        clean = colors.replace(",", " ").split()
        colors = [token.lower() for token in clean] if clean else [colors.lower()]
    else:
        colors = [str(color).lower() for color in colors]

    if not colors:
        raise ValueError(f"{field_name} must contain at least one MiniGrid color")

    invalid_colors = [color for color in colors if color not in COLOR_NAMES]
    if invalid_colors:
        raise ValueError(
            f"{field_name} must only contain valid MiniGrid colors "
            f"{tuple(COLOR_NAMES)}; got {colors}"
        )

    return tuple(colors)


def _normalize_observed_type(
    observed_type, field_name, allowed_objects=ALLOWED_WALL_DOOR_OBS_OBJECTS
):
    if observed_type is None:
        raise ValueError(f"{field_name} cannot be None")

    observed_type = str(observed_type).lower()
    if observed_type not in OBJECT_TO_IDX:
        raise ValueError(
            f"{field_name} must be a known MiniGrid object type; got "
            f"{observed_type}"
        )
    if observed_type not in allowed_objects:
        raise ValueError(
            f"{field_name} must be one of {allowed_objects}; got "
            f"{observed_type}"
        )

    return observed_type


def _normalize_profile_name(remap_profile):
    if remap_profile is None:
        return "Identity"

    remap_profile = str(remap_profile)
    lookup_key = remap_profile.replace("-", "").replace("_", "").lower()
    for profile_name in OBJECT_REMAP_PROFILES:
        profile_key = profile_name.replace("-", "").replace("_", "").lower()
        if lookup_key == profile_key:
            return profile_name

    raise ValueError(
        f"unknown remap_profile {remap_profile!r}; valid profiles are "
        f"{tuple(OBJECT_REMAP_PROFILES)}"
    )


def _normalize_door_state_encoding(door_state_encoding):
    door_state_encoding = str(door_state_encoding).lower()
    valid_modes = ("preserve", "observed_default", "zero")
    if door_state_encoding not in valid_modes:
        raise ValueError(
            f"door_state_encoding must be one of {valid_modes}; got "
            f"{door_state_encoding}"
        )
    return door_state_encoding


def _door_state(is_open, is_locked):
    if is_open:
        return 0
    if is_locked:
        return 2
    return 1


def _observed_default_state(observed_type):
    if observed_type == "door":
        return 1
    return 0


def _encode_as(observed_type, observed_color, state):
    return (
        OBJECT_TO_IDX[observed_type],
        COLOR_TO_IDX[observed_color],
        state,
    )


class RemappedWall(Wall):
    """Wall-behaving object with a configurable symbolic observation type."""

    def __init__(self, observed_type="wall", observed_color="grey"):
        super().__init__("grey")
        self.observed_type = _normalize_observed_type(
            observed_type, "wall_observed_type"
        )
        self.observed_color = _normalize_color_sequence(
            [observed_color], ("grey",), "wall_colors"
        )[0]

    def encode(self):
        return _encode_as(
            self.observed_type,
            self.observed_color,
            _observed_default_state(self.observed_type),
        )


class RemappedDoor(Door):
    """Door-behaving object with a configurable symbolic observation type."""

    def __init__(
        self,
        color,
        observed_type="door",
        observed_color=None,
        door_state_encoding="preserve",
        is_open=False,
        is_locked=False,
    ):
        super().__init__(color, is_open=is_open, is_locked=is_locked)
        self.observed_type = _normalize_observed_type(
            observed_type, "door_observed_type"
        )
        if observed_color is None:
            observed_color = color
        self.observed_color = _normalize_color_sequence(
            [observed_color], (color,), "door_observed_colors"
        )[0]
        self.door_state_encoding = _normalize_door_state_encoding(
            door_state_encoding
        )

    def encode(self):
        if self.door_state_encoding == "preserve":
            state = _door_state(self.is_open, self.is_locked)
        elif self.door_state_encoding == "observed_default":
            state = _observed_default_state(self.observed_type)
        else:
            state = 0

        return _encode_as(self.observed_type, self.observed_color, state)


class RemappedFloor(Floor):
    """Passable floor tile with a configurable symbolic observation type."""

    def __init__(self, observed_type="floor", observed_color="grey"):
        observed_color = _normalize_color_sequence(
            [observed_color], ("grey",), "floor_colors"
        )[0]
        super().__init__(observed_color)
        self.observed_type = _normalize_observed_type(
            observed_type,
            "floor_observed_type",
            allowed_objects=ALLOWED_FLOOR_OBS_OBJECTS,
        )
        self.observed_color = observed_color

    def encode(self):
        return _encode_as(
            self.observed_type,
            self.observed_color,
            _observed_default_state(self.observed_type),
        )


def _profile_defaults(remap_profile):
    profile_name = _normalize_profile_name(remap_profile)
    return profile_name, dict(OBJECT_REMAP_PROFILES[profile_name])


def _normalize_task_set(task_set):
    if not isinstance(task_set, dict):
        task_set = {"remap_profile": task_set}

    profile_name, defaults = _profile_defaults(task_set.get("remap_profile"))
    name = task_set.get("name", profile_name)
    if not name:
        raise ValueError("task sets must include a non-empty `name`")

    door_colors = _normalize_door_colors(task_set.get("door_colors"))
    wall_observed_type = _normalize_observed_type(
        task_set.get("wall_observed_type", defaults["wall_observed_type"]),
        "wall_observed_type",
    )
    wall_colors = _normalize_color_sequence(
        task_set.get("wall_colors", defaults["wall_colors"]),
        ("grey",),
        "wall_colors",
    )
    door_observed_type = _normalize_observed_type(
        task_set.get("door_observed_type", defaults["door_observed_type"]),
        "door_observed_type",
    )
    door_observed_color_defaults = defaults["door_observed_colors"]
    if door_observed_color_defaults is None:
        door_observed_color_defaults = door_colors
    door_observed_colors = _normalize_color_sequence(
        task_set.get("door_observed_colors", task_set.get("door_observed_color")),
        door_observed_color_defaults,
        "door_observed_colors",
    )
    door_state_encoding = _normalize_door_state_encoding(
        task_set.get("door_state_encoding", "preserve")
    )
    floor_observed_type = _normalize_observed_type(
        task_set.get("floor_observed_type", defaults["floor_observed_type"]),
        "floor_observed_type",
        allowed_objects=ALLOWED_FLOOR_OBS_OBJECTS,
    )
    floor_colors = _normalize_color_sequence(
        task_set.get(
            "floor_colors",
            task_set.get("floor_observed_colors", defaults["floor_colors"]),
        ),
        ("grey",),
        "floor_colors",
    )

    return {
        "name": str(name),
        "remap_profile": profile_name,
        "door_colors": door_colors,
        "wall_observed_type": wall_observed_type,
        "wall_colors": wall_colors,
        "door_observed_type": door_observed_type,
        "door_observed_colors": door_observed_colors,
        "door_state_encoding": door_state_encoding,
        "floor_observed_type": floor_observed_type,
        "floor_colors": floor_colors,
    }


class CurriculumMultiRoomObjectRemapEnv(_BaseCurriculumMultiRoomEnv):
    """
    MultiRoom fork with family-specific object-symbol remappings.

    Physical behavior is unchanged: true walls block movement, true doors still
    toggle/open, and remapped floor cells are passable. Only the symbolic
    observation emitted by walls, doors, and floors is remapped.
    """

    def __init__(
        self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        targetNumRooms=2,
        layoutNumRooms=None,
        door_colors=None,
        remap_profile="Identity",
        wall_observed_type=None,
        wall_colors=None,
        door_observed_type=None,
        door_observed_colors=None,
        door_state_encoding="preserve",
        floor_observed_type=None,
        floor_colors=None,
        floor_observed_colors=None,
        **kwargs,
    ):
        remap_profile = kwargs.pop("remap_profile", remap_profile)
        wall_observed_type = kwargs.pop("wall_observed_type", wall_observed_type)
        wall_colors = kwargs.pop("wall_colors", wall_colors)
        door_observed_type = kwargs.pop("door_observed_type", door_observed_type)
        door_observed_colors = kwargs.pop(
            "door_observed_colors", door_observed_colors
        )
        door_state_encoding = kwargs.pop(
            "door_state_encoding", door_state_encoding
        )
        floor_observed_type = kwargs.pop("floor_observed_type", floor_observed_type)
        if "floor_colors" in kwargs:
            floor_colors = kwargs.pop("floor_colors")
            kwargs.pop("floor_observed_colors", None)
        else:
            floor_colors = kwargs.pop(
                "floor_observed_colors", floor_observed_colors or floor_colors
            )

        profile_name, defaults = _profile_defaults(remap_profile)
        self.remap_profile = profile_name

        if wall_observed_type is None:
            wall_observed_type = defaults["wall_observed_type"]
        if wall_colors is None:
            wall_colors = defaults["wall_colors"]
        if door_observed_type is None:
            door_observed_type = defaults["door_observed_type"]
        if door_colors is None:
            door_colors = ["yellow"]
        if door_observed_colors is None:
            door_observed_colors = defaults["door_observed_colors"]
        if door_observed_colors is None:
            door_observed_colors = door_colors
        if floor_observed_type is None:
            floor_observed_type = defaults["floor_observed_type"]
        if floor_colors is None:
            floor_colors = defaults["floor_colors"]

        self.wall_observed_type = _normalize_observed_type(
            wall_observed_type, "wall_observed_type"
        )
        self.wall_colors = _normalize_color_sequence(
            wall_colors, ("grey",), "wall_colors"
        )
        self.door_observed_type = _normalize_observed_type(
            door_observed_type, "door_observed_type"
        )
        self.door_observed_colors = _normalize_color_sequence(
            door_observed_colors,
            _normalize_door_colors(door_colors),
            "door_observed_colors",
        )
        self.door_state_encoding = _normalize_door_state_encoding(
            door_state_encoding
        )
        self.floor_observed_type = _normalize_observed_type(
            floor_observed_type,
            "floor_observed_type",
            allowed_objects=ALLOWED_FLOOR_OBS_OBJECTS,
        )
        self.floor_colors = _normalize_color_sequence(
            floor_colors, ("grey",), "floor_colors"
        )

        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=maxRoomSize,
            targetNumRooms=targetNumRooms,
            layoutNumRooms=layoutNumRooms,
            door_colors=door_colors,
            **kwargs,
        )

    def _make_wall(self, observed_color):
        return RemappedWall(
            observed_type=self.wall_observed_type,
            observed_color=observed_color,
        )

    def _make_door(self, door_color, observed_color):
        return RemappedDoor(
            color=door_color,
            observed_type=self.door_observed_type,
            observed_color=observed_color,
            door_state_encoding=self.door_state_encoding,
        )

    def _make_floor(self, observed_color):
        return RemappedFloor(
            observed_type=self.floor_observed_type,
            observed_color=observed_color,
        )

    def _paint_room_floors(self, roomList):
        for idx, room in enumerate(roomList):
            topX, topY = room.top
            sizeX, sizeY = room.size
            floor_color = self.floor_colors[idx % len(self.floor_colors)]

            for x in range(topX + 1, topX + sizeX - 1):
                for y in range(topY + 1, topY + sizeY - 1):
                    if self.grid.get(x, y) is None:
                        self.grid.set(x, y, self._make_floor(floor_color))

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
                door_observed_color = self.door_observed_colors[
                    (idx - 1) % len(self.door_observed_colors)
                ]
                entry_door = self._make_door(door_color, door_observed_color)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], entry_door)

                prev_room = roomList[idx - 1]
                prev_room.exitDoorPos = room.entryDoorPos

        self.place_agent(roomList[0].top, roomList[0].size)
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)
        self._paint_room_floors(roomList)

        self.mission = "traverse the rooms to get to the goal"

    def get_object_remap_debug_info(self):
        return {
            "remap_profile": self.remap_profile,
            "wall_observed_type": self.wall_observed_type,
            "wall_colors": self.wall_colors,
            "door_observed_type": self.door_observed_type,
            "door_observed_colors": self.door_observed_colors,
            "door_state_encoding": self.door_state_encoding,
            "floor_observed_type": self.floor_observed_type,
            "floor_colors": self.floor_colors,
        }

    def get_room_debug_info(self):
        return [
            {
                "idx": idx,
                "top": room.top,
                "size": room.size,
                "entryDoorPos": room.entryDoorPos,
                "exitDoorPos": room.exitDoorPos,
            }
            for idx, room in enumerate(self.rooms)
        ]


def register_curriculum_object_remap(
    min_target_rooms=2,
    max_target_rooms=5,
    min_num_rooms=2,
    max_num_rooms=5,
    max_room_size=5,
    layout_num_rooms=5,
    task_sets=None,
    entry_point=(
        "CurriculumMinigrid.curriculumMultiRoomEnvObjectRemap:"
        "CurriculumMultiRoomObjectRemapEnv"
    ),
):
    if layout_num_rooms is None:
        layout_num_rooms = max(max_num_rooms, max_target_rooms)

    if task_sets is None:
        task_sets = [
            {"name": "Identity", "remap_profile": "Identity"},
            {"name": "IdentitySwap", "remap_profile": "IdentitySwap"},
            {"name": "BoxKey", "remap_profile": "BoxKey"},
            {"name": "KeyBoxSwap", "remap_profile": "KeyBoxSwap"},
            {"name": "BallLava", "remap_profile": "BallLava"},
            {"name": "LavaBall", "remap_profile": "LavaBall"},
        ]
    if not task_sets:
        raise ValueError("task_sets must contain at least one task set")

    normalized_task_sets = [_normalize_task_set(task_set) for task_set in task_sets]

    seen_env_ids = set()
    for task_set in normalized_task_sets:
        env_prefix = f"CurriculumMultiRoomObjectRemapEnv-{task_set['name']}"

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
                    "remap_profile": task_set["remap_profile"],
                    "wall_observed_type": task_set["wall_observed_type"],
                    "wall_colors": task_set["wall_colors"],
                    "door_observed_type": task_set["door_observed_type"],
                    "door_observed_colors": task_set["door_observed_colors"],
                    "door_state_encoding": task_set["door_state_encoding"],
                    "floor_observed_type": task_set["floor_observed_type"],
                    "floor_colors": task_set["floor_colors"],
                },
            )


register_curriculum_object_remap()
