from gym.envs.registration import register, registry

from gym_minigrid.minigrid import (
    COLOR_NAMES,
    Door,
    Floor,
    Goal,
    Grid,
    Wall,
)

from .curriculumMultiRoomEnvV2 import (
    CurriculumMultiRoomEnv as _BaseCurriculumMultiRoomEnv,
    _normalize_door_colors,
)


__all__ = ["CurriculumMultiRoomEnvV3", "register_curriculum_v3"]


def _normalize_color_sequence(colors, default_colors=(), allow_empty=False):
    if colors is None:
        colors = list(default_colors)
    elif isinstance(colors, str):
        clean = colors.replace(",", " ").split()
        colors = [token.lower() for token in clean] if clean else [colors.lower()]
    else:
        colors = [str(color).lower() for color in colors]

    if not colors and not allow_empty:
        raise ValueError("color sequence must contain at least one MiniGrid color")

    invalid_colors = [color for color in colors if color not in COLOR_NAMES]
    if invalid_colors:
        raise ValueError(
            f"color sequence must only contain valid MiniGrid colors "
            f"{tuple(COLOR_NAMES)}; got {colors}"
        )

    return tuple(colors)


def _env_is_registered(env_id):
    try:
        return env_id in registry.env_specs
    except AttributeError:
        return env_id in registry


class _VisualCurriculumMultiRoomEnv(_BaseCurriculumMultiRoomEnv):
    DEFAULT_WALL_COLORS = ("grey",)
    DEFAULT_TEXTURE_COLORS = ()
    DEFAULT_TEXTURE_PATTERN = "none"

    def __init__(
        self,
        minNumRooms=2,
        maxNumRooms=6,
        maxRoomSize=5,
        targetNumRooms=2,
        layoutNumRooms=None,
        door_colors=None,
        route_signature=None,
        grid_size=30,
        wall_colors=None,
        texture_colors=None,
        texture_pattern=None,
        **kwargs,
    ):
        wall_colors = kwargs.pop("wall_colors", wall_colors)
        texture_colors = kwargs.pop("texture_colors", texture_colors)
        texture_pattern = kwargs.pop("texture_pattern", texture_pattern)

        if wall_colors is None:
            wall_colors = self.DEFAULT_WALL_COLORS
        if texture_colors is None:
            texture_colors = self.DEFAULT_TEXTURE_COLORS
        if texture_pattern is None:
            texture_pattern = self.DEFAULT_TEXTURE_PATTERN

        self.wall_colors = _normalize_color_sequence(wall_colors, ("grey",))
        self.texture_colors = _normalize_color_sequence(
            texture_colors,
            (),
            allow_empty=True,
        )
        self.texture_pattern = str(texture_pattern).lower()

        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=maxRoomSize,
            targetNumRooms=targetNumRooms,
            layoutNumRooms=layoutNumRooms,
            door_colors=door_colors,
            route_signature=route_signature,
            grid_size=grid_size,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        roomList = self._generate_room_chain(width, height, self.layoutNumRooms)
        activeRooms = roomList[: self.targetNumRooms]

        if len(activeRooms) != self.targetNumRooms:
            raise RuntimeError(
                f"generated {len(activeRooms)} active rooms, expected {self.targetNumRooms}"
            )

        self.full_rooms = roomList
        self.rooms = activeRooms

        self.grid = Grid(width, height)

        for idx, room in enumerate(activeRooms):
            topX, topY = room.top
            sizeX, sizeY = room.size
            wallColor = self.wall_colors[idx % len(self.wall_colors)]

            for x in range(topX, topX + sizeX):
                self.grid.set(x, topY, Wall(wallColor))
                self.grid.set(x, topY + sizeY - 1, Wall(wallColor))

            for y in range(topY, topY + sizeY):
                self.grid.set(topX, y, Wall(wallColor))
                self.grid.set(topX + sizeX - 1, y, Wall(wallColor))

            if idx > 0:
                doorColor = self.door_colors[(idx - 1) % len(self.door_colors)]
                self.grid.set(
                    room.entryDoorPos[0],
                    room.entryDoorPos[1],
                    Door(doorColor),
                )

        self.place_agent(activeRooms[0].top, activeRooms[0].size)
        self.goal_pos = self.place_obj(Goal(), activeRooms[-1].top, activeRooms[-1].size)
        self._paint_room_textures(activeRooms)
        self.mission = "traverse the rooms to get to the goal"

    def _paint_room_textures(self, rooms):
        if not self.texture_colors or self.texture_pattern == "none":
            return

        for room_idx, room in enumerate(rooms):
            topX, topY = room.top
            sizeX, sizeY = room.size
            color = self.texture_colors[room_idx % len(self.texture_colors)]

            for x in range(topX + 1, topX + sizeX - 1):
                for y in range(topY + 1, topY + sizeY - 1):
                    if self.grid.get(x, y) is not None:
                        continue
                    local_x = x - (topX + 1)
                    local_y = y - (topY + 1)
                    if self._uses_texture_cell(local_x, local_y, room_idx):
                        self.grid.set(x, y, Floor(color))

    def _uses_texture_cell(self, local_x, local_y, room_idx):
        if self.texture_pattern == "solid":
            return True
        if self.texture_pattern == "checker_a":
            return (local_x + local_y + room_idx) % 2 == 0
        if self.texture_pattern == "checker_b":
            return (local_x + local_y + room_idx) % 2 == 1
        if self.texture_pattern == "vertical":
            return local_x % 2 == 0
        if self.texture_pattern == "horizontal":
            return local_y % 2 == 0
        if self.texture_pattern == "diagonal":
            return local_x == local_y
        if self.texture_pattern == "anti_diagonal":
            return local_x != local_y
        raise ValueError(
            "texture_pattern must be one of: none, solid, checker_a, checker_b, "
            "vertical, horizontal, diagonal, anti_diagonal"
        )


class CurriculumMultiRoomEnvV3(_VisualCurriculumMultiRoomEnv):
    """V3: family-specific wall colors, no floor texture by default."""


def _default_wall_color_families():
    return (
        {
            "name": "UpRight",
            "door_colors": ["yellow"],
            "route_signature": "up_right",
            "wall_colors": ["yellow"],
        },
        {
            "name": "DownLeft",
            "door_colors": ["blue"],
            "route_signature": "down_left",
            "wall_colors": ["blue"],
        },
    )


def _normalize_visual_family(family):
    if not isinstance(family, dict):
        raise ValueError("family specs must be dictionaries")
    if not family.get("name"):
        raise ValueError("family specs must include a non-empty `name`")

    spec = {
        "name": str(family["name"]),
        "door_colors": _normalize_door_colors(family.get("door_colors")),
        "route_signature": family.get("route_signature"),
    }

    if "wall_colors" in family:
        spec["wall_colors"] = _normalize_color_sequence(family["wall_colors"], ("grey",))
    if "texture_colors" in family:
        spec["texture_colors"] = _normalize_color_sequence(
            family["texture_colors"],
            (),
            allow_empty=True,
        )
    if "texture_pattern" in family:
        spec["texture_pattern"] = family["texture_pattern"]

    return spec


def _register_curriculum_variant(
    variant_name,
    entry_point,
    families,
    min_target_rooms=2,
    max_target_rooms=6,
    min_num_rooms=2,
    max_num_rooms=6,
    max_room_size=5,
    layout_num_rooms=6,
    grid_size=30,
):
    if layout_num_rooms is None:
        layout_num_rooms = max(max_num_rooms, max_target_rooms)

    seen_env_ids = set()
    for raw_family in families:
        family = _normalize_visual_family(raw_family)
        env_prefix = f"CurriculumMultiRoomEnv{variant_name}-{family['name']}"

        for target_num_rooms in range(min_target_rooms, max_target_rooms + 1):
            env_id = f"{env_prefix}-R{target_num_rooms}-v0"
            if env_id in seen_env_ids:
                raise ValueError(f"duplicate env id generated for curriculum: {env_id}")
            seen_env_ids.add(env_id)

            if _env_is_registered(env_id):
                continue

            kwargs = {
                "minNumRooms": min_num_rooms,
                "maxNumRooms": max_num_rooms,
                "maxRoomSize": max_room_size,
                "targetNumRooms": target_num_rooms,
                "layoutNumRooms": layout_num_rooms,
                "door_colors": family["door_colors"],
                "route_signature": family["route_signature"],
                "grid_size": grid_size,
            }
            for extra_key in ("wall_colors", "texture_colors", "texture_pattern"):
                if extra_key in family:
                    kwargs[extra_key] = family[extra_key]

            register(id=env_id, entry_point=entry_point, kwargs=kwargs)


def register_curriculum_v3(
    min_target_rooms=2,
    max_target_rooms=6,
    min_num_rooms=2,
    max_num_rooms=6,
    max_room_size=5,
    layout_num_rooms=6,
    grid_size=30,
    families=None,
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvV3:CurriculumMultiRoomEnvV3",
):
    if families is None:
        families = _default_wall_color_families()

    _register_curriculum_variant(
        "V3",
        entry_point,
        families,
        min_target_rooms=min_target_rooms,
        max_target_rooms=max_target_rooms,
        min_num_rooms=min_num_rooms,
        max_num_rooms=max_num_rooms,
        max_room_size=max_room_size,
        layout_num_rooms=layout_num_rooms,
        grid_size=grid_size,
    )


register_curriculum_v3()
