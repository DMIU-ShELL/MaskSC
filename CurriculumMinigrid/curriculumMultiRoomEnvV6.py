from gym.envs.registration import register

from gym_minigrid.minigrid import Ball, Box, Door, Goal, Grid, Wall

from .curriculumMultiRoomEnvV2 import _normalize_door_colors
from .curriculumMultiRoomEnvV3 import (
    _VisualCurriculumMultiRoomEnv,
    _env_is_registered,
    _normalize_color_sequence,
)


__all__ = ["CurriculumMultiRoomEnvV6", "register_curriculum_v6"]


VALID_DECOR_TYPES = ("ball", "box", "wall_stub")
VALID_DECOR_POSITIONS = ("nw", "ne", "sw", "se")


def _normalize_string_sequence(values, default_values, valid_values, name):
    if values is None:
        values = list(default_values)
    elif isinstance(values, str):
        clean = values.replace(",", " ").split()
        values = [token.lower() for token in clean] if clean else [values.lower()]
    else:
        values = [str(value).lower() for value in values]

    if not values:
        raise ValueError(f"{name} must contain at least one value")

    invalid_values = [value for value in values if value not in valid_values]
    if invalid_values:
        raise ValueError(
            f"{name} must only contain values from {valid_values}; got {values}"
        )

    return tuple(values)


def _make_decor_object(decor_type, color):
    if decor_type == "ball":
        return Ball(color)
    if decor_type == "box":
        return Box(color)
    if decor_type == "wall_stub":
        return Wall(color)
    raise ValueError(f"unknown decor_type: {decor_type}")


class CurriculumMultiRoomEnvV6(_VisualCurriculumMultiRoomEnv):
    """
    V6: family-specific existing-object room texture.

    A deterministic decorative object is placed in each active room. The object
    is never part of the objective and is placed before agent/goal placement so
    MiniGrid's normal placement helpers avoid it.
    """

    DEFAULT_DECOR_TYPES = ("ball",)
    DEFAULT_DECOR_COLORS = ("green",)
    DEFAULT_DECOR_POSITIONS = ("nw",)

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
        decor_types=None,
        decor_colors=None,
        decor_positions=None,
        **kwargs,
    ):
        decor_types = kwargs.pop("decor_types", decor_types)
        decor_colors = kwargs.pop("decor_colors", decor_colors)
        decor_positions = kwargs.pop("decor_positions", decor_positions)

        self.decor_types = _normalize_string_sequence(
            decor_types,
            self.DEFAULT_DECOR_TYPES,
            VALID_DECOR_TYPES,
            "decor_types",
        )
        self.decor_colors = _normalize_color_sequence(
            decor_colors,
            self.DEFAULT_DECOR_COLORS,
        )
        self.decor_positions = _normalize_string_sequence(
            decor_positions,
            self.DEFAULT_DECOR_POSITIONS,
            VALID_DECOR_POSITIONS,
            "decor_positions",
        )

        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=maxRoomSize,
            targetNumRooms=targetNumRooms,
            layoutNumRooms=layoutNumRooms,
            door_colors=door_colors,
            route_signature=route_signature,
            grid_size=grid_size,
            wall_colors=wall_colors,
            texture_colors=(),
            texture_pattern="none",
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

        self._place_room_decorations(activeRooms)
        self.place_agent(activeRooms[0].top, activeRooms[0].size)
        self.goal_pos = self.place_obj(Goal(), activeRooms[-1].top, activeRooms[-1].size)
        self.mission = "traverse the rooms to get to the goal"

    def _place_room_decorations(self, rooms):
        for room_idx, room in enumerate(rooms):
            decor_type = self.decor_types[room_idx % len(self.decor_types)]
            color = self.decor_colors[room_idx % len(self.decor_colors)]
            position = self.decor_positions[room_idx % len(self.decor_positions)]
            pos = self._safe_decor_position(room, position)
            if pos is not None:
                self.grid.set(pos[0], pos[1], _make_decor_object(decor_type, color))

    def _safe_decor_position(self, room, preferred_position):
        protected = self._door_adjacent_cells(room)
        candidates = [self._decor_position(room, preferred_position)]
        candidates.extend(self._interior_cells_by_corner_preference(room))

        seen = set()
        for pos in candidates:
            if pos in seen:
                continue
            seen.add(pos)
            if pos in protected:
                continue
            if self.grid.get(pos[0], pos[1]) is not None:
                continue
            return pos

        return None

    def _door_adjacent_cells(self, room):
        topX, topY = room.top
        sizeX, sizeY = room.size
        protected = set()

        for door_pos in (room.entryDoorPos, room.exitDoorPos):
            if door_pos is None:
                continue
            doorX, doorY = door_pos
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x = doorX + dx
                y = doorY + dy
                if topX < x < topX + sizeX - 1 and topY < y < topY + sizeY - 1:
                    protected.add((x, y))

        return protected

    def _interior_cells_by_corner_preference(self, room):
        topX, topY = room.top
        sizeX, sizeY = room.size

        corner_order = (
            (topX + 1, topY + 1),
            (topX + sizeX - 2, topY + sizeY - 2),
            (topX + sizeX - 2, topY + 1),
            (topX + 1, topY + sizeY - 2),
        )
        cells = list(corner_order)

        for y in range(topY + 1, topY + sizeY - 1):
            for x in range(topX + 1, topX + sizeX - 1):
                cells.append((x, y))

        return cells

    def _decor_position(self, room, position):
        topX, topY = room.top
        sizeX, sizeY = room.size

        x_lo = topX + 1
        x_hi = topX + sizeX - 2
        y_lo = topY + 1
        y_hi = topY + sizeY - 2

        if position == "nw":
            return (x_lo, y_lo)
        if position == "ne":
            return (x_hi, y_lo)
        if position == "sw":
            return (x_lo, y_hi)
        if position == "se":
            return (x_hi, y_hi)
        raise ValueError(f"unknown decor position: {position}")


def _normalize_decor_family(family):
    if not isinstance(family, dict):
        raise ValueError("family specs must be dictionaries")
    if not family.get("name"):
        raise ValueError("family specs must include a non-empty `name`")

    spec = {
        "name": str(family["name"]),
        "door_colors": _normalize_door_colors(family.get("door_colors")),
        "route_signature": family.get("route_signature"),
        "wall_colors": _normalize_color_sequence(family.get("wall_colors"), ("grey",)),
        "decor_types": _normalize_string_sequence(
            family.get("decor_types"),
            ("ball",),
            VALID_DECOR_TYPES,
            "decor_types",
        ),
        "decor_colors": _normalize_color_sequence(
            family.get("decor_colors"),
            ("green",),
        ),
        "decor_positions": _normalize_string_sequence(
            family.get("decor_positions"),
            ("nw",),
            VALID_DECOR_POSITIONS,
            "decor_positions",
        ),
    }
    return spec


def _register_decor_curriculum_variant(
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
        family = _normalize_decor_family(raw_family)
        env_prefix = f"CurriculumMultiRoomEnv{variant_name}-{family['name']}"

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
                    "door_colors": family["door_colors"],
                    "route_signature": family["route_signature"],
                    "grid_size": grid_size,
                    "wall_colors": family["wall_colors"],
                    "decor_types": family["decor_types"],
                    "decor_colors": family["decor_colors"],
                    "decor_positions": family["decor_positions"],
                },
            )


def _default_decor_families():
    return (
        {
            "name": "UpRight",
            "door_colors": ["yellow"],
            "route_signature": "up_right",
            "wall_colors": ["grey"],
            "decor_types": ["ball"],
            "decor_colors": ["green"],
            "decor_positions": ["nw"],
        },
        {
            "name": "DownLeft",
            "door_colors": ["blue"],
            "route_signature": "down_left",
            "wall_colors": ["grey"],
            "decor_types": ["box"],
            "decor_colors": ["red"],
            "decor_positions": ["se"],
        },
    )


def register_curriculum_v6(
    min_target_rooms=2,
    max_target_rooms=6,
    min_num_rooms=2,
    max_num_rooms=6,
    max_room_size=5,
    layout_num_rooms=6,
    grid_size=30,
    families=None,
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvV6:CurriculumMultiRoomEnvV6",
):
    if families is None:
        families = _default_decor_families()

    _register_decor_curriculum_variant(
        "V6",
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


register_curriculum_v6()
