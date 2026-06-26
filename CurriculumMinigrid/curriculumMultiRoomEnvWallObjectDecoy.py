from gym.envs.registration import register

from gym_minigrid.minigrid import Door, Goal, Grid

from .curriculumMultiRoomEnv import _normalize_door_colors
from .curriculumMultiRoomEnvWallObject import (
    CurriculumMultiRoomWallObjectEnv,
    _env_is_registered,
    _normalize_task_set as _normalize_wall_object_task_set,
)


__all__ = [
    "CurriculumMultiRoomWallObjectDecoyEnv",
    "register_curriculum_wall_object_decoy",
]


WALL_RIGHT = 0
WALL_DOWN = 1
WALL_LEFT = 2
WALL_UP = 3

_WALL_TO_VECTOR = {
    WALL_RIGHT: (1, 0),
    WALL_DOWN: (0, 1),
    WALL_LEFT: (-1, 0),
    WALL_UP: (0, -1),
}


def _default_decoy_door_colors(door_colors):
    primary_color = _normalize_door_colors(door_colors)[0]
    if primary_color == "blue":
        return ("yellow",)
    return ("blue",)


def _normalize_decoy_task_set(task_set):
    base_task_set = _normalize_wall_object_task_set(task_set)
    raw_decoy_colors = task_set.get(
        "decoy_door_colors", task_set.get("decoy_door_color")
    )

    if raw_decoy_colors is None:
        decoy_door_colors = _default_decoy_door_colors(base_task_set["door_colors"])
    else:
        decoy_door_colors = _normalize_door_colors(raw_decoy_colors)

    return {
        **base_task_set,
        "decoy_door_colors": decoy_door_colors,
    }


class CurriculumMultiRoomWallObjectDecoyEnv(CurriculumMultiRoomWallObjectEnv):
    """
    Wall-object curriculum fork with wrong-colored dead-end corridor decoys.

    The real route is unchanged. Each non-final room tries to receive one
    closed decoy door, usually opposite the real exit, opening into a short
    one-cell-wide dead-end corridor.
    """

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
        decoy_door_colors=None,
        decoy_corridor_length=3,
        require_decoys=False,
        **kwargs,
    ):
        decoy_door_colors = kwargs.pop("decoy_door_colors", decoy_door_colors)
        decoy_corridor_length = kwargs.pop(
            "decoy_corridor_length", decoy_corridor_length
        )
        require_decoys = kwargs.pop("require_decoys", require_decoys)

        if decoy_door_colors is None:
            self.decoy_door_colors = _default_decoy_door_colors(door_colors)
        else:
            self.decoy_door_colors = _normalize_door_colors(decoy_door_colors)

        self.decoy_corridor_length = int(decoy_corridor_length)
        if self.decoy_corridor_length < 1:
            raise ValueError("decoy_corridor_length must be at least 1")

        self.require_decoys = bool(require_decoys)
        self.decoy_debug_info = []

        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=maxRoomSize,
            targetNumRooms=targetNumRooms,
            layoutNumRooms=layoutNumRooms,
            door_colors=door_colors,
            wall_object=wall_object,
            wall_colors=wall_colors,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        roomList = self._generate_room_chain(width, height, self.layoutNumRooms)
        roomList = roomList[: self.targetNumRooms]

        assert len(roomList) > 0
        self.rooms = roomList
        self.decoy_debug_info = []

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

        placed_decoys = self._add_decoy_corridors(roomList)
        required_decoys = max(0, len(roomList) - 1)
        if self.require_decoys and placed_decoys != required_decoys:
            raise RuntimeError(
                "Could not place all decoy corridors. Try increasing the grid size, "
                "reducing maxRoomSize, or shortening decoy_corridor_length."
            )

        self.place_agent(roomList[0].top, roomList[0].size)
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "traverse the rooms to get to the goal"

    def _add_decoy_corridors(self, roomList):
        placed = 0
        for room_idx, room in enumerate(roomList[:-1]):
            real_exit_pos = roomList[room_idx + 1].entryDoorPos
            real_exit_wall = self._door_wall_for_room(room, real_exit_pos)

            for wall in self._candidate_decoy_walls(real_exit_wall):
                for door_pos in self._candidate_door_positions(room, wall):
                    if door_pos == room.entryDoorPos or door_pos == real_exit_pos:
                        continue

                    if self._try_place_decoy_corridor(room_idx, wall, door_pos):
                        placed += 1
                        break
                else:
                    continue
                break

        return placed

    def _candidate_decoy_walls(self, real_exit_wall):
        preferred_wall = (real_exit_wall + 2) % 4
        walls = [preferred_wall]
        walls.extend(
            wall
            for wall in (WALL_RIGHT, WALL_DOWN, WALL_LEFT, WALL_UP)
            if wall not in (real_exit_wall, preferred_wall)
        )
        return walls

    def _candidate_door_positions(self, room, wall):
        topX, topY = room.top
        sizeX, sizeY = room.size

        if wall == WALL_RIGHT:
            x = topX + sizeX - 1
            return [(x, y) for y in self._middle_out_range(topY + 1, topY + sizeY - 1)]

        if wall == WALL_DOWN:
            y = topY + sizeY - 1
            return [(x, y) for x in self._middle_out_range(topX + 1, topX + sizeX - 1)]

        if wall == WALL_LEFT:
            x = topX
            return [(x, y) for y in self._middle_out_range(topY + 1, topY + sizeY - 1)]

        if wall == WALL_UP:
            y = topY
            return [(x, y) for x in self._middle_out_range(topX + 1, topX + sizeX - 1)]

        raise ValueError(f"unknown wall id: {wall}")

    def _middle_out_range(self, start, stop):
        values = list(range(start, stop))
        if not values:
            return values

        center = len(values) // 2
        ordered = [values[center]]
        for offset in range(1, len(values)):
            left = center - offset
            right = center + offset
            if left >= 0:
                ordered.append(values[left])
            if right < len(values):
                ordered.append(values[right])
        return ordered

    def _try_place_decoy_corridor(self, room_idx, wall, door_pos):
        door_obj = self.grid.get(door_pos[0], door_pos[1])
        if door_obj is None or getattr(door_obj, "type", None) == "door":
            return False

        corridor_cells, border_cells = self._decoy_corridor_geometry(wall, door_pos)
        required_empty_cells = corridor_cells + border_cells
        if not all(self._cell_is_empty(pos) for pos in required_empty_cells):
            return False

        decoy_door_color = self.decoy_door_colors[
            room_idx % len(self.decoy_door_colors)
        ]
        self.grid.set(door_pos[0], door_pos[1], Door(decoy_door_color))

        for border_pos in border_cells:
            self.grid.set(
                border_pos[0],
                border_pos[1],
                self._make_decoy_border(room_idx),
            )

        self.decoy_debug_info.append(
            {
                "room_idx": room_idx,
                "wall": wall,
                "door_pos": door_pos,
                "door_color": decoy_door_color,
                "corridor_cells": corridor_cells,
                "border_cells": border_cells,
            }
        )
        return True

    def _decoy_corridor_geometry(self, wall, door_pos):
        dx, dy = _WALL_TO_VECTOR[wall]
        door_x, door_y = door_pos

        corridor_cells = [
            (door_x + dx * step, door_y + dy * step)
            for step in range(1, self.decoy_corridor_length + 1)
        ]

        if dx:
            side_vectors = ((0, -1), (0, 1))
        else:
            side_vectors = ((-1, 0), (1, 0))

        border_cells = []
        for cell_x, cell_y in corridor_cells:
            for side_dx, side_dy in side_vectors:
                border_cells.append((cell_x + side_dx, cell_y + side_dy))

        end_cap = (
            door_x + dx * (self.decoy_corridor_length + 1),
            door_y + dy * (self.decoy_corridor_length + 1),
        )
        border_cells.append(end_cap)

        return corridor_cells, border_cells

    def _make_decoy_border(self, room_idx):
        wall_color = self.wall_colors[room_idx % len(self.wall_colors)]
        return self._make_wall(wall_color)

    def _cell_is_empty(self, pos):
        x, y = pos
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        if self._cell_is_inside_active_room(pos):
            return False
        return self.grid.get(x, y) is None

    def _cell_is_inside_active_room(self, pos):
        x, y = pos
        for room in self.rooms:
            topX, topY = room.top
            sizeX, sizeY = room.size
            if topX <= x < topX + sizeX and topY <= y < topY + sizeY:
                return True
        return False

    def _door_wall_for_room(self, room, door_pos):
        topX, topY = room.top
        sizeX, sizeY = room.size
        x, y = door_pos

        if x == topX + sizeX - 1:
            return WALL_RIGHT
        if y == topY + sizeY - 1:
            return WALL_DOWN
        if x == topX:
            return WALL_LEFT
        if y == topY:
            return WALL_UP

        raise ValueError(f"door position {door_pos} is not on room wall {room.top}")

    def get_decoy_debug_info(self):
        return list(self.decoy_debug_info)


def register_curriculum_wall_object_decoy(
    min_target_rooms=2,
    max_target_rooms=5,
    min_num_rooms=2,
    max_num_rooms=5,
    max_room_size=5,
    layout_num_rooms=5,
    decoy_corridor_length=3,
    require_decoys=False,
    task_sets=None,
    entry_point=(
        "CurriculumMinigrid.curriculumMultiRoomEnvWallObjectDecoy:"
        "CurriculumMultiRoomWallObjectDecoyEnv"
    ),
):
    if layout_num_rooms is None:
        layout_num_rooms = max(max_num_rooms, max_target_rooms)

    if task_sets is None:
        task_sets = [
            {
                "name": "Wall",
                "door_colors": ["yellow"],
                "decoy_door_colors": ["blue"],
                "wall_object": "wall",
                "wall_colors": ["grey"],
            },
            {
                "name": "Wall2",
                "door_colors": ["blue"],
                "decoy_door_colors": ["yellow"],
                "wall_object": "wall2",
                "wall_colors": ["purple"],
            },
        ]
    if not task_sets:
        raise ValueError("task_sets must contain at least one task set")

    normalized_task_sets = [_normalize_decoy_task_set(task_set) for task_set in task_sets]

    seen_env_ids = set()
    for task_set in normalized_task_sets:
        env_prefix = f"CurriculumMultiRoomWallObjectDecoyEnv-{task_set['name']}"

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
                    "decoy_door_colors": task_set["decoy_door_colors"],
                    "decoy_corridor_length": decoy_corridor_length,
                    "require_decoys": require_decoys,
                },
            )


register_curriculum_wall_object_decoy()
