from gym.envs.registration import register, registry

from gym_minigrid.minigrid import (
    COLOR_NAMES,
    Door,
    Goal,
    Grid,
    MiniGridEnv,
    MissionSpace,
    Wall,
)
from gym_minigrid.wrappers import ImgObsWrapper, ReseedWrapper  # noqa: F401


WALL_RIGHT = 0
WALL_DOWN = 1
WALL_LEFT = 2
WALL_UP = 3

WALL_IDS = (WALL_RIGHT, WALL_DOWN, WALL_LEFT, WALL_UP)
OPPOSITE_WALL = {
    WALL_RIGHT: WALL_LEFT,
    WALL_DOWN: WALL_UP,
    WALL_LEFT: WALL_RIGHT,
    WALL_UP: WALL_DOWN,
}
ROUTE_SIGNATURES = {
    "up_right": (WALL_UP, WALL_RIGHT),
    "down_left": (WALL_DOWN, WALL_LEFT),
}


class MultiRoom:
    def __init__(self, top, size, entryDoorPos, exitDoorPos):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos


def _normalize_door_colors(door_colors):
    if door_colors is None:
        colors = ["yellow"]
    elif isinstance(door_colors, str):
        clean = door_colors.replace(",", " ").split()
        colors = [token.lower() for token in clean] if clean else [door_colors.lower()]
    else:
        colors = [str(color).lower() for color in door_colors]

    if not colors:
        raise ValueError("door_colors must contain at least one MiniGrid color")

    invalid_colors = [color for color in colors if color not in COLOR_NAMES]
    if invalid_colors:
        raise ValueError(
            f"door_colors must only contain valid MiniGrid colors {tuple(COLOR_NAMES)}; "
            f"got {colors}"
        )

    return tuple(colors)


def _normalize_route_signature(route_signature):
    if route_signature is None:
        return None

    if isinstance(route_signature, str):
        key = route_signature.strip().lower().replace("-", "_").replace(" ", "_")
        if key in ROUTE_SIGNATURES:
            walls = ROUTE_SIGNATURES[key]
        else:
            clean = route_signature.replace(",", " ").split()
            walls = tuple(int(token) for token in clean)
    else:
        walls = tuple(int(wall) for wall in route_signature)

    if not walls:
        raise ValueError("route_signature must contain at least one wall id")

    invalid_walls = [wall for wall in walls if wall not in WALL_IDS]
    if invalid_walls:
        raise ValueError(
            f"route_signature wall ids must be in {WALL_IDS}; got {tuple(walls)}"
        )

    return tuple(walls)


def _default_family_specs():
    return (
        {
            "name": "UpRight",
            "door_colors": ["yellow"],
            "route_signature": "up_right",
        },
        {
            "name": "DownLeft",
            "door_colors": ["blue"],
            "route_signature": "down_left",
        },
    )


def _normalize_family_spec(family):
    if not isinstance(family, dict):
        raise ValueError("family specs must be dictionaries")

    name = family.get("name")
    if not name:
        raise ValueError("family specs must include a non-empty `name`")

    return {
        "name": str(name),
        "door_colors": _normalize_door_colors(family.get("door_colors")),
        "route_signature": family.get("route_signature"),
    }


def _normalize_legacy_task_set(task_set, task_set_idx):
    if isinstance(task_set, dict):
        raw_colors = task_set.get("door_colors", task_set.get("door_color"))
        if raw_colors is None:
            raise ValueError(
                "task set dictionaries must include `door_colors` or `door_color`"
            )
        door_colors = _normalize_door_colors(raw_colors)
        name = task_set.get("name")
        legacy_ids = task_set.get("legacy_ids", task_set_idx == 0 and name is None)
        route_signature = task_set.get("route_signature")
    else:
        door_colors = _normalize_door_colors(task_set)
        name = None
        legacy_ids = task_set_idx == 0
        route_signature = None

    if legacy_ids and name:
        raise ValueError("task sets cannot specify both `name` and `legacy_ids=True`")

    if legacy_ids:
        name = ""
    elif name is None:
        name = "".join(color.capitalize() for color in door_colors)

    return {
        "name": name,
        "door_colors": door_colors,
        "route_signature": route_signature,
    }


class CurriculumMultiRoomEnv(MiniGridEnv):
    """
    Hierarchical ZigZag MultiRoom benchmark for sparse-reward lifelong RL.

    The environment keeps the standard MiniGrid action and observation spaces.
    Tasks differ only in the generated room-chain state distribution. A deeper
    task uses the prefix hierarchy of the previous task plus one extra room.
    """

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
        **kwargs,
    ):
        route_signature = kwargs.pop("route_signature", route_signature)
        grid_size = kwargs.pop("grid_size", grid_size)

        if minNumRooms <= 0:
            raise ValueError("minNumRooms must be > 0")
        if maxNumRooms < minNumRooms:
            raise ValueError("maxNumRooms must be >= minNumRooms")
        if maxRoomSize < 4:
            raise ValueError("maxRoomSize must be >= 4")
        if targetNumRooms < 2:
            raise ValueError("targetNumRooms must be >= 2")
        if grid_size < maxRoomSize:
            raise ValueError("grid_size must be >= maxRoomSize")

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize
        self.targetNumRooms = targetNumRooms
        if layoutNumRooms is None:
            layoutNumRooms = max(maxNumRooms, targetNumRooms)
        if layoutNumRooms < targetNumRooms:
            raise ValueError("layoutNumRooms must be >= targetNumRooms")
        self.layoutNumRooms = layoutNumRooms
        self.door_colors = _normalize_door_colors(door_colors)
        self.route_signature = _normalize_route_signature(route_signature)
        self.grid_size = grid_size

        self.rooms = []
        self.full_rooms = []

        mission_space = MissionSpace(
            mission_func=lambda: "traverse the rooms to get to the goal"
        )

        super().__init__(
            mission_space=mission_space,
            width=self.grid_size,
            height=self.grid_size,
            max_steps=self.targetNumRooms * 20,
            **kwargs,
        )

    def _reward(self):
        return 1

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
        wall = Wall()

        for idx, room in enumerate(activeRooms):
            topX, topY = room.top
            sizeX, sizeY = room.size

            for x in range(topX, topX + sizeX):
                self.grid.set(x, topY, wall)
                self.grid.set(x, topY + sizeY - 1, wall)

            for y in range(topY, topY + sizeY):
                self.grid.set(topX, y, wall)
                self.grid.set(topX + sizeX - 1, y, wall)

            if idx > 0:
                doorColor = self.door_colors[(idx - 1) % len(self.door_colors)]
                self.grid.set(
                    room.entryDoorPos[0],
                    room.entryDoorPos[1],
                    Door(doorColor),
                )

        self.place_agent(activeRooms[0].top, activeRooms[0].size)
        self.goal_pos = self.place_obj(Goal(), activeRooms[-1].top, activeRooms[-1].size)
        self.mission = "traverse the rooms to get to the goal"

    def _generate_room_chain(self, width, height, numRooms, max_attempts=1000):
        best_len = 0

        for _ in range(max_attempts):
            curRoomList = []
            entryDoorPos = self._sample_initial_entry_pos(width, height)

            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=WALL_LEFT,
                entryDoorPos=entryDoorPos,
                width=width,
                height=height,
            )

            if len(curRoomList) == numRooms:
                return curRoomList
            best_len = max(best_len, len(curRoomList))

        raise RuntimeError(
            "Could not generate a CurriculumMultiRoomEnv room chain with "
            f"{numRooms} rooms after {max_attempts} attempts "
            f"(best attempt placed {best_len} rooms). Try increasing grid_size, "
            "decreasing maxRoomSize, or reducing target/layout rooms."
        )

    def _sample_initial_entry_pos(self, width, height):
        min_room_size = 4
        max_x = max(0, width - self.maxRoomSize)
        max_y = max(0, height - self.maxRoomSize)

        if self.route_signature == ROUTE_SIGNATURES["up_right"]:
            x_hi = max(0, min(max_x, width // 3))
            y_lo = max(0, min(max_y, (2 * height) // 3 - self.maxRoomSize))
            return (
                self._rand_int(0, x_hi + 1),
                self._rand_int(y_lo, max_y + 1),
            )

        if self.route_signature == ROUTE_SIGNATURES["down_left"]:
            x_lo = max(0, min(max_x, (2 * width) // 3 - self.maxRoomSize))
            y_hi = max(0, min(max_y, height // 3))
            return (
                self._rand_int(x_lo, max_x + 1),
                self._rand_int(0, y_hi + 1),
            )

        return (
            self._rand_int(0, max(1, width - min_room_size + 1)),
            self._rand_int(0, max(1, height - min_room_size + 1)),
        )

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos,
        width,
        height,
    ):
        sizeX = self._rand_int(minSz, maxSz + 1)
        sizeY = self._rand_int(minSz, maxSz + 1)

        if len(roomList) == 0:
            topX, topY = entryDoorPos
        elif entryDoorWall == WALL_RIGHT:
            topX = entryDoorPos[0] - sizeX + 1
            topY = self._rand_int(entryDoorPos[1] - sizeY + 2, entryDoorPos[1])
        elif entryDoorWall == WALL_DOWN:
            topX = self._rand_int(entryDoorPos[0] - sizeX + 2, entryDoorPos[0])
            topY = entryDoorPos[1] - sizeY + 1
        elif entryDoorWall == WALL_LEFT:
            topX = entryDoorPos[0]
            topY = self._rand_int(entryDoorPos[1] - sizeY + 2, entryDoorPos[1])
        elif entryDoorWall == WALL_UP:
            topX = self._rand_int(entryDoorPos[0] - sizeX + 2, entryDoorPos[0])
            topY = entryDoorPos[1]
        else:
            raise ValueError(f"invalid entryDoorWall: {entryDoorWall}")

        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > width or topY + sizeY > height:
            return False

        for room in roomList[:-1]:
            nonOverlap = (
                topX + sizeX <= room.top[0]
                or room.top[0] + room.size[0] <= topX
                or topY + sizeY <= room.top[1]
                or room.top[1] + room.size[1] <= topY
            )
            if not nonOverlap:
                return False

        roomList.append(MultiRoom((topX, topY), (sizeX, sizeY), entryDoorPos, None))

        if numLeft == 1:
            return True

        transition_idx = len(roomList) - 1
        for _ in range(8):
            exitDoorWall = self._select_exit_wall(transition_idx, entryDoorWall)
            if exitDoorWall is None:
                break

            exitDoorPos = self._sample_exit_door_pos(
                topX,
                topY,
                sizeX,
                sizeY,
                exitDoorWall,
            )
            nextEntryWall = OPPOSITE_WALL[exitDoorWall]
            roomList[-1].exitDoorPos = exitDoorPos

            prevLen = len(roomList)
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos,
                width=width,
                height=height,
            )

            if success:
                return True

            del roomList[prevLen:]

        roomList[-1].exitDoorPos = None
        return False

    def _select_exit_wall(self, transition_idx, entryDoorWall):
        if self.route_signature is None:
            wallSet = set(WALL_IDS)
            wallSet.remove(entryDoorWall)
            return self._rand_elem(sorted(wallSet))

        exitDoorWall = self.route_signature[transition_idx % len(self.route_signature)]
        if exitDoorWall == entryDoorWall:
            return None
        return exitDoorWall

    def _sample_exit_door_pos(self, topX, topY, sizeX, sizeY, exitDoorWall):
        if exitDoorWall == WALL_RIGHT:
            return (topX + sizeX - 1, topY + self._rand_int(1, sizeY - 1))
        if exitDoorWall == WALL_DOWN:
            return (topX + self._rand_int(1, sizeX - 1), topY + sizeY - 1)
        if exitDoorWall == WALL_LEFT:
            return (topX, topY + self._rand_int(1, sizeY - 1))
        if exitDoorWall == WALL_UP:
            return (topX + self._rand_int(1, sizeX - 1), topY)
        raise ValueError(f"invalid exitDoorWall: {exitDoorWall}")

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


def _env_is_registered(env_id):
    try:
        return env_id in registry.env_specs
    except AttributeError:
        return env_id in registry


def register_curriculum(
    min_target_rooms=2,
    max_target_rooms=6,
    min_num_rooms=2,
    max_num_rooms=6,
    max_room_size=5,
    layout_num_rooms=6,
    grid_size=30,
    families=None,
    task_sets=None,
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvV2:CurriculumMultiRoomEnv",
):
    """
    Register one environment per family and target room count.

    By default this registers:
    - CurriculumMultiRoomEnv-UpRight-R2-v0 ... R6-v0
    - CurriculumMultiRoomEnv-DownLeft-R2-v0 ... R6-v0

    `task_sets` is retained for legacy color-only random-route registrations.
    """
    if layout_num_rooms is None:
        layout_num_rooms = max(max_num_rooms, max_target_rooms)

    if families is not None and task_sets is not None:
        raise ValueError("Use either `families` or legacy `task_sets`, not both")

    if task_sets is not None:
        family_specs = [
            _normalize_legacy_task_set(task_set, task_set_idx)
            for task_set_idx, task_set in enumerate(task_sets)
        ]
    else:
        if families is None:
            families = _default_family_specs()
        family_specs = [_normalize_family_spec(family) for family in families]

    seen_env_ids = set()
    for family in family_specs:
        env_prefix = "CurriculumMultiRoomEnv"
        if family["name"]:
            env_prefix = f"{env_prefix}-{family['name']}"

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
                },
            )


register_curriculum()
