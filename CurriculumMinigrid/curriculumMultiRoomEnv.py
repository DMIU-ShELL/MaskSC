from gym_minigrid.minigrid import (
    COLOR_NAMES,
    Door,
    Goal,
    Grid,
    MiniGridEnv,
    MissionSpace,
    Wall,
)
from gym_minigrid.wrappers import ReseedWrapper, ImgObsWrapper


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


def _default_task_set_name(door_colors):
    return "".join(color.capitalize() for color in door_colors)


def _normalize_task_set(task_set, task_set_idx):
    if isinstance(task_set, dict):
        raw_colors = task_set.get("door_colors", task_set.get("door_color"))
        if raw_colors is None:
            raise ValueError(
                "task set dictionaries must include `door_colors` or `door_color`"
            )
        door_colors = _normalize_door_colors(raw_colors)
        name = task_set.get("name")
        legacy_ids = task_set.get("legacy_ids", task_set_idx == 0 and name is None)
    else:
        door_colors = _normalize_door_colors(task_set)
        name = None
        legacy_ids = task_set_idx == 0

    if legacy_ids and name:
        raise ValueError("task sets cannot specify both `name` and `legacy_ids=True`")

    if legacy_ids:
        name = ""
    elif name is None:
        name = _default_task_set_name(door_colors)

    return {
        "name": name,
        "door_colors": door_colors,
    }


class CurriculumMultiRoomEnv(MiniGridEnv):

    """
    ### Description

    This environment has a series of connected rooms with doors that must be
    opened in order to get to the next room. The final room has the green goal
    square the agent must get to. This environment is extremely difficult to
    solve using RL alone. However, by gradually increasing the number of rooms
    and building a curriculum, the environment can be solved.

    ### Mission Space

    "traverse the rooms to get to the goal"

    ### Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [gym_minigrid/minigrid.py](gym_minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    S: size of map SxS.
    N: number of rooms.

    - `MiniGrid-MultiRoom-N2-S4-v0` (two small rooms)
    - `MiniGrid-MultiRoom-N4-S5-v0` (four rooms)
    - `MiniGrid-MultiRoom-N6-v0` (six rooms)

    """

    def __init__(
        self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        targetNumRooms=2,
        layoutNumRooms=None,
        door_colors=None,
        **kwargs
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize
        self.targetNumRooms = targetNumRooms
        if layoutNumRooms is None:
            layoutNumRooms = max(maxNumRooms, targetNumRooms)
        assert layoutNumRooms >= targetNumRooms
        self.layoutNumRooms = layoutNumRooms
        self.door_colors = _normalize_door_colors(door_colors)

        self.rooms = []

        mission_space = MissionSpace(
            mission_func=lambda: "traverse the rooms to get to the goal"
        )

        self.size = 25

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=self.targetNumRooms * 20,
            **kwargs
        )

    def _gen_grid(self, width, height):
        roomList = self._generate_room_chain(width, height, self.layoutNumRooms)
        roomList = roomList[: self.targetNumRooms]

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Use the configured family color sequence for curriculum variants.
                doorColor = self.door_colors[(idx - 1) % len(self.door_colors)]
                entryDoor = Door(doorColor)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], entryDoor)

                prevRoom = roomList[idx - 1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "traverse the rooms to get to the goal"

    def _generate_room_chain(self, width, height, numRooms):
        roomList = []

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (self._rand_int(0, width - 2), self._rand_int(0, height - 2))

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        return roomList

    def _placeRoom(self, numLeft, roomList, minSz, maxSz, entryDoorWall, entryDoorPos):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz + 1)
        sizeY = self._rand_int(minSz, maxSz + 1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = (
                topX + sizeX < room.top[0]
                or room.top[0] + room.size[0] <= topX
                or topY + sizeY < room.top[1]
                or room.top[1] + room.size[1] <= topY
            )

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(MultiRoom((topX, topY), (sizeX, sizeY), entryDoorPos, None))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        success = False
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = {0, 1, 2, 3}
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (topX + sizeX - 1, topY + self._rand_int(1, sizeY - 1))
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY + sizeY - 1)
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (topX, topY + self._rand_int(1, sizeY - 1))
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY)
            else:
                assert False

            # Recursively create the other rooms
            prevLen = len(roomList)
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos,
            )

            if success:
                break
            del roomList[prevLen:]

        return success


from gym.envs.registration import register

def register_curriculum(
    min_target_rooms=2,
    max_target_rooms=11,
    min_num_rooms=2,
    max_num_rooms=10,
    max_room_size=5,
    layout_num_rooms=None,
    task_sets=None,
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnv:CurriculumMultiRoomEnv",
):
    """
    Register one curriculum family per task-set entry.

    Examples for `task_sets`:
    - ["yellow", "blue"]
    - [{"name": "Green", "door_colors": ["green"]}]                 produces CurriculumMultiRoomEnv-Green-R2-v0 ... R11-v0 with all doors being green
    - [{"name": "Alt", "door_colors": ["yellow", "blue"]}]          produces CurriculumMultiRoomEnv-Alt-R2-v0 ... R11-v0 and doors cycle by position: yellow, red, yellow, red, ...

    The first unnamed task set keeps the legacy env ids:
    `CurriculumMultiRoomEnv-R{rooms}-v0`.
    """
    if layout_num_rooms is None:
        layout_num_rooms = max(max_num_rooms, max_target_rooms)

    if task_sets is None:
        task_sets = ["yellow"]
    if not task_sets:
        raise ValueError("task_sets must contain at least one task set")

    normalized_task_sets = [
        _normalize_task_set(task_set, task_set_idx)
        for task_set_idx, task_set in enumerate(task_sets)
    ]

    seen_env_ids = set()
    for task_set in normalized_task_sets:
        env_prefix = "CurriculumMultiRoomEnv"
        if task_set["name"]:
            env_prefix = f"{env_prefix}-{task_set['name']}"

        for target_num_rooms in range(min_target_rooms, max_target_rooms + 1):
            env_id = f"{env_prefix}-R{target_num_rooms}-v0"
            if env_id in seen_env_ids:
                raise ValueError(f"duplicate env id generated for curriculum: {env_id}")
            seen_env_ids.add(env_id)

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
                },
            )


register_curriculum(
    task_sets = [
        {"name": "Yellow", "door_colors": ["yellow"]},
        {"name": "Blue", "door_colors": ["blue"]},
    ]
)
