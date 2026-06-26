# curriculum_multiroom_key_env.py
#
# MiniGrid benchmark for interleaved hierarchical DoorKey-style curricula.
#
# Design:
# - Each task belongs to a family (A/B/C/D), where each family defines an
#   ordered color sequence, e.g. ["yellow", "red", "blue", "green"].
# - A task of depth d uses the first d colors in that sequence.
# - The environment contains depth+1 rooms:
#     room 0 (start), room 1, ..., room d (goal room)
# - To enter room i (i >= 1), the agent must unlock a locked door whose color
#   is sequence[i-1]. The matching key is placed in room i-1.
#
# This yields a prefix hierarchy:
#   A1 = [yellow]
#   A2 = [yellow, red]
#   A3 = [yellow, red, blue]
# etc.
#
# Notes:
# - This file targets gym-minigrid style APIs like your existing environment.
# - It assumes Door(color, is_locked=True) and Key(color) behave as in MiniGrid:
#   pickup grabs a key, toggle unlocks/opens the adjacent matching locked door.
# - Start with no distractor keys/doors for stability and easier debugging.
#
# Example task kwargs:
#   {"family_id": "A", "depth": 3}
#
# Example curriculum:
#   A1, B1, C1, D1, A2, B2, C2, D2, ...

from typing import Dict, List, Optional, Tuple

from gym.envs.registration import register

from gym_minigrid.minigrid import (
    COLOR_NAMES,
    Door,
    Goal,
    Grid,
    Key,
    MiniGridEnv,
    MissionSpace,
    Wall,
)
from gym_minigrid.wrappers import ReseedWrapper, ImgObsWrapper  # noqa: F401


class MultiRoom:
    def __init__(
        self,
        top: Tuple[int, int],
        size: Tuple[int, int],
        entryDoorPos: Optional[Tuple[int, int]],
        exitDoorPos: Optional[Tuple[int, int]],
        doorColor: Optional[str] = None,
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos
        self.doorColor = doorColor


class CurriculumMultiRoomKeyEnv(MiniGridEnv):
    """
    Multi-room locked-door environment for hierarchical lifelong RL curricula.

    Each task is parameterized by:
      - family_id: which ordered color sequence to use
      - depth: prefix length of that sequence

    A task with depth=d creates d+1 rooms:
      room 0 -> room 1 uses color_prefix[0]
      room 1 -> room 2 uses color_prefix[1]
      ...
      room d-1 -> room d uses color_prefix[d-1]

    The matching key for each locked door is placed in the immediately
    preceding room.

    Mission:
      "collect keys, unlock doors, and reach the goal"

    Rewards:
      Standard MiniGrid sparse success reward.

    Termination:
      - Goal reached
      - Timeout
    """

    DEFAULT_FAMILY_SEQUENCES: Dict[str, List[str]] = {
        "A": ["yellow", "red", "blue", "green", "purple", "grey"],
        "B": ["blue", "yellow", "green", "purple", "red", "grey"],
        "C": ["green", "purple", "yellow", "red", "blue", "grey"],
        "D": ["red", "blue", "purple", "yellow", "green", "grey"],
    }

    def __init__(
        self,
        family_id: str = "A",
        depth: int = 2,
        family_sequences: Optional[Dict[str, List[str]]] = None,
        minNumRooms: int = 2,
        maxNumRooms: int = 12,
        maxRoomSize: int = 5,
        width: int = 25,
        height: int = 25,
        max_steps_per_room: int = 30,
        layout_seed_offset: int = 0,
        **kwargs,
    ):
        assert depth >= 1, "depth must be >= 1"
        assert maxRoomSize >= 4, "maxRoomSize must be >= 4"

        if family_sequences is None:
            family_sequences = self.DEFAULT_FAMILY_SEQUENCES

        assert family_id in family_sequences, f"Unknown family_id={family_id}"
        assert depth <= len(
            family_sequences[family_id]
        ), f"depth={depth} exceeds available family sequence length"

        self.family_id = family_id
        self.depth = depth
        self.family_sequences = family_sequences
        self.color_prefix = family_sequences[family_id][:depth]

        # Number of rooms is depth + 1:
        # room 0 is the start room, each additional room requires one door/key stage.
        self.targetNumRooms = depth + 1

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize
        self.rooms: List[MultiRoom] = []

        self.layout_seed_offset = layout_seed_offset

        mission_space = MissionSpace(
            mission_func=lambda: "collect keys, unlock doors, and reach the goal"
        )

        self.size = width
        self.goal_pos = None

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=self.targetNumRooms * max_steps_per_room,
            **kwargs,
        )

    def _gen_grid(self, width: int, height: int):
        roomList: List[MultiRoom] = []

        # Try repeatedly until we obtain enough rooms.
        while len(roomList) < self.targetNumRooms:
            curRoomList: List[MultiRoom] = []

            # Initial anchor position for recursive room placement.
            # Keep some border margin for robustness.
            entryDoorPos = (
                self._rand_int(1, max(2, width - 2)),
                self._rand_int(1, max(2, height - 2)),
            )

            self._placeRoom(
                numLeft=self.targetNumRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        assert len(roomList) >= self.targetNumRooms
        roomList = roomList[: self.targetNumRooms]
        self.rooms = roomList

        self.grid = Grid(width, height)
        wall = Wall()

        # Draw rooms and place locked entry doors for rooms 1..depth.
        for idx, room in enumerate(roomList):
            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw horizontal walls
            for i in range(sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw vertical walls
            for j in range(sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # Place entry door for every room except the first.
            if idx > 0:
                door_color = self.color_prefix[idx - 1]
                entryDoor = Door(door_color, is_locked=True)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], entryDoor)
                room.doorColor = door_color

                prevRoom = roomList[idx - 1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Place the matching key for each door in the preceding room.
        for idx in range(1, len(roomList)):
            prev_room = roomList[idx - 1]
            key_color = self.color_prefix[idx - 1]
            self._place_key_in_room(prev_room, key_color)

        # Place the agent in the first room.
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the goal in the last room.
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "collect keys, unlock doors, and reach the goal"

    def _place_key_in_room(self, room: MultiRoom, color: str):
        """
        Place a single key of the given color in an empty interior tile of the room.
        """
        key = Key(color)

        topX, topY = room.top
        sizeX, sizeY = room.size

        # Try many random placements before giving up.
        for _ in range(200):
            x = self._rand_int(topX + 1, topX + sizeX - 1)
            y = self._rand_int(topY + 1, topY + sizeY - 1)

            if self.grid.get(x, y) is not None:
                continue
            if room.entryDoorPos is not None and (x, y) == room.entryDoorPos:
                continue
            if room.exitDoorPos is not None and (x, y) == room.exitDoorPos:
                continue

            self.grid.set(x, y, key)
            return

        raise RuntimeError(
            f"Failed to place key color={color} in room top={room.top}, size={room.size}"
        )

    def _placeRoom(
        self,
        numLeft: int,
        roomList: List[MultiRoom],
        minSz: int,
        maxSz: int,
        entryDoorWall: int,
        entryDoorPos: Tuple[int, int],
    ) -> bool:
        """
        Recursive room placement adapted from your original environment.
        """
        sizeX = self._rand_int(minSz, maxSz + 1)
        sizeY = self._rand_int(minSz, maxSz + 1)

        # First room is anchored at entryDoorPos.
        if len(roomList) == 0:
            topX, topY = entryDoorPos

        # Entry on right wall
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)

        # Entry on bottom wall
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1

        # Entry on left wall
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)

        # Entry on top wall
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]

        else:
            raise ValueError(f"Invalid entryDoorWall={entryDoorWall}")

        # Out of bounds
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # Check overlap with existing rooms
        for room in roomList[:-1]:
            nonOverlap = (
                topX + sizeX < room.top[0]
                or room.top[0] + room.size[0] <= topX
                or topY + sizeY < room.top[1]
                or room.top[1] + room.size[1] <= topY
            )
            if not nonOverlap:
                return False

        roomList.append(
            MultiRoom(
                top=(topX, topY),
                size=(sizeX, sizeY),
                entryDoorPos=entryDoorPos if len(roomList) > 0 else None,
                exitDoorPos=None,
                doorColor=None,
            )
        )

        if numLeft == 1:
            return True

        success = False

        # Try several possible outgoing walls/positions for the next room.
        for _ in range(8):
            wallSet = {0, 1, 2, 3}
            if len(roomList) > 1:
                wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Sample exit door position on the selected wall.
            if exitDoorWall == 0:  # right
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1),
                )
            elif exitDoorWall == 1:  # bottom
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1,
                )
            elif exitDoorWall == 2:  # left
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1),
                )
            elif exitDoorWall == 3:  # top
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY,
                )
            else:
                raise ValueError(f"Invalid exitDoorWall={exitDoorWall}")

            success = self._placeRoom(
                numLeft=numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos,
            )

            if success:
                break

        return success

    @property
    def task_name(self) -> str:
        return f"{self.family_id}{self.depth}"

    @property
    def task_metadata(self) -> dict:
        return {
            "family_id": self.family_id,
            "depth": self.depth,
            "color_prefix": list(self.color_prefix),
            "num_rooms": self.targetNumRooms,
        }


# -----------------------------------------------------------------------------
# Optional registration helpers
# -----------------------------------------------------------------------------

def make_default_family_sequences():
    return {
        "A": ["yellow", "red", "blue", "green", "purple", "grey"],
        "B": ["blue", "yellow", "green", "purple", "red", "grey"],
        "C": ["green", "purple", "yellow", "red", "blue", "grey"],
        "D": ["red", "blue", "purple", "yellow", "green", "grey"],
    }


def register_curriculum_multiroom_key_envs(
    max_depth: int = 6,
    families: Tuple[str, ...] = ("A", "B", "C", "D"),
    entry_point: str = "CurriculumMinigrid.curriculum_multiroom_key_env:CurriculumMultiRoomKeyEnv",
):
    """
    Register a small family/depth grid of tasks for debugging.
    For lifelong experiments, you may prefer to instantiate via kwargs directly.
    """
    family_sequences = make_default_family_sequences()

    for fam in families:
        for depth in range(1, max_depth + 1):
            register(
                id=f"CurriculumMultiRoomKey-{fam}{depth}-v0",
                entry_point=entry_point,
                kwargs={
                    "family_id": fam,
                    "depth": depth,
                    "family_sequences": family_sequences,
                    "maxRoomSize": 5,
                    "width": 25,
                    "height": 25,
                    "max_steps_per_room": 30,
                },
            )


# Uncomment if you want automatic registration when importing this file.
register_curriculum_multiroom_key_envs(max_depth=4)