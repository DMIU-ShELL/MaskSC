from gym.envs.registration import register

from gym_minigrid.minigrid import Lava

from .curriculumMultiRoomEnvWallObject import _env_is_registered
from .curriculumMultiRoomEnvWallObjectDecoy import (
    CurriculumMultiRoomWallObjectDecoyEnv,
    _normalize_decoy_task_set,
)


__all__ = [
    "CurriculumMultiRoomWallObjectLavaDecoyEnv",
    "register_curriculum_wall_object_lava_decoy",
]


class CurriculumMultiRoomWallObjectLavaDecoyEnv(
    CurriculumMultiRoomWallObjectDecoyEnv
):
    """Dead-end decoy fork with lava replacing the decoy corridor borders."""

    def _make_decoy_border(self, room_idx):
        return Lava()


def register_curriculum_wall_object_lava_decoy(
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
        "CurriculumMinigrid.curriculumMultiRoomEnvWallObjectLavaDecoy:"
        "CurriculumMultiRoomWallObjectLavaDecoyEnv"
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
        env_prefix = f"CurriculumMultiRoomWallObjectLavaDecoyEnv-{task_set['name']}"

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


register_curriculum_wall_object_lava_decoy()
