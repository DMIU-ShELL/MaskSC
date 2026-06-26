from .curriculumMultiRoomEnvV6 import (
    CurriculumMultiRoomEnvV6,
    _register_decor_curriculum_variant,
)


__all__ = ["CurriculumMultiRoomEnvV7", "register_curriculum_v7"]


class CurriculumMultiRoomEnvV7(CurriculumMultiRoomEnvV6):
    """V7: combined V3 wall colors and V6 decorative room objects."""


def _default_wall_decor_families():
    return (
        {
            "name": "UpRight",
            "door_colors": ["yellow"],
            "route_signature": "up_right",
            "wall_colors": ["yellow"],
            "decor_types": ["ball"],
            "decor_colors": ["yellow"],
            "decor_positions": ["nw"],
        },
        {
            "name": "DownLeft",
            "door_colors": ["blue"],
            "route_signature": "down_left",
            "wall_colors": ["blue"],
            "decor_types": ["box"],
            "decor_colors": ["blue"],
            "decor_positions": ["se"],
        },
    )


def register_curriculum_v7(
    min_target_rooms=2,
    max_target_rooms=6,
    min_num_rooms=2,
    max_num_rooms=6,
    max_room_size=5,
    layout_num_rooms=6,
    grid_size=30,
    families=None,
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvV7:CurriculumMultiRoomEnvV7",
):
    if families is None:
        families = _default_wall_decor_families()

    _register_decor_curriculum_variant(
        "V7",
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


register_curriculum_v7()
