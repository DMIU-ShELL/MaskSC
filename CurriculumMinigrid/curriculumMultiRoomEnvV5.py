from .curriculumMultiRoomEnvV3 import (
    _VisualCurriculumMultiRoomEnv,
    _register_curriculum_variant,
)


__all__ = ["CurriculumMultiRoomEnvV5", "register_curriculum_v5"]


class CurriculumMultiRoomEnvV5(_VisualCurriculumMultiRoomEnv):
    """V5: combined family-specific wall colors and non-blocking floor texture."""

    DEFAULT_WALL_COLORS = ("green",)
    DEFAULT_TEXTURE_COLORS = ("yellow",)
    DEFAULT_TEXTURE_PATTERN = "solid"


def _default_combined_families():
    return (
        {
            "name": "UpRight",
            "door_colors": ["yellow"],
            "route_signature": "up_right",
            "wall_colors": ["yellow"],
            "texture_colors": ["yellow"],
            "texture_pattern": "solid",
        },
        {
            "name": "DownLeft",
            "door_colors": ["blue"],
            "route_signature": "down_left",
            "wall_colors": ["blue"],
            "texture_colors": ["blue"],
            "texture_pattern": "solid",
        },
    )


def register_curriculum_v5(
    min_target_rooms=2,
    max_target_rooms=6,
    min_num_rooms=2,
    max_num_rooms=6,
    max_room_size=5,
    layout_num_rooms=6,
    grid_size=30,
    families=None,
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvV5:CurriculumMultiRoomEnvV5",
):
    if families is None:
        families = _default_combined_families()

    _register_curriculum_variant(
        "V5",
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


register_curriculum_v5()
