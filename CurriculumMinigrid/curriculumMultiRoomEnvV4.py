from .curriculumMultiRoomEnvV3 import (
    _VisualCurriculumMultiRoomEnv,
    _register_curriculum_variant,
)


__all__ = ["CurriculumMultiRoomEnvV4", "register_curriculum_v4"]


class CurriculumMultiRoomEnvV4(_VisualCurriculumMultiRoomEnv):
    """V4: family-specific non-blocking floor texture, default grey walls."""

    DEFAULT_TEXTURE_COLORS = ("green",)
    DEFAULT_TEXTURE_PATTERN = "solid"


def _default_texture_families():
    return (
        {
            "name": "UpRight",
            "door_colors": ["yellow"],
            "route_signature": "up_right",
            "wall_colors": ["grey"],
            "texture_colors": ["green"],
            "texture_pattern": "solid",
        },
        {
            "name": "DownLeft",
            "door_colors": ["blue"],
            "route_signature": "down_left",
            "wall_colors": ["grey"],
            "texture_colors": ["purple"],
            "texture_pattern": "solid",
        },
    )


def register_curriculum_v4(
    min_target_rooms=2,
    max_target_rooms=6,
    min_num_rooms=2,
    max_num_rooms=6,
    max_room_size=5,
    layout_num_rooms=6,
    grid_size=30,
    families=None,
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvV4:CurriculumMultiRoomEnvV4",
):
    if families is None:
        families = _default_texture_families()

    _register_curriculum_variant(
        "V4",
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


register_curriculum_v4()
