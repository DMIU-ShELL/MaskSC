import argparse
import json
import os
from pathlib import Path
import re
import sys

import gym
from PIL import Image, ImageDraw
from gym_minigrid.minigrid import IDX_TO_COLOR, IDX_TO_OBJECT
from gym_minigrid.wrappers import ReseedWrapper, RGBImgObsWrapper

# Make the local CurriculumMinigrid package importable when this script is run
# from either the repo root or mask-lrl-cluster-optimization/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#from CurriculumMinigrid import curriculumMultiRoomEnvV2, curriculumMultiRoomEnvV3, curriculumMultiRoomEnvV4, curriculumMultiRoomEnvV5, curriculumMultiRoomEnvV6, curriculumMultiRoomEnvV7
from CurriculumMinigrid import curriculumMultiRoomEnvObjectRemap 


COLOR_RGB = {
    "red": (215, 48, 39),
    "green": (50, 160, 70),
    "blue": (50, 100, 220),
    "purple": (145, 75, 190),
    "yellow": (235, 200, 60),
    "grey": (145, 145, 145),
}

OBJECT_LABELS = {
    "wall": "W",
    "door": "D",
    "key": "K",
    "ball": "O",
    "box": "X",
    "goal": "G",
    "lava": "L",
    "floor": "F",
    "unseen": "?",
}


# Function to convert MiniGrid environment observation to RGB image
def minigrid_to_image(observation):
    # Check if the observation is a tuple
    if isinstance(observation, tuple):
        # If observation is a tuple, extract the first element
        img_data = observation[0]['image']
    else:
        # If observation is a dictionary, extract the 'image' key
        img_data = observation['image']
    
    # Convert image data to a PIL Image
    img = Image.fromarray(img_data)
    
    return img


def lookup_idx(mapping, idx):
    idx = int(idx)
    return mapping[idx]


def color_to_rgb(color_name):
    return COLOR_RGB.get(color_name, (180, 180, 180))


def text_bbox(draw, text):
    if hasattr(draw, "textbbox"):
        return draw.textbbox((0, 0), text)
    width, height = draw.textsize(text)
    return (0, 0, width, height)


def draw_centered_text(draw, box, text, fill):
    if not text:
        return

    left, top, right, bottom = box
    text_left, text_top, text_right, text_bottom = text_bbox(draw, text)
    text_width = text_right - text_left
    text_height = text_bottom - text_top
    x = left + ((right - left) - text_width) / 2
    y = top + ((bottom - top) - text_height) / 2
    draw.text((x, y), text, fill=fill)


def draw_symbolic_cell(draw, x, y, tile_size, obj_name, color_name, state, draw_labels):
    left = x * tile_size
    top = y * tile_size
    right = left + tile_size
    bottom = top + tile_size
    pad = max(2, tile_size // 8)
    color = color_to_rgb(color_name)

    draw.rectangle((left, top, right, bottom), fill=(250, 250, 250), outline=(210, 210, 210))

    if obj_name in ("empty", None):
        return

    if obj_name == "unseen":
        draw.rectangle((left, top, right, bottom), fill=(35, 35, 35), outline=(80, 80, 80))
    elif obj_name == "wall":
        draw.rectangle((left, top, right, bottom), fill=color, outline=(70, 70, 70))
    elif obj_name == "door":
        draw.rectangle(
            (left + pad, top + pad, right - pad, bottom - pad),
            fill=color,
            outline=(40, 40, 40),
        )
        if state == 0:
            draw.rectangle(
                (left + pad, top + pad, right - pad, bottom - pad),
                fill=(245, 245, 245),
                outline=color,
            )
            draw.line((left + pad, top + pad, left + pad, bottom - pad), fill=color, width=2)
        elif state == 2:
            draw.line((left + pad, top + pad, right - pad, bottom - pad), fill=(0, 0, 0), width=2)
            draw.line((left + pad, bottom - pad, right - pad, top + pad), fill=(0, 0, 0), width=2)
    elif obj_name == "key":
        cy = top + tile_size // 2
        draw.ellipse(
            (left + pad, cy - pad, left + 3 * pad, cy + pad),
            outline=color,
        )
        draw.line((left + 3 * pad, cy, right - pad, cy), fill=color, width=2)
        draw.line((right - 2 * pad, cy, right - 2 * pad, cy + pad), fill=color, width=2)
    elif obj_name == "ball":
        draw.ellipse(
            (left + pad, top + pad, right - pad, bottom - pad),
            fill=color,
            outline=(40, 40, 40),
        )
    elif obj_name == "box":
        draw.rectangle(
            (left + pad, top + pad, right - pad, bottom - pad),
            fill=(250, 250, 250),
            outline=color,
        )
        draw.line((left + pad, top + pad, right - pad, bottom - pad), fill=color, width=2)
    elif obj_name == "goal":
        draw.rectangle((left + pad, top + pad, right - pad, bottom - pad), fill=color, outline=color)
    elif obj_name == "lava":
        draw.rectangle((left, top, right, bottom), fill=(230, 85, 30), outline=(140, 30, 10))
        mid_y = top + tile_size // 2
        draw.line(
            (left + pad, mid_y, left + 2 * pad, top + pad, left + 3 * pad, mid_y, right - pad, top + pad),
            fill=(255, 210, 50),
            width=2,
        )
    else:
        draw.rectangle((left + pad, top + pad, right - pad, bottom - pad), fill=color, outline=(40, 40, 40))

    if draw_labels:
        label = OBJECT_LABELS.get(obj_name, obj_name[:1].upper())
        if obj_name == "door":
            label = f"{label}{state}"
        draw_centered_text(
            draw,
            (left, top, right, bottom),
            label,
            fill=(0, 0, 0) if obj_name != "unseen" else (255, 255, 255),
        )


def symbolic_grid_to_image(encoded_grid, agent_pos=None, tile_size=24, draw_labels=True):
    width, height = encoded_grid.shape[:2]
    img = Image.new("RGB", (width * tile_size, height * tile_size), "white")
    draw = ImageDraw.Draw(img)

    for x in range(width):
        for y in range(height):
            obj_idx, color_idx, state = encoded_grid[x, y]
            obj_name = lookup_idx(IDX_TO_OBJECT, obj_idx)
            color_name = lookup_idx(IDX_TO_COLOR, color_idx)
            draw_symbolic_cell(
                draw,
                x,
                y,
                tile_size,
                obj_name,
                color_name,
                int(state),
                draw_labels,
            )

    if agent_pos is not None:
        agent_x, agent_y = agent_pos
        left = agent_x * tile_size
        top = agent_y * tile_size
        right = left + tile_size
        bottom = top + tile_size
        pad = max(3, tile_size // 6)
        draw.polygon(
            [
                (left + tile_size / 2, top + pad),
                (right - pad, bottom - pad),
                (left + pad, bottom - pad),
            ],
            fill=(20, 20, 20),
            outline=(255, 255, 255),
        )

    return img

def resolve_path(path_str):
    path = Path(path_str)
    if path.exists():
        return path

    script_relative_path = Path(__file__).resolve().parent / path_str
    if script_relative_path.exists():
        return script_relative_path

    return path


def sanitize_label(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def looks_like_curriculum_path(value):
    if not value:
        return False
    return value.endswith(".json") and resolve_path(value).is_file()


def load_curriculum(env_config_path, fallback_seed):
    """Return [(env_id, seed), ...] from a MiniGrid curriculum config."""
    env_config_path = resolve_path(env_config_path)
    with open(env_config_path, "r") as f:
        env_config = json.load(f)

    env_names = env_config["tasks"]
    seeds = env_config.get("seeds", fallback_seed)

    if isinstance(seeds, int):
        seeds = [seeds] * len(env_names)
    elif isinstance(seeds, list):
        assert len(seeds) == len(env_names), (
            "number of seeds in config should match number of tasks"
        )
    else:
        raise ValueError("invalid seed specification in config file")

    return list(zip(env_names, seeds))


def reset_env(env):
    ret = env.reset()
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0]
    return ret


def save_minigrid_image(
    env_name,
    filename,
    seed,
    mode="rgb",
    tile_size=24,
    draw_symbolic_labels=True,
):
    if looks_like_curriculum_path(env_name):
        raise ValueError(
            f"`{env_name}` looks like a curriculum config path, not a Gym environment id. "
            "Pass it as `--env_config_path <path>` or as the only positional argument."
        )

    env = gym.make(env_name)

    if mode == "rgb":
        env = RGBImgObsWrapper(env)
        env = ReseedWrapper(env, seeds=[seed])
        obs = reset_env(env)
        img = minigrid_to_image(obs)
    elif mode == "symbolic":
        env = ReseedWrapper(env, seeds=[seed])
        reset_env(env)
        unwrapped = env.unwrapped
        img = symbolic_grid_to_image(
            unwrapped.grid.encode(),
            agent_pos=getattr(unwrapped, "agent_pos", None),
            tile_size=tile_size,
            draw_labels=draw_symbolic_labels,
        )
    else:
        raise ValueError(f"unknown plot mode: {mode}")

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    img.save(filename)
    env.close()

    return img


def make_contact_sheet(images, labels, out_path, columns=4, label_height=24):
    if not images:
        return

    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images) + label_height
    rows = (len(images) + columns - 1) // columns

    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (img, label) in enumerate(zip(images, labels)):
        row, col = divmod(idx, columns)
        x = col * cell_w
        y = row * cell_h
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
        sheet.paste(img, (x, y + label_height))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('envs', help='environment ids to render, or a single curriculum JSON path', nargs='*')
    parser.add_argument('--env_config_path', help='render all tasks from this curriculum config')
    parser.add_argument('--seed', help='fallback seed when env_config_path has no seeds', type=int, default=1234)
    parser.add_argument('--output_dir', help='directory for generated images', default='minigrid_images')
    parser.add_argument(
        '--mode',
        help='rgb uses MiniGrid rendering; symbolic plots grid.encode() object/color/state values',
        choices=('rgb', 'symbolic'),
        default='rgb',
    )
    parser.add_argument('--tile_size', help='symbolic-mode tile size in pixels', type=int, default=24)
    parser.add_argument(
        '--no_symbolic_labels',
        help='hide object labels in symbolic mode',
        action='store_true',
    )
    parser.add_argument('--contact_sheet', help='also save a combined contact-sheet PNG', action='store_true')
    parser.add_argument('--columns', help='number of columns in the contact sheet', type=int, default=4)
    args = parser.parse_args()

    if args.env_config_path:
        env_seed_pairs = load_curriculum(args.env_config_path, args.seed)
    else:
        if not args.envs:
            raise ValueError("provide env ids or --env_config_path")
        if len(args.envs) == 1 and looks_like_curriculum_path(args.envs[0]):
            env_seed_pairs = load_curriculum(args.envs[0], args.seed)
        else:
            env_seed_pairs = [(env_name, args.seed) for env_name in args.envs]

    images = []
    labels = []
    for task_idx, (env_name, seed) in enumerate(env_seed_pairs):
        mode_prefix = "" if args.mode == "rgb" else f"{args.mode}_"
        label = f"{task_idx:02d}_{mode_prefix}{sanitize_label(env_name)}_seed{seed}"
        filename = Path(args.output_dir) / f"{label}.pdf"
        img = save_minigrid_image(
            env_name,
            filename,
            seed,
            mode=args.mode,
            tile_size=args.tile_size,
            draw_symbolic_labels=not args.no_symbolic_labels,
        )
        images.append(img)
        labels.append(label)
        print(f"saved {filename}")

    if args.contact_sheet:
        out_path = Path(args.output_dir) / "curriculum_contact_sheet.pdf"
        make_contact_sheet(images, labels, out_path, columns=args.columns)
        print(f"saved {out_path}")
