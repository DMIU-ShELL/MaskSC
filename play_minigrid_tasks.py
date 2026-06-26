import argparse
import json
import os
import sys

import gym
from gym_minigrid.wrappers import ReseedWrapper


# Make the local CurriculumMinigrid package importable when this script is run
# from either the repo root or mask-lrl-cluster-optimization/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CurriculumMinigrid import curriculum_multiroom_key_env  # noqa: F401


RAW_ACTION_MEANINGS = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

ACTION_CHARS = {
    "a": 0,
    "d": 1,
    "w": 2,
    "p": 3,
    "x": 4,
    "o": 5,
    "e": 6,
}

DIR_TO_TOKEN = {
    0: "A>",
    1: "Av",
    2: "A<",
    3: "A^",
}

COLOR_TO_CHAR = {
    "red": "R",
    "green": "G",
    "blue": "B",
    "purple": "P",
    "yellow": "Y",
    "grey": "X",
}

ANSI_RESET = "\033[0m"
ANSI_BY_COLOR = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "purple": "\033[35m",
    "grey": "\033[37m",
}


def unpack_reset(ret):
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0]
    return ret


def unpack_step(ret):
    if isinstance(ret, tuple) and len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, reward, terminated or truncated, info
    return ret


def load_curriculum(env_config_path, fallback_seed):
    with open(env_config_path, "r") as f:
        env_config = json.load(f)

    env_names = env_config["tasks"]
    seeds = env_config.get("seeds", fallback_seed)

    if isinstance(seeds, int):
        seeds = [seeds] * len(env_names)
    elif isinstance(seeds, list):
        if len(seeds) != len(env_names):
            raise ValueError("number of seeds in config should match number of tasks")
    else:
        raise ValueError("invalid seed specification in config file")

    return list(zip(env_names, seeds))


def maybe_colorize(token, color, use_color):
    if not use_color:
        return token
    prefix = ANSI_BY_COLOR.get(color)
    if prefix is None:
        return token
    return f"{prefix}{token}{ANSI_RESET}"


def object_token(obj, use_color):
    if obj is None:
        return ".."

    obj_type = getattr(obj, "type", type(obj).__name__.lower())
    color = getattr(obj, "color", None)
    color_char = COLOR_TO_CHAR.get(color, "?")

    if obj_type == "wall":
        return maybe_colorize("##", color, use_color)
    if obj_type == "goal":
        return maybe_colorize("GG", color, use_color)
    if obj_type == "key":
        return maybe_colorize(f"K{color_char}", color, use_color)
    if obj_type == "door":
        if getattr(obj, "is_open", False):
            prefix = "O"
        elif getattr(obj, "is_locked", False):
            prefix = "L"
        else:
            prefix = "D"
        return maybe_colorize(f"{prefix}{color_char}", color, use_color)
    if obj_type == "ball":
        return maybe_colorize(f"B{color_char}", color, use_color)
    if obj_type == "box":
        return maybe_colorize(f"X{color_char}", color, use_color)
    if obj_type == "lava":
        return "LV"

    return maybe_colorize(f"{obj_type[:1].upper()}{color_char}", color, use_color)


class MiniGridTaskSession:
    def __init__(self, task_specs, task_idx=0):
        self.task_specs = task_specs
        self.task_idx = task_idx
        self.env = None
        self.obs = None
        self.episode_return = 0.0
        self.episode_steps = 0
        self.last_reward = 0.0
        self.last_done = False
        self.use_color = sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")
        self.load_task(task_idx)

    def load_task(self, task_idx):
        if task_idx < 0 or task_idx >= len(self.task_specs):
            print(f"task index {task_idx} is out of range")
            return False

        if self.env is not None:
            self.env.close()

        self.task_idx = task_idx
        env_name, seed = self.task_specs[task_idx]
        env = gym.make(env_name)
        self.env = ReseedWrapper(env, seeds=[seed])
        self.reset()
        return True

    def reset(self):
        self.obs = unpack_reset(self.env.reset())
        self.episode_return = 0.0
        self.episode_steps = 0
        self.last_reward = 0.0
        self.last_done = False

    def step(self, action):
        if self.last_done:
            print("episode already ended; use `reset`, `next`, or `prev`")
            return

        self.obs, reward, done, info = unpack_step(self.env.step(action))
        self.episode_steps += 1
        self.episode_return += reward
        self.last_reward = reward
        self.last_done = done

    def current_env_name(self):
        return self.task_specs[self.task_idx][0]

    def current_seed(self):
        return self.task_specs[self.task_idx][1]

    def carrying_str(self):
        carrying = self.env.unwrapped.carrying
        if carrying is None:
            return "none"
        color = getattr(carrying, "color", "?")
        item_type = getattr(carrying, "type", type(carrying).__name__.lower())
        return f"{item_type}:{color}"

    def status_lines(self):
        mission = getattr(self.env.unwrapped, "mission", "")
        lines = [
            f"task {self.task_idx}: {self.current_env_name()} | seed={self.current_seed()}",
            f"steps={self.episode_steps}/{self.env.unwrapped.max_steps} | "
            f"return={self.episode_return:.3f} | "
            f"last_reward={self.last_reward:.3f} | "
            f"done={self.last_done} | "
            f"carrying={self.carrying_str()}",
        ]
        if mission:
            lines.append(f"mission: {mission}")
        return lines

    def render_ascii(self):
        env = self.env.unwrapped
        width = env.width
        height = env.height

        header_tens = "    " + "".join(
            f"{(x // 10) if x >= 10 else ' ':>3}" for x in range(width)
        )
        header_ones = "    " + "".join(f"{x % 10:>3}" for x in range(width))

        lines = [header_tens, header_ones]

        agent_x, agent_y = env.agent_pos
        for y in range(height):
            row_tokens = []
            for x in range(width):
                if x == agent_x and y == agent_y:
                    token = DIR_TO_TOKEN.get(env.agent_dir, "A?")
                else:
                    token = object_token(env.grid.get(x, y), self.use_color)
                row_tokens.append(f"{token:>3}")
            lines.append(f"{y:>3} " + "".join(row_tokens))

        return "\n".join(lines)

    def print_state(self):
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(self.render_ascii())
        print("")
        for line in self.status_lines():
            print(line)
        print("")
        print("commands:")
        print("  actions: a=left d=right w=forward p=pickup o=toggle x=drop e=done")
        print("  you can chain actions, e.g. `wwaow`")
        print("  reset | next | prev | goto <idx> | info | help | quit")


class TerminalMiniGridPlayer:
    def __init__(self, task_specs, task_idx=0):
        self.session = MiniGridTaskSession(task_specs, task_idx)

    def _apply_action_sequence(self, action_sequence):
        for char in action_sequence:
            action = ACTION_CHARS[char]
            self.session.step(action)
            if self.session.last_done:
                break

    def _handle_command(self, command):
        if not command:
            return True

        if command == "quit" or command == "q":
            return False

        if command == "help":
            return True

        if command == "info":
            return True

        if command == "reset":
            self.session.reset()
            return True

        if command == "next":
            self.session.load_task(self.session.task_idx + 1)
            return True

        if command == "prev":
            self.session.load_task(self.session.task_idx - 1)
            return True

        if command.startswith("goto "):
            try:
                task_idx = int(command.split(maxsplit=1)[1])
            except ValueError:
                print("invalid task index")
                return True
            self.session.load_task(task_idx)
            return True

        if all(ch in ACTION_CHARS for ch in command):
            self._apply_action_sequence(command)
            return True

        print(f"unknown command: {command}")
        return True

    def run(self):
        while True:
            self.session.print_state()
            try:
                command = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("")
                break

            should_continue = self._handle_command(command)
            if not should_continue:
                break

        if self.session.env is not None:
            self.session.env.close()


class WindowMiniGridPlayer:
    def __init__(self, task_specs, task_idx=0, tile_size=32, agent_pov=False):
        from gym_minigrid.window import Window

        self.task_specs = task_specs
        self.task_idx = task_idx
        self.tile_size = tile_size
        self.agent_pov = agent_pov
        self.window = Window("MiniGrid Task Player")
        self.window.reg_key_handler(self.key_handler)
        self.session = MiniGridTaskSession(task_specs, task_idx)
        self.redraw()

    def redraw(self):
        img = self.session.env.unwrapped.get_frame(
            highlight=True,
            tile_size=self.tile_size,
            agent_pov=self.agent_pov,
        )
        caption = " | ".join(self.session.status_lines())
        self.window.set_caption(caption)
        self.window.show_img(img)

    def key_handler(self, event):
        key = event.key
        if key == "escape":
            self.close()
            return
        if key in ("backspace", "r"):
            self.session.reset()
        elif key == "n":
            self.session.load_task(self.session.task_idx + 1)
        elif key == "b":
            self.session.load_task(self.session.task_idx - 1)
        elif key in ("left", "a"):
            self.session.step(0)
        elif key in ("right", "d"):
            self.session.step(1)
        elif key in ("up", "w"):
            self.session.step(2)
        elif key == "p":
            self.session.step(3)
        elif key == "x":
            self.session.step(4)
        elif key in (" ", "space", "o"):
            self.session.step(5)
        elif key == "enter":
            self.session.step(6)
        self.redraw()

    def run(self):
        self.window.show(block=True)

    def close(self):
        if self.session.env is not None:
            self.session.env.close()
        self.window.close()


def main():
    parser = argparse.ArgumentParser(
        description="Play MiniGrid tasks from a curriculum config."
    )
    parser.add_argument(
        "--env_config_path",
        default="./env_configs/minigrid_7.json",
        help="path to a MiniGrid curriculum JSON",
    )
    parser.add_argument(
        "--task_idx",
        type=int,
        default=0,
        help="task index to start from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="fallback seed if the config does not provide per-task seeds",
    )
    parser.add_argument(
        "--mode",
        choices=["terminal", "window"],
        default="terminal",
        help="play in the terminal or using the MiniGrid matplotlib window",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=32,
        help="render tile size for window mode",
    )
    parser.add_argument(
        "--agent_pov",
        action="store_true",
        help="render the agent-centric view in window mode",
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="print tasks from the config and exit",
    )
    args = parser.parse_args()

    task_specs = load_curriculum(args.env_config_path, args.seed)

    if args.list_tasks:
        for idx, (env_name, seed) in enumerate(task_specs):
            print(f"{idx:02d}: {env_name} | seed={seed}")
        return

    if args.task_idx < 0 or args.task_idx >= len(task_specs):
        raise ValueError(
            f"--task_idx should be in [0, {len(task_specs) - 1}], got {args.task_idx}"
        )

    if args.mode == "terminal":
        player = TerminalMiniGridPlayer(task_specs, args.task_idx)
        player.run()
    else:
        player = WindowMiniGridPlayer(
            task_specs,
            args.task_idx,
            tile_size=args.tile_size,
            agent_pov=args.agent_pov,
        )
        try:
            player.run()
        finally:
            player.close()


if __name__ == "__main__":
    main()
