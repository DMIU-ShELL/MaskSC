import argparse
import json
import random


DEFAULT_PROFILE_POOL = [
    "Identity",
    "IdentitySwap",
    "BoxKey",
    "KeyBoxSwap",
    "BallLava",
    "LavaBall",
]



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an interleaved MiniGrid object-remap curriculum config."
    )
    parser.add_argument("--seed", type=int, required=True, help="RL/run seed.")
    parser.add_argument("--output", required=True, help="Path to write the JSON config.")
    parser.add_argument("--num-families", type=int, default=2)
    parser.add_argument("--min-rooms", type=int, default=2)
    parser.add_argument("--max-rooms", type=int, default=5)
    parser.add_argument(
        "--profile-pool",
        nargs="+",
        default=DEFAULT_PROFILE_POOL,
        help="Remap profiles to sample from.",
    )
    parser.add_argument(
        "--with-replacement",
        action="store_true",
        help="Allow the same remap profile to be sampled for multiple families.",
    )
    parser.add_argument(
        "--env-seed-stride",
        type=int,
        default=1,
        help="Family i uses environment seed seed + i * env_seed_stride.",
    )
    parser.add_argument("--no-one-hot", dest="one_hot", action="store_false")
    parser.set_defaults(one_hot=True)
    parser.add_argument("--action-dim", type=int, default=4)
    return parser.parse_args()


def sample_profiles(rng, profile_pool, num_families, with_replacement):
    if not with_replacement and num_families > len(profile_pool):
        raise ValueError(
            "num-families cannot exceed profile-pool size without --with-replacement"
        )

    if with_replacement:
        return [rng.choice(profile_pool) for _ in range(num_families)]

    return rng.sample(profile_pool, num_families)


def build_config(args):
    rng = random.Random(args.seed)
    profiles = sample_profiles(
        rng,
        args.profile_pool,
        args.num_families,
        args.with_replacement,
    )

    tasks = []
    seeds = []
    for num_rooms in range(args.min_rooms, args.max_rooms + 1):
        for family_idx, profile in enumerate(profiles):
            tasks.append(f"CurriculumMultiRoomObjectRemapEnv-{profile}-R{num_rooms}-v0")
            seeds.append(args.seed + family_idx * args.env_seed_stride)

    return {
        "tasks": tasks,
        "one_hot": args.one_hot,
        "label_dim": len(tasks),
        "action_dim": args.action_dim,
        "seeds": seeds,
        "sampled_object_remap_profiles": profiles,
        "config_seed": args.seed,
    }


def main():
    args = parse_args()
    config = build_config(args)
    with open(args.output, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    print("sampled profiles:", ", ".join(config["sampled_object_remap_profiles"]))
    print("wrote:", args.output)


if __name__ == "__main__":
    main()
