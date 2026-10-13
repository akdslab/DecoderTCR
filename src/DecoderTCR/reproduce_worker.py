"""Execute one reproduction with the installed package."""

import argparse

from .core import ValidationError
from .reproduction import merge_reproduction_shards, reproduce, reproduce_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_id")
    parser.add_argument("--mode", choices=("metrics", "full"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--merge-shards", nargs="+")
    parser.add_argument("--parent-output")
    args = parser.parse_args()
    if (args.shard_index is None) != (args.shard_count is None):
        raise ValidationError("Shard index and count must be supplied together")
    if args.mode != "full" and (
        args.shard_index is not None or args.merge_shards is not None or args.parent_output
    ):
        raise ValidationError("Shard and parent-output operations require full mode")
    if args.shard_index is not None and args.merge_shards is not None:
        raise ValidationError("Choose a shard worker or a shard merge")
    if args.parent_output and (args.shard_index is not None or args.merge_shards is not None):
        raise ValidationError("A parent output cannot be combined with shard operations")
    if args.shard_index is not None:
        reproduce_shard(
            args.result_id,
            args.shard_index,
            args.shard_count,
            args.output,
        )
    elif args.merge_shards is not None:
        merge_reproduction_shards(args.result_id, args.merge_shards, args.output)
    else:
        reproduce(
            args.result_id,
            mode=args.mode,
            output=args.output,
            parent_output=args.parent_output,
        )


if __name__ == "__main__":
    main()
