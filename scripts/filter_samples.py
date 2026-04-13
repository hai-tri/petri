#!/usr/bin/env python3
"""
Filter samples from a Petri .eval log by score thresholds.

Usage:
    # Find samples where 'concerning' or 'needs_attention' scored >= 3:
    uv run python scripts/filter_samples.py \
        --log logs/baseline.eval \
        --categories "concerning,needs_attention" \
        --threshold 3 \
        --direction above

    # Find samples where 'admirable' scored below 5:
    uv run python scripts/filter_samples.py \
        --log logs/baseline.eval \
        --categories "admirable" \
        --threshold 5 \
        --direction below

    # Output as CSV:
    uv run python scripts/filter_samples.py \
        --log logs/baseline.eval \
        --categories "concerning,needs_attention,disappointing" \
        --threshold 3 --direction above \
        --output filtered.csv

    # List all available score categories:
    uv run python scripts/filter_samples.py \
        --log logs/baseline.eval \
        --list-categories
"""

import argparse
import csv
import json
import sys
import zipfile


def load_samples(eval_path: str) -> dict[str, dict]:
    """Load all samples and their scores from an .eval zip file."""
    samples = {}
    with zipfile.ZipFile(eval_path) as z:
        for name in z.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            data = json.loads(z.read(name))
            # Extract sample ID from filename (e.g. "samples/42_epoch_1.json" -> "42")
            sample_id = name.split("/")[1].split("_epoch_")[0]
            scores = {}
            for scorer, scorer_data in data.get("scores", {}).items():
                if isinstance(scorer_data, dict) and "value" in scorer_data:
                    val = scorer_data["value"]
                    if isinstance(val, dict):
                        scores.update(val)
            samples[sample_id] = scores
    return samples


def filter_samples(
    samples: dict[str, dict],
    categories: list[str],
    threshold: float,
    direction: str,
    match_mode: str = "any",
) -> dict[str, dict]:
    """Filter samples by score threshold.

    Args:
        match_mode: "any" = sample matches if ANY category meets threshold,
                    "all" = sample matches only if ALL categories meet threshold.
    """
    filtered = {}
    for sample_id, scores in samples.items():
        matches = []
        for cat in categories:
            val = scores.get(cat)
            if val is None:
                continue
            if direction == "above" and val >= threshold:
                matches.append(cat)
            elif direction == "below" and val < threshold:
                matches.append(cat)

        if match_mode == "any" and matches:
            filtered[sample_id] = scores
        elif match_mode == "all" and len(matches) == len(categories):
            filtered[sample_id] = scores

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter samples from a Petri .eval log by score thresholds"
    )
    parser.add_argument("--log", required=True, help="Path to the .eval log file")
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated list of score categories to filter on",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3,
        help="Score threshold (default: 3)",
    )
    parser.add_argument(
        "--direction",
        choices=["above", "below"],
        default="above",
        help="Filter for scores above or below threshold (default: above)",
    )
    parser.add_argument(
        "--match",
        choices=["any", "all"],
        default="any",
        help="Match if 'any' or 'all' categories meet threshold (default: any)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file path (prints to stdout if not set)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available score categories and exit",
    )

    args = parser.parse_args()
    samples = load_samples(args.log)

    if not samples:
        print("No samples found in log.", file=sys.stderr)
        sys.exit(1)

    # Collect all category names
    all_categories = set()
    for scores in samples.values():
        all_categories.update(scores.keys())

    if args.list_categories:
        print(f"Available categories ({len(all_categories)}):")
        for cat in sorted(all_categories):
            # Show value range across samples
            vals = [s.get(cat) for s in samples.values() if s.get(cat) is not None]
            if vals:
                print(f"  {cat}: min={min(vals)}, max={max(vals)}, mean={sum(vals)/len(vals):.1f}")
            else:
                print(f"  {cat}: no values")
        return

    if not args.categories:
        print("Error: --categories required (or use --list-categories)", file=sys.stderr)
        sys.exit(1)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    # Validate categories
    invalid = [c for c in categories if c not in all_categories]
    if invalid:
        print(f"Warning: unknown categories: {invalid}", file=sys.stderr)
        print(f"Available: {sorted(all_categories)}", file=sys.stderr)

    filtered = filter_samples(samples, categories, args.threshold, args.direction, args.match)

    # Sort by sample ID numerically
    def sort_key(sid):
        try:
            return int(sid)
        except ValueError:
            return sid

    sorted_ids = sorted(filtered.keys(), key=sort_key)

    print(f"\n{len(sorted_ids)}/{len(samples)} samples matched "
          f"({args.direction} {args.threshold} in {args.match} of {categories})\n")

    # Display results
    display_cats = categories + [c for c in sorted(all_categories) if c not in categories]
    header = ["Sample ID"] + categories
    rows = []
    for sid in sorted_ids:
        scores = filtered[sid]
        row = [sid] + [str(scores.get(c, "")) for c in categories]
        rows.append(row)

    # Print table
    col_widths = [max(len(header[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(header))]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*header))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))

    # Write CSV if requested
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
