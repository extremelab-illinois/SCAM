#!/usr/bin/env python3
"""Merge GitHub traffic API clones/views JSON into a running CSV history.

GitHub's /traffic/clones and /traffic/views endpoints only return the
trailing 14 days. This script upserts each day's row into a persistent CSV
(keyed by date), so repeated weekly runs stitch together a continuous
history instead of losing everything older than 14 days.
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone

FIELDS = ["date", "clones_count", "clones_uniques", "views_count", "views_uniques"]


def load_daily(path, key):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for entry in data.get(key, []):
        date = entry["timestamp"][:10]  # YYYY-MM-DD
        out[date] = (entry["count"], entry["uniques"])
    return out


def load_existing(csv_path):
    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rows[row["date"]] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clones", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    clones = load_daily(args.clones, "clones")
    views = load_daily(args.views, "views")
    rows = load_existing(args.csv)

    for date in set(clones) | set(views):
        c_count, c_uniq = clones.get(date, (0, 0))
        v_count, v_uniq = views.get(date, (0, 0))
        existing = rows.get(date, {})
        rows[date] = {
            "date": date,
            "clones_count": c_count or existing.get("clones_count", 0),
            "clones_uniques": c_uniq or existing.get("clones_uniques", 0),
            "views_count": v_count or existing.get("views_count", 0),
            "views_uniques": v_uniq or existing.get("views_uniques", 0),
        }

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for date in sorted(rows):
            writer.writerow(rows[date])

    last_7 = sorted(rows)[-7:]
    sum_clones = sum(int(rows[d]["clones_count"]) for d in last_7)
    sum_views = sum(int(rows[d]["views_count"]) for d in last_7)
    latest = rows[last_7[-1]] if last_7 else None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    summary_lines = [f"SCAM traffic snapshot ({today})", "Last 7 days:", f"  clones: {sum_clones}", f"  views:  {sum_views}"]
    if latest:
        summary_lines.append(
            f"Latest day ({latest['date']}): "
            f"{latest['clones_uniques']} unique cloners, "
            f"{latest['views_uniques']} unique visitors"
        )

    with open(args.summary_out, "w") as f:
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
