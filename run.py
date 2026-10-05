"""Run the whole pipeline for any Google Play app, in its own workspace folder.

Usage:
    python run.py --package com.spotify.music
    python run.py --package com.spotify.music --competitor com.apple.android.music --competitor deezer.android.app

First run: looks the apps up on Google Play, creates workspaces/<package>/ with its own config, downloads
and cleans the reviews, and drafts a theme list from them. Then it stops, so you can read and edit
the themes (workspaces/<package>/config/themes.yaml) before any labeling happens.
Second run (same command): labels the reviews and prints the command that opens the dashboard.

Add --yes to skip the stop and do everything in one go. Add --refresh to download fresh reviews.
run.py only runs the same scripts you can run by hand (scrape, clean, suggest_themes, classify),
with RADAR_WORKSPACE pointing at the workspace folder.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from google_play_scraper import app as fetch_app_info

from common import ROOT

MAX_COMPETITORS = 2  # the dashboard's colors are checked for colorblind safety for up to 3 apps


def short_name(title: str) -> str:
    """'Spotify: Music and Podcasts' -> 'Spotify'."""
    return re.split(r"\s*[:\-–|]\s+", title.strip())[0].strip() or title.strip()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:30] or "app"


def describe(info: dict) -> str:
    """A one-line description for the labeling prompt, from the app's Play Store category."""
    genre = (info.get("genre") or "").strip().lower()
    if not genre:
        return "an app on Google Play"
    return f"{'an' if genre[0] in 'aeiou' else 'a'} {genre} app"


def create_workspace(workspace: Path, packages: list[str], args) -> None:
    """Write the workspace's apps.yaml and classifier.yaml from the Play Store listings."""
    apps, keys = [], set()
    for i, package in enumerate(packages):
        try:
            info = fetch_app_info(package, lang=args.lang, country=args.country)
        except Exception:
            raise SystemExit(f"Couldn't find '{package}' on Google Play ({args.country}). Check the package ID: "
                             "it's the id= part of the app's Play Store URL.")
        name = short_name(info["title"])
        key = slug(name)
        while key in keys:
            key += "-2"
        keys.add(key)
        apps.append({"key": key, "name": name, "package": package, "description": describe(info),
                     "role": "target" if i == 0 else "competitor"})
        print(f"  {'target' if i == 0 else 'competitor':<10} {name} ({package}) - {describe(info)}")

    config_dir = workspace / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    apps_yaml = {"defaults": {"lang": args.lang, "country": args.country, "max_reviews": args.max_reviews},
                 "apps": apps}
    (config_dir / "apps.yaml").write_text(
        "# Written by run.py. Edit freely: e.g. improve a description, then run classify.py again.\n"
        + yaml.safe_dump(apps_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    classifier = yaml.safe_load((ROOT / "config" / "classifier.yaml").read_text(encoding="utf-8"))
    classifier["reviews_per_app"] = args.label
    (config_dir / "classifier.yaml").write_text(
        "# Copied from the project defaults by run.py, with this workspace's sample size.\n"
        + yaml.safe_dump(classifier, sort_keys=False), encoding="utf-8")


def run_step(title: str, script: str, workspace: Path, *extra: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    env = {**os.environ, "RADAR_WORKSPACE": str(workspace)}
    result = subprocess.run([sys.executable, str(ROOT / script), *extra], env=env, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"\n{script} stopped (exit code {result.returncode}). Fix the problem above, then run "
                         "the same command again: finished work is kept.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Google Play reviews of any app.")
    parser.add_argument("--package", required=True, help="Play Store package ID of the app to study")
    parser.add_argument("--competitor", action="append", default=[], help="package ID of a competitor (up to 2)")
    parser.add_argument("--workspace", help="folder for this analysis (default: workspaces/<package>)")
    parser.add_argument("--max-reviews", type=int, default=2000, help="most recent reviews to download per app")
    parser.add_argument("--label", type=int, default=600, help="random reviews to label per app")
    parser.add_argument("--country", default="us", help="Play Store country (default: us)")
    parser.add_argument("--lang", default="en", help="review language (default: en)")
    parser.add_argument("--refresh", action="store_true", help="download fresh reviews even if some exist")
    parser.add_argument("--yes", action="store_true", help="don't stop to review the suggested themes")
    args = parser.parse_args()
    if len(args.competitor) > MAX_COMPETITORS:
        parser.error(f"at most {MAX_COMPETITORS} competitors")

    packages = [args.package, *args.competitor]
    if args.workspace:
        workspace = Path(args.workspace).resolve()
    else:
        workspace = ROOT / "workspaces" / re.sub(r"[^a-z0-9]+", "-", args.package.lower())  # com-spotify-music
    apps_file = workspace / "config" / "apps.yaml"

    if not apps_file.exists():
        print(f"Setting up workspace {workspace.relative_to(ROOT) if workspace.is_relative_to(ROOT) else workspace}")
        create_workspace(workspace, packages, args)
    apps = yaml.safe_load(apps_file.read_text(encoding="utf-8"))["apps"]

    have_data = all((workspace / "data" / "processed" / f"{a['key']}_clean.csv").exists() for a in apps)
    if args.refresh or not have_data:
        run_step("1/4 Download reviews", "scrape.py", workspace)
        run_step("2/4 Clean reviews", "clean.py", workspace)

    if not (workspace / "config" / "themes.yaml").exists():
        run_step("3/4 Draft a theme list", "suggest_themes.py", workspace)
        if not args.yes:
            themes = workspace / "config" / "themes.yaml"
            print(f"\nStopped so you can check the themes before labeling:\n  {themes}\n"
                  f"Edit them if needed (the reading samples in {workspace / 'data' / 'processed'} help), "
                  "then run the same command again.")
            return

    run_step("4/4 Label reviews", "classify.py", workspace)
    shown = workspace.relative_to(ROOT) if workspace.is_relative_to(ROOT) else workspace
    print(f"\nDone. Open the dashboard with:\n  RADAR_WORKSPACE={shown} .venv/bin/streamlit run app.py")


if __name__ == "__main__":
    main()
