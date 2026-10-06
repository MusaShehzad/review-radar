"""Publish the dashboard as a Hugging Face Space (free hosting with a public link).

One-time setup: create a free account at huggingface.co, make an access token with "write" access
(Settings -> Access Tokens), then log in once:
    .venv/bin/hf auth login

Publish (run again any time to update the Space with your latest data and code):
    .venv/bin/python deploy_hf.py
    .venv/bin/python deploy_hf.py --name my-space-name

Only what the dashboard needs is uploaded: the code, config, labeled data and store snapshots.
Your .env (API keys), raw downloads and the label cache stay on your computer.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from common import ROOT

FILES = ["app.py", "ui.py", "analyze.py", "common.py", "requirements.txt", "Dockerfile", ".streamlit/config.toml"]
FOLDERS = {"config": "*.yaml", "data/processed": "*.csv", "data/raw": "*_app.json"}
EXTRA = ["data/processed/validation_results.json"]  # shown on the Method page once it exists

SPACE_README = """---
title: Review Radar
emoji: 📡
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: What Google Play reviewers say about ImagineArt vs rivals
---

Dashboard of LLM-labeled Google Play reviews: themes, release impact and competitor gaps.
Code and method: https://github.com/{github}
"""


def stage(folder: Path, github: str) -> None:
    """Copy the files the dashboard needs into `folder`, plus the Space's README."""
    for name in FILES + [p for p in EXTRA if (ROOT / p).exists()]:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for sub, pattern in FOLDERS.items():
        for path in (ROOT / sub).glob(pattern):
            target = folder / sub / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    (folder / "README.md").write_text(SPACE_README.format(github=github), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the dashboard as a Hugging Face Space.")
    parser.add_argument("--name", default="review-radar", help="Space name (default: review-radar)")
    parser.add_argument("--github", default="MusaShehzad/review-radar", help="GitHub repo linked from the Space")
    args = parser.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        raise SystemExit("Not logged in to Hugging Face. Run: .venv/bin/hf auth login")
    space = f"{user}/{args.name}"

    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp), args.github)
        files = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*") if p.is_file())
        print(f"Uploading {len(files)} files to {space}...")
        api.upload_folder(repo_id=space, repo_type="space", folder_path=tmp,
                          commit_message="Update dashboard", delete_patterns=["data/**", "config/*.yaml"])  # drop stale data
    print(f"\nDone. The Space builds in a few minutes:\n  https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
