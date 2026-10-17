"""Hugging Face Hub uploads: one local folder → one repo (model or Space), created public if missing."""
from pathlib import Path

IGNORE = ["__pycache__/*", "*.pyc", ".DS_Store"]


def listing(folder: Path) -> list[tuple[str, int]]:
    return [(str(p.relative_to(folder)), p.stat().st_size) for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".DS_Store"]


def publish(repo_id: str, folder: Path, repo_type: str, dry_run: bool, message: str) -> str:
    """→ the commit URL ("" for a dry run). Prints what is uploaded."""
    files = listing(folder)
    print(f"{repo_type} {repo_id} ← {folder} ({len(files)} files, {sum(s for _, s in files) / 1e6:.1f} MB)")
    for name, size in files:
        print(f"  {size / 1e6:9.1f} MB  {name}")
    if dry_run:
        return ""
    from huggingface_hub import create_repo, upload_folder

    create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False,
                **({"space_sdk": "gradio"} if repo_type == "space" else {}))
    commit = upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type=repo_type, commit_message=message,
                           ignore_patterns=IGNORE)
    print(f"  → {commit.commit_url}")
    return commit.commit_url
