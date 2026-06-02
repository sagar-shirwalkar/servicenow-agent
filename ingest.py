#!/usr/bin/env python3
"""
One-time ingestion script with auto-clone support for ServiceNowDocs-australia.
Usage: uv run ingest.py [--force] [--repo-path ./ServiceNowDocs-australia]
"""

import argparse
import subprocess
from pathlib import Path
from data_pipeline import parse_and_categorize
from rag_system import index_data

DEFAULT_REPO_URL = "https://github.com/ServiceNow/ServiceNowDocs.git"
DEFAULT_BRANCH = "australia"
DEFAULT_LOCAL_PATH = "./ServiceNowDocs-australia"

def ensure_repo_cloned(repo_path: str, repo_url: str, branch: str) -> Path:
    """Clone or update the repo, returns resolved local path."""
    local = Path(repo_path)
    
    if local.exists() and (local / ".git").exists():
        print(f"🔄 Pulling latest {branch} branch...")
        subprocess.run(
            ["git", "-C", str(local), "pull", "origin", branch],
            check=True,
            capture_output=True
        )
    else:
        print(f"📥 Cloning {repo_url} (branch: {branch}) to {local}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "-b", branch, repo_url, str(local)],
            check=True
        )
    
    return local.resolve()

def main():
    parser = argparse.ArgumentParser(description="Ingest ServiceNowDocs into RAG")
    parser.add_argument("--repo-path", default=DEFAULT_LOCAL_PATH, help="Local path to repo")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="GitHub repo URL")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Git branch to clone")
    parser.add_argument("--force", action="store_true", help="Force re-indexing")
    args = parser.parse_args()
    
    # Ensure repo is available locally
    repo_path = ensure_repo_cloned(args.repo_path, args.repo_url, args.branch)
    print(f"📂 Repository ready at: {repo_path}")
    
    # Parse documentation
    print(f"\n🔍 Parsing documentation...")
    dataset = parse_and_categorize(str(repo_path))
    
    total = sum(len(v) for v in dataset.values())
    if total == 0:
        print("\n❌ DIAGNOSTIC: No chunks extracted!")
        print(f"   Checked: {repo_path}")
        md_count = len(list(repo_path.rglob("*.md")))
        print(f"   .md files found: {md_count}")
        if md_count > 0:
            print("   💡 Check data_pipeline.py categorization logic")
        raise RuntimeError("Ingestion failed: empty dataset")
    
    # Index into ChromaDB
    print(f"\n🔢 Indexing {total} chunks...")
    index_data(dataset, force=args.force)
    
    print(f"\n✅ Ingestion complete. RAG pipeline ready.")
    print(f"   • API chunks: {len(dataset['api'])}")
    print(f"   • Code chunks: {len(dataset['code'])}")
    print(f"   • Docs chunks: {len(dataset['docs'])}")

if __name__ == "__main__":
    main()
