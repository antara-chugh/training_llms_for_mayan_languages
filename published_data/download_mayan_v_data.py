"""
Download MayanV (https://github.com/transducens/mayanv) into a local folder, "mayan_data_raw".


"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/transducens/mayanv"
DEFAULT_DEST = Path("./mayan_data_raw")


def download(dest: Path, force: bool = False) -> None:
    if dest.exists() and any(dest.iterdir()):
        if not force:
            print(f"{dest} already exists and is non-empty — skipping.")
            print("Pass --force to re-download.")
            return
        print(f"--force set: removing existing {dest}")
        shutil.rmtree(dest)

    if shutil.which("git") is None:
        print(
            "ERROR: git is not installed.\n"
            f"Manually download {REPO_URL} (e.g. via 'Download ZIP' on GitHub) "
            f"and extract it to: {dest}"
        )
        sys.exit(1)

    print(f"Cloning {REPO_URL} into {dest} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(dest)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("ERROR: git clone failed.")
        print(result.stderr)
        sys.exit(1)

    print(f"\nDone. Repo cloned to: {dest.resolve()}")
    print(f"Q'anjob'al data should be under: {dest.resolve() / 'MayanV' / 'kjb'}")
    print("All files are left as-is — build_mayanv_en.py handles translation/alignment downstream.")


def main():
    parser = argparse.ArgumentParser(description="Download the MayanV corpus.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination folder")
    parser.add_argument("--force", action="store_true", help="Re-download even if dest exists")
    args = parser.parse_args()
    download(args.dest, force=args.force)


if __name__ == "__main__":
    main()