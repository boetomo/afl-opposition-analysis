"""Download a recorded torpdata snapshot without modifying its contents."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


def verify_file(path: Path, source: dict) -> None:
    """Reject incomplete files or a different upstream snapshot."""
    if path.stat().st_size != source["size_bytes"]:
        raise ValueError(f"Size mismatch for {path}; expected {source['size_bytes']} bytes.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}; the file differs from data/sources.json.")


def download_season(
    season: int,
    manifest_path: Path = Path("data/sources.json"),
    output_dir: Path = Path("data/raw"),
) -> Path:
    """Reuse a verified local file, or download and verify it before saving."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["seasons"].get(str(season))
    if source is None:
        raise ValueError(f"Season {season} has no recorded source in {manifest_path}.")

    destination = output_dir / source["filename"]
    if destination.exists():
        verify_file(destination, source)
        print(f"Using verified local file: {destination.name}")
        return destination

    output_dir.mkdir(parents=True, exist_ok=True)
    request = Request(source["url"], headers={"User-Agent": "afl-opposition-analysis/0.1"})
    temporary_path = None
    try:
        # A failed download must never appear under the final dataset filename.
        with NamedTemporaryFile(dir=output_dir, suffix=".part", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, temporary)
        verify_file(temporary_path, source)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Downloaded and verified: {destination.name}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2023)
    args = parser.parse_args()
    try:
        download_season(args.season)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
