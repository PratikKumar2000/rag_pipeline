"""Stage a complete document tree and selected ZIP contents for review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile


EXCLUDED_DIRECTORIES = {"package", "artifacts"}
EXCLUDED_ARCHIVES = {"lib.zip"}


def excluded(path: Path | PurePosixPath) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool(parts & EXCLUDED_DIRECTORIES) or path.name.lower() in EXCLUDED_ARCHIVES


def extract_archive(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    count = 0
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            member_path = PurePosixPath(member.filename)
            if excluded(member_path):
                continue
            target = (destination / Path(*member_path.parts)).resolve()
            if not target.is_relative_to(root):
                print(f"Skipping unsafe archive entry: {member.filename}")
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=r"C:\rag_docs")
    parser.add_argument("--destination", default=r"C:\rag_source\curated")
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.destination)
    direct_root = destination / "direct"
    bundles_root = destination / "bundles"
    direct_root.mkdir(parents=True, exist_ok=True)
    bundles_root.mkdir(parents=True, exist_ok=True)

    archives = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".zip"
        and not excluded(path.relative_to(source))
    )
    manifest: list[dict[str, object]] = []
    direct_count = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() == ".zip":
            continue
        relative = path.relative_to(source)
        if excluded(relative):
            continue
        target = (direct_root / relative).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        direct_count += 1

    for number, archive in enumerate(archives, start=1):
        target = bundles_root / f"bundle_{number:04d}_{archive.stem}"
        try:
            count = extract_archive(archive, target)
            manifest.append(
                {
                    "archive": str(archive),
                    "output": str(target),
                    "files": count,
                    "status": "extracted",
                }
            )
            print(f"[{number}/{len(archives)}] {archive} -> {target} ({count} files)")
        except (OSError, zipfile.BadZipFile) as exc:
            manifest.append(
                {
                    "archive": str(archive),
                    "output": str(target),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[{number}/{len(archives)}] FAILED {archive}: {exc}")

    manifest_path = destination / "staging_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "source": str(source),
                "destination": str(destination),
                "direct_files": direct_count,
                "archives": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(f"Copied {direct_count} direct files and processed {len(archives)} ZIP archives.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
