"""Download and verify the published FP8 transformer release assets."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


BASE_URL = (
    "https://github.com/ALX-CODE/lingbot-video-1.3b-fp8/"
    "releases/latest/download"
)
ASSETS = {
    "config.json": "4c010f3ea6236b7317d46bcea1fc6c99f53fe743b19cfdb8be6c0ac326acec47",
    "diffusion_pytorch_model.safetensors": (
        "7815ce190cbbed4859a6270e441af460c3f758b9b2adec21bacb362094850921"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, destination: Path) -> None:
    final = destination / name
    if final.is_file() and sha256(final) == ASSETS[name]:
        print(f"Verified existing {final}")
        return

    partial = final.with_suffix(final.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(f"{BASE_URL}/{name}")
    if downloaded:
        request.add_header("Range", f"bytes={downloaded}-")

    with urllib.request.urlopen(request) as response:
        if downloaded and response.status != 206:
            downloaded = 0
            partial.unlink(missing_ok=True)
        mode = "ab" if downloaded else "wb"
        total = response.headers.get("Content-Length")
        total_bytes = downloaded + int(total) if total else None
        with partial.open(mode) as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    print(
                        f"\r{name}: {downloaded / 1e9:.2f}/{total_bytes / 1e9:.2f} GB "
                        f"({100 * downloaded / total_bytes:.1f}%)",
                        end="",
                        flush=True,
                    )
        print()

    actual = sha256(partial)
    if actual != ASSETS[name]:
        raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
    partial.replace(final)
    print(f"Verified {final}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Complete official LingBot Dense 1.3B model directory",
    )
    args = parser.parse_args()
    destination = args.model_dir.expanduser().resolve() / "transformer_fp8_dense"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        download(name, destination)
    print("FP8 transformer is ready:", destination)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Download interrupted; rerun the command to resume.")
