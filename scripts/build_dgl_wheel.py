#!/usr/bin/env python3
"""Build a pip/uv-installable CUDA wheel of DGL from the official dglteam conda package.

Why: DGL stopped publishing CUDA wheels on PyPI, and its own wheel index
(data.dgl.ai) is unreachable from our machines (TLS error). The conda channel on
anaconda.org has the same official builds, so we repack one as a wheel.

Usage (stdlib only, run before ``uv sync``):

    python3 scripts/build_dgl_wheel.py            # -> wheels/dgl-2.4.0+cu121-cp311-cp311-linux_x86_64.whl
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

DGL_VERSION = "2.4.0"
TORCH_TAG = "th23"
CUDA_TAG = "cu121"
PY_TAG = "py311"
URL = (
    f"https://conda.anaconda.org/dglteam/label/{TORCH_TAG}_{CUDA_TAG}/linux-64/"
    f"dgl-{DGL_VERSION}.{TORCH_TAG}.{CUDA_TAG}-{PY_TAG}_0.tar.bz2"
)
SITE = "lib/python3.11/site-packages/"


def download(url: str, dest: Path, retries: int = 20) -> None:
    """Download with HTTP range resume (the connection drops on large files)."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers["Content-Length"])
    if dest.exists() and dest.stat().st_size == total:
        return
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except Exception as e:  # network hiccup: resume
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(2)
        print(f"  {dest.stat().st_size / 1e6:.0f} / {total / 1e6:.0f} MB")
    if dest.stat().st_size != total:
        raise SystemExit("download incomplete")


def record_line(name: str, data: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [name, f"sha256={digest}", str(len(data))]


def requires_dist(requires_txt: str) -> list[str]:
    out, extra = [], None
    for line in requires_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            extra = line.strip("[]")
            continue
        out.append(f"{line}; extra == '{extra}'" if extra else line)
    return out


def build(pkg: Path, out_dir: Path) -> Path:
    local_version = f"{DGL_VERSION}+{CUDA_TAG}"
    dist_info = f"dgl-{local_version}.dist-info"
    wheel_path = out_dir / f"dgl-{local_version}-cp311-cp311-linux_x86_64.whl"
    records: list[list[str]] = []
    pkg_info = requires = ""
    with tarfile.open(pkg, "r:bz2") as tar, zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as whl:
        for m in tar.getmembers():
            if not m.isfile() or not m.name.startswith(SITE):
                continue
            rel = m.name[len(SITE) :]
            data = tar.extractfile(m).read()
            if ".egg-info/" in rel:
                if rel.endswith("PKG-INFO"):
                    pkg_info = data.decode()
                elif rel.endswith("requires.txt"):
                    requires = data.decode()
                continue
            if rel.endswith(".pyc") or "/__pycache__/" in rel:
                continue
            info = zipfile.ZipInfo(rel, date_time=(2024, 1, 1, 0, 0, 0))
            info.external_attr = (m.mode & 0o777 | 0o100000) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            whl.writestr(info, data)
            records.append(record_line(rel, data))

        meta_lines = [ln for ln in pkg_info.splitlines() if not ln.startswith(("Version:", "Requires-Dist:"))]
        if not any(ln.startswith("Metadata-Version") for ln in meta_lines):
            meta_lines = ["Metadata-Version: 2.1", "Name: dgl", *meta_lines]
        meta_lines.insert(2, f"Version: {local_version}")
        meta_lines[3:3] = [f"Requires-Dist: {r}" for r in requires_dist(requires)]
        metadata = ("\n".join(meta_lines) + "\n").encode()
        wheel_meta = b"Wheel-Version: 1.0\nGenerator: baed-build-dgl-wheel\nRoot-Is-Purelib: false\nTag: cp311-cp311-linux_x86_64\n"
        for name, data in [(f"{dist_info}/METADATA", metadata), (f"{dist_info}/WHEEL", wheel_meta)]:
            whl.writestr(name, data)
            records.append(record_line(name, data))
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows([*records, [f"{dist_info}/RECORD", "", ""]])
        whl.writestr(f"{dist_info}/RECORD", buf.getvalue())
    return wheel_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "wheels"))
    parser.add_argument("--cache", default=None, help="where to keep the downloaded conda package")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkg = Path(args.cache) if args.cache else out_dir / Path(URL).name
    print(f"downloading {URL}")
    download(URL, pkg)
    whl = build(pkg, out_dir)
    print(f"built {whl} ({whl.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
