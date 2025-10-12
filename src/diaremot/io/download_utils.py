"""Utility helpers for fetching remote assets with the standard library."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


def download_file(
    url: str, destination: Path, *, chunk_size: int = 1 << 20, timeout: int | float = 30
) -> None:
    """Download a file from ``url`` and atomically move it into ``destination``.

    Parameters
    ----------
    url:
        The URL pointing to the resource to download.
    destination:
        Path where the file will be stored. Parent directories are created
        automatically.
    chunk_size:
        Size (in bytes) for streamed chunks. Defaults to 1 MiB which balances
        throughput and memory usage for large model artefacts.
    timeout:
        Timeout (seconds) passed to :func:`urllib.request.urlopen`.

    The transfer streams to a temporary file in the destination directory to
    avoid partial writes on interruption.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with urlopen(url, timeout=timeout) as response:
            with NamedTemporaryFile(dir=str(destination.parent), delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                for chunk in iter(lambda: response.read(chunk_size), b""):
                    tmp_file.write(chunk)
        if tmp_path is None:
            raise RuntimeError("temporary download file was not created")
        os.replace(tmp_path, destination)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
