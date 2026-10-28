"""Repair only the audited cu12 0.7.1 Linux AArch64 wheel's SBSA tag.

The published wheel filename is AArch64, but its internal WHEEL tag is SBSA.
Pip 25.3 installs it, then strict ``pip check`` rejects that internal tag.
This module runs in PartLLM's venv before pip check; it never suppresses pip.
The exact same non-Python native wheel is audited on CPython 3.11 and 3.12.
"""
from __future__ import annotations

import argparse
import base64
import csv
import ctypes
from contextlib import ExitStack
import hashlib
from importlib import metadata
import io
import os
from pathlib import Path
import platform
import secrets
import stat
import sys
import sysconfig


NAME = "nvidia-cusparselt-cu12"
VERSION = "0.7.1"
INFO = "nvidia_cusparselt_cu12-0.7.1.dist-info"
LIBRARY = "nvidia/cusparselt/lib/libcusparseLt.so.0"
ORIGINAL_WHEEL = (
    b"Wheel-Version: 1.0\n"
    b"Generator: setuptools (75.8.0)\n"
    b"Root-Is-Purelib: true\n"
    b"Tag: py3-none-manylinux2014_sbsa\n\n"
)
NORMALIZED_WHEEL = ORIGINAL_WHEEL.replace(b"manylinux2014_sbsa", b"manylinux2014_aarch64")
RECORD_ORDER = (
    "nvidia/cusparselt/LICENSE.txt",
    "nvidia/cusparselt/include/cusparseLt.h",
    LIBRARY,
    INFO + "/INSTALLER",
    INFO + "/METADATA",
    INFO + "/RECORD",
    INFO + "/WHEEL",
    INFO + "/top_level.txt",
)
# Audited official 0.7.1 cu12 AArch64 wheel file identities. A changed wheel
# must be re-audited rather than silently treated as the same exception.
FIXED_FILES = {
    "nvidia/cusparselt/LICENSE.txt": ("e8d158885a681b95ec7a6fc06dd8d4a52989f374cb1380c8a4c8fb27fd3d5d5e", 17948),
    "nvidia/cusparselt/include/cusparseLt.h": ("74580d3104ed58e1708d2ee746f65c2a9e1557f91b1d158d2d93c1591e118c38", 17876),
    LIBRARY: ("2c677e678d1955a6dedd66274dfe4cc0f930fea6421f1b8a5ec08cbb1ea18b17", 440496193),
    INFO + "/INSTALLER": ("ceebae7b8927a3227e5303cf5e0f1f7b34bb542ad7250ac03fbcde36ec2f1508", 4),
    INFO + "/METADATA": ("b264cea6951b70b52a3fd2ffaa88f9109eaf379633aa8d6e0832ecc92ddf6fba", 6974),
    INFO + "/top_level.txt": ("a1f202f9d6cad2cdb68cde79b207d5a5b847593eb765d24b59e49be8aff5f812", 18),
}
ORIGINAL_RECORD_SHA256 = "e02265be65d5f1aab74b97ffd2d53ca2ccf9d41ba0ca9cddf8d9b4bd34262425"
NORMALIZED_RECORD_SHA256 = "54b7063009a7cb86f1a0b2dd0bc0af777f8946440084365671e1745a1d61a976"


def _digest_fields(digest_hex: str, size: int) -> tuple[str, str]:
    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest_hex)).rstrip(b"=").decode("ascii")
    return "sha256=" + encoded, str(size)


def _record_for(wheel: bytes) -> bytes:
    if set(RECORD_ORDER) != set(FIXED_FILES) | {INFO + "/WHEEL", INFO + "/RECORD"} or len(RECORD_ORDER) != len(set(RECORD_ORDER)):
        raise RuntimeError("cuSPARSELt audited RECORD contract is inconsistent")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    for name in RECORD_ORDER:
        if name == INFO + "/RECORD":
            writer.writerow((name, "", ""))
        elif name == INFO + "/WHEEL":
            writer.writerow((name, *_digest_fields(hashlib.sha256(wheel).hexdigest(), len(wheel))))
        else:
            writer.writerow((name, *_digest_fields(*FIXED_FILES[name])))
    return output.getvalue().encode("ascii")


def _approved_site_root(dist) -> Path:
    """Do not repair distributions inherited through PYTHONPATH/system site."""
    prefix = Path(sys.prefix).resolve()
    if prefix == Path(sys.base_prefix).resolve():
        raise RuntimeError("cuSPARSELt repair requires the extension venv")
    root = Path(dist.locate_file("")).resolve()
    sites = {Path(sysconfig.get_path(kind)).resolve() for kind in ("purelib", "platlib")}
    if root not in sites or not root.is_relative_to(prefix):
        raise RuntimeError("cuSPARSELt or Torch distribution is outside the extension venv")
    return root


def _check_binding(parent: int, name: str, descriptor: int, *, directory: bool) -> None:
    linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
    opened = os.fstat(descriptor)
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(linked.st_mode) or not kind(opened.st_mode) or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino):
        raise RuntimeError(f"cuSPARSELt unsafe or changed path: {name}")


def _open_at(parent: int, name: str, *, directory: bool) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
        try:
            _check_binding(parent, name, descriptor, directory=directory)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    except OSError as exc:
        raise RuntimeError(f"cuSPARSELt missing, changed, or symlinked path: {name}") from exc


def _open_relative(site: int, relative: str, stack: ExitStack) -> tuple[int, int, str, tuple[tuple[int, str, int], ...]]:
    parts = Path(relative).parts
    if not parts or any(part in (".", "..") for part in parts):
        raise RuntimeError("cuSPARSELt invalid audited relative path")
    parent = site
    bindings = []
    for part in parts[:-1]:
        child = _open_at(parent, part, directory=True)
        stack.callback(os.close, child)
        bindings.append((parent, part, child))
        parent = child
    leaf = _open_at(parent, parts[-1], directory=False)
    stack.callback(os.close, leaf)
    return leaf, parent, parts[-1], tuple(bindings)


def _read_fd(descriptor: int, *, limit: int | None = None) -> bytes:
    size = os.fstat(descriptor).st_size
    if limit is not None and size > limit:
        raise RuntimeError("cuSPARSELt audited metadata file is unexpectedly large")
    data = bytearray()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise RuntimeError("cuSPARSELt audited file changed during read")
        data.extend(chunk)
        offset += len(chunk)
    return bytes(data)


def _verify_open_file(descriptor: int, parent: int, leaf: str, expected: tuple[str, int]) -> None:
    digest = hashlib.sha256()
    size = 0
    while size < expected[1]:
        chunk = os.pread(descriptor, min(1024 * 1024, expected[1] - size), size)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    _check_binding(parent, leaf, descriptor, directory=False)
    if (digest.hexdigest(), size, os.fstat(descriptor).st_size) != (expected[0], expected[1], expected[1]):
        raise RuntimeError(f"cuSPARSELt audited file integrity mismatch: {leaf}; reinstall before Repair")


def _revalidate_audited_files(audited) -> None:
    """Keep every verified file bound to its original site-rooted ancestry."""
    for descriptor, parent, leaf, bindings, identity in audited:
        for ancestor, name, child in bindings:
            _check_binding(ancestor, name, child, directory=True)
        _check_binding(parent, leaf, descriptor, directory=False)
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != identity:
            raise RuntimeError(f"cuSPARSELt audited file changed after verification: {leaf}")


def _atomic_write_at(directory: int, name: str, data: bytes, expected: int) -> None:
    """Publish through the already-open dist-info directory, never a path alias."""
    stage = f".{name}.partllm-{secrets.token_hex(12)}"
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=directory)
    try:
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fchmod(descriptor, stat.S_IMODE(os.fstat(expected).st_mode))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _check_binding(directory, name, expected, directory=False)
        os.replace(stage, name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        try:
            os.unlink(stage, dir_fd=directory)
        except FileNotFoundError:
            pass


def normalize_cusparselt_metadata(lane: str) -> bool:
    """Return True if repaired, False if inapplicable or already normalized."""
    if ((sys.platform, platform.machine().lower(), lane) != ("linux", "aarch64", "cu128") or
            tuple(sys.version_info[:2]) not in ((3, 11), (3, 12))):
        return False
    torch = metadata.distribution("torch")
    torch_root = _approved_site_root(torch)
    if torch.metadata.get_all("Name") != ["torch"] or torch.metadata.get_all("Version") != ["2.9.1+cu128"]:
        raise RuntimeError("Torch must be exactly 2.9.1+cu128 for the cuSPARSELt metadata repair")
    dist = metadata.distribution(NAME)
    if dist.metadata.get_all("Name") != [NAME] or dist.metadata.get_all("Version") != [VERSION]:
        raise RuntimeError("cuSPARSELt metadata repair requires the exact nvidia-cusparselt-cu12 0.7.1 distribution")
    root = _approved_site_root(dist)
    if root != torch_root:
        raise RuntimeError("cuSPARSELt and Torch must be installed in the same extension venv")
    original_record = _record_for(ORIGINAL_WHEEL)
    normalized_record = _record_for(NORMALIZED_WHEEL)
    if (hashlib.sha256(original_record).hexdigest() != ORIGINAL_RECORD_SHA256 or
            hashlib.sha256(normalized_record).hexdigest() != NORMALIZED_RECORD_SHA256):
        raise RuntimeError("cuSPARSELt audited RECORD contract hash mismatch")
    try:
        site = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise RuntimeError("cuSPARSELt venv site-packages is unsafe") from exc
    with ExitStack() as stack:
        stack.callback(os.close, site)
        dist_dir = _open_at(site, INFO, directory=True)
        stack.callback(os.close, dist_dir)
        wheel_fd = _open_at(dist_dir, "WHEEL", directory=False)
        stack.callback(os.close, wheel_fd)
        record_fd = _open_at(dist_dir, "RECORD", directory=False)
        stack.callback(os.close, record_fd)
        wheel, record = _read_fd(wheel_fd, limit=4096), _read_fd(record_fd, limit=65536)
        _check_binding(dist_dir, "WHEEL", wheel_fd, directory=False)
        _check_binding(dist_dir, "RECORD", record_fd, directory=False)
        if wheel == ORIGINAL_WHEEL and record == original_record:
            changed = True
        elif wheel == NORMALIZED_WHEEL and record in (original_record, normalized_record):
            # Normalized WHEEL + original RECORD is the only interrupted state.
            changed = record != normalized_record
        else:
            raise RuntimeError("cuSPARSELt WHEEL/RECORD differs from both audited states; reinstall before Repair")
        audited = []
        library_fd = None
        for relative, expected in FIXED_FILES.items():
            descriptor, parent, leaf, bindings = _open_relative(site, relative, stack)
            _verify_open_file(descriptor, parent, leaf, expected)
            info = os.fstat(descriptor)
            identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            audited.append((descriptor, parent, leaf, bindings, identity))
            if relative == LIBRARY:
                # Hash the same descriptor that remains open for dlopen below.
                library_fd = descriptor
        if library_fd is None:
            raise RuntimeError("cuSPARSELt native library is absent from the audited contract")
        header = os.pread(library_fd, 20, 0)
        if header[:7] != b"\x7fELF\x02\x01\x01" or header[16:20] != b"\x03\x00\xb7\x00":
            raise RuntimeError("cuSPARSELt library is not an ELF64 little-endian AArch64 shared object")
        try:
            ctypes.CDLL(f"/proc/self/fd/{library_fd}")
        except OSError as exc:
            raise RuntimeError("cuSPARSELt AArch64 native library failed to load") from exc
        _revalidate_audited_files(audited)
        _check_binding(site, INFO, dist_dir, directory=True)
        linked_site = os.stat(root, follow_symlinks=False)
        opened_site = os.fstat(site)
        if (linked_site.st_dev, linked_site.st_ino) != (opened_site.st_dev, opened_site.st_ino):
            raise RuntimeError("cuSPARSELt venv site-packages changed during repair")
        if changed:
            _revalidate_audited_files(audited)
            _check_binding(site, INFO, dist_dir, directory=True)
            if wheel != NORMALIZED_WHEEL:
                _atomic_write_at(dist_dir, "WHEEL", NORMALIZED_WHEEL, wheel_fd)
            with ExitStack() as check:
                new_wheel, _, _, _ = _open_relative(site, INFO + "/WHEEL", check)
                if _read_fd(new_wheel, limit=4096) != NORMALIZED_WHEEL:
                    raise RuntimeError("cuSPARSELt normalized WHEEL changed during repair")
            _revalidate_audited_files(audited)
            _check_binding(site, INFO, dist_dir, directory=True)
            _atomic_write_at(dist_dir, "RECORD", normalized_record, record_fd)
            with ExitStack() as check:
                new_record, _, _, _ = _open_relative(site, INFO + "/RECORD", check)
                if _read_fd(new_record, limit=65536) != normalized_record:
                    raise RuntimeError("cuSPARSELt normalized RECORD changed during repair")
            print("[PartLLM setup] Validated native cuSPARSELt 0.7.1; normalized audited WHEEL and RECORD", flush=True)
        return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    normalize_cusparselt_metadata(parser.parse_args().lane)
