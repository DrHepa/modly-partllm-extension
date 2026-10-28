"""The audited Linux ARM64 cu128 wheel metadata correction on CPython 3.11/3.12."""
import base64
import csv
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

import cusparselt_metadata as repair


def _field(data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return "sha256=" + digest, str(len(data))


@pytest.fixture(params=[(3, 11, 9), (3, 12, 3)], ids=["cp311", "cp312"])
def python_version(request):
    return request.param


@pytest.fixture
def wheel_tree(tmp_path, monkeypatch, python_version):
    """Tiny contract with the same layout and transition as the official wheel."""
    root = tmp_path / "site-packages"
    info = root / repair.INFO
    info.mkdir(parents=True)
    lib = root / repair.LIBRARY
    lib.parent.mkdir(parents=True)
    elf = b"\x7fELF\x02\x01\x01" + bytes(9) + b"\x03\x00\xb7\x00" + bytes(44)
    fixed = {
        repair.LIBRARY: elf,
        repair.INFO + "/METADATA": b"Name: nvidia-cusparselt-cu12\nVersion: 0.7.1\n",
    }
    for name, content in fixed.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    original = b"Wheel-Version: 1.0\nTag: py3-none-manylinux2014_sbsa\n"
    normalized = original.replace(b"manylinux2014_sbsa", b"manylinux2014_aarch64")
    wheel_key, record_key = repair.INFO + "/WHEEL", repair.INFO + "/RECORD"

    def record(data):
        out = io.StringIO(newline="")
        rows = [[key, *_field(value)] for key, value in fixed.items()]
        rows += [[wheel_key, *_field(data)], [record_key, "", ""]]
        csv.writer(out).writerows(rows)
        return out.getvalue().encode()

    original_record, normalized_record = record(original), record(normalized)
    (root / wheel_key).write_bytes(original)
    (root / record_key).write_bytes(original_record)
    monkeypatch.setattr(repair, "ORIGINAL_WHEEL", original)
    monkeypatch.setattr(repair, "NORMALIZED_WHEEL", normalized)
    monkeypatch.setattr(repair, "FIXED_FILES", {key: (hashlib.sha256(value).hexdigest(), len(value)) for key, value in fixed.items()})
    monkeypatch.setattr(repair, "RECORD_ORDER", (*fixed, wheel_key, record_key))
    monkeypatch.setattr(repair, "ORIGINAL_RECORD_SHA256", hashlib.sha256(original_record).hexdigest())
    monkeypatch.setattr(repair, "NORMALIZED_RECORD_SHA256", hashlib.sha256(normalized_record).hexdigest())
    monkeypatch.setattr(repair, "sys", SimpleNamespace(platform="linux", version_info=python_version,
        prefix=str(root.parent), base_prefix="/usr"))
    monkeypatch.setattr(repair.sysconfig, "get_path", lambda kind: str(root))
    monkeypatch.setattr(repair.platform, "machine", lambda: "aarch64")

    class Metadata:
        def __init__(self, name, version):
            self.name, self.version = name, version
        def get_all(self, key):
            return {"Name": [self.name], "Version": [self.version]}.get(key)

    class Dist:
        def __init__(self, name, version):
            self.metadata = Metadata(name, version)

        def locate_file(self, relative):
            return root / relative

    monkeypatch.setattr(repair.metadata, "distribution", lambda name: Dist(name, "2.9.1+cu128" if name == "torch" else repair.VERSION))
    loaded = []
    monkeypatch.setattr(repair.ctypes, "CDLL", lambda name: loaded.append(name))
    return root, original, normalized, original_record, normalized_record, loaded


def test_target_wheel_is_repaired_and_idempotent(wheel_tree):
    root, original, normalized, old_record, new_record, loaded = wheel_tree
    wheel = root / repair.INFO / "WHEEL"
    record = root / repair.INFO / "RECORD"
    assert repair.normalize_cusparselt_metadata("cu128") is True
    assert wheel.read_bytes() == normalized
    assert record.read_bytes() == new_record
    assert len(loaded) == 1 and loaded[0].startswith("/proc/self/fd/")
    assert repair.normalize_cusparselt_metadata("cu128") is False
    assert record.read_bytes() == new_record


def test_interrupted_write_recovers_only_known_original_record(wheel_tree):
    root, _, normalized, old_record, new_record, _ = wheel_tree
    (root / repair.INFO / "WHEEL").write_bytes(normalized)
    assert repair.normalize_cusparselt_metadata("cu128") is True
    assert (root / repair.INFO / "RECORD").read_bytes() == new_record


def test_interrupted_record_write_is_recoverable(wheel_tree, monkeypatch):
    root, _, normalized, _, new_record, _ = wheel_tree
    write = repair._atomic_write_at
    def interrupted(directory, name, data, expected):
        if name == "RECORD":
            raise OSError("simulated interrupted write")
        write(directory, name, data, expected)
    monkeypatch.setattr(repair, "_atomic_write_at", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (root / repair.INFO / "WHEEL").read_bytes() == normalized
    monkeypatch.setattr(repair, "_atomic_write_at", write)
    assert repair.normalize_cusparselt_metadata("cu128") is True
    assert (root / repair.INFO / "RECORD").read_bytes() == new_record


def test_reverse_partial_state_is_rejected(wheel_tree):
    root, original, _, _, new_record, _ = wheel_tree
    (root / repair.INFO / "RECORD").write_bytes(new_record)
    with pytest.raises(RuntimeError, match="WHEEL/RECORD"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (root / repair.INFO / "WHEEL").read_bytes() == original


@pytest.mark.parametrize("name,content", [
    ("WHEEL", b"Tag: py3-none-any\n"),
    ("RECORD", b"malformed,,\n"),
    ("METADATA", b"tampered"),
])
def test_unknown_metadata_is_not_repaired(wheel_tree, name, content):
    root, *_ = wheel_tree
    path = root / repair.INFO / name
    path.write_bytes(content)
    with pytest.raises(RuntimeError, match="cuSPARSELt"):
        repair.normalize_cusparselt_metadata("cu128")


def test_corrupted_elf_is_fatal_before_wheel_change(wheel_tree):
    root, original, *_ = wheel_tree
    lib = root / repair.LIBRARY
    lib.write_bytes(b"not an AArch64 ELF")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (root / repair.INFO / "WHEEL").read_bytes() == original


def test_native_loader_failure_is_fatal_before_wheel_change(wheel_tree, monkeypatch):
    root, original, *_ = wheel_tree
    def failed(_):
        raise OSError("missing native dependency")
    monkeypatch.setattr(repair.ctypes, "CDLL", failed)
    with pytest.raises(RuntimeError, match="failed to load"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (root / repair.INFO / "WHEEL").read_bytes() == original


def test_swap_after_hash_cannot_load_unverified_library(wheel_tree, monkeypatch):
    root, original_wheel, *_ = wheel_tree
    library = root / repair.LIBRARY
    verified_library = library.read_bytes()
    verify = repair._verify_open_file
    def swap_after_hash(descriptor, parent, leaf, expected):
        verify(descriptor, parent, leaf, expected)
        if leaf == library.name:
            library.rename(library.with_name("original.so"))
            library.write_bytes(b"attacker-controlled library")
    monkeypatch.setattr(repair, "_verify_open_file", swap_after_hash)
    loaded = []
    monkeypatch.setattr(repair.ctypes, "CDLL", lambda path: loaded.append(Path(path).read_bytes()))
    with pytest.raises(RuntimeError, match="changed path"):
        repair.normalize_cusparselt_metadata("cu128")
    assert loaded == [verified_library]
    assert (root / repair.INFO / "WHEEL").read_bytes() == original_wheel


def test_swapped_library_ancestor_blocks_metadata_write(wheel_tree, monkeypatch):
    root, original_wheel, *_ = wheel_tree
    library = root / repair.LIBRARY
    verified_library = library.read_bytes()
    verify = repair._verify_open_file
    def swap_directory_after_hash(descriptor, parent, leaf, expected):
        verify(descriptor, parent, leaf, expected)
        if leaf == library.name:
            original_dir = library.parent.with_name("lib-verified")
            library.parent.rename(original_dir)
            library.parent.mkdir()
            library.write_bytes(b"unverified library")
    monkeypatch.setattr(repair, "_verify_open_file", swap_directory_after_hash)
    loaded = []
    monkeypatch.setattr(repair.ctypes, "CDLL", lambda path: loaded.append(Path(path).read_bytes()))
    with pytest.raises(RuntimeError, match="changed path"):
        repair.normalize_cusparselt_metadata("cu128")
    assert loaded == [verified_library]
    assert (root / repair.INFO / "WHEEL").read_bytes() == original_wheel


def test_non_target_lane_never_looks_up_distribution(monkeypatch, python_version):
    monkeypatch.setattr(repair, "sys", SimpleNamespace(platform="linux", version_info=python_version))
    monkeypatch.setattr(repair.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(repair.metadata, "distribution", lambda _: pytest.fail("distribution queried"))
    assert repair.normalize_cusparselt_metadata("cpu") is False
    assert repair.normalize_cusparselt_metadata("cu130") is False


def test_wrong_torch_version_does_not_change_metadata(wheel_tree, monkeypatch):
    root, original, *_ = wheel_tree
    distribution = repair.metadata.distribution
    monkeypatch.setattr(repair.metadata, "distribution", lambda name: distribution(name if name != "torch" else "other"))
    with pytest.raises(RuntimeError, match="Torch"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (root / repair.INFO / "WHEEL").read_bytes() == original


@pytest.mark.parametrize("unsupported_version", [(3, 10, 16), (3, 13, 1)], ids=["cp310", "cp313"])
def test_other_python_abis_are_rejected_before_distribution_lookup(wheel_tree, monkeypatch, unsupported_version):
    root, original, *_ = wheel_tree
    monkeypatch.setattr(repair, "sys", SimpleNamespace(platform="linux", version_info=unsupported_version))
    monkeypatch.setattr(repair.metadata, "distribution", lambda _: pytest.fail("distribution queried"))
    assert repair.normalize_cusparselt_metadata("cu128") is False
    assert (root / repair.INFO / "WHEEL").read_bytes() == original


def test_external_distribution_root_is_rejected(wheel_tree, monkeypatch):
    root, original, *_ = wheel_tree
    monkeypatch.setattr(repair, "sys", SimpleNamespace(platform="linux", version_info=repair.sys.version_info,
        prefix=str(root.parent / "other-venv"), base_prefix="/usr"))
    with pytest.raises(RuntimeError, match="venv"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (root / repair.INFO / "WHEEL").read_bytes() == original


def test_external_torch_distribution_is_rejected(wheel_tree, monkeypatch):
    root, original, *_ = wheel_tree
    distribution = repair.metadata.distribution
    class ExternalTorch:
        metadata = distribution("torch").metadata
        def locate_file(self, relative):
            return root.parent / "external" / relative
    monkeypatch.setattr(repair.metadata, "distribution", lambda name:
        ExternalTorch() if name == "torch" else distribution(name))
    with pytest.raises(RuntimeError, match="venv"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (root / repair.INFO / "WHEEL").read_bytes() == original


def test_external_cusparselt_distribution_is_rejected(wheel_tree, monkeypatch):
    root, original, *_ = wheel_tree
    distribution = repair.metadata.distribution
    class ExternalCuSparseLt:
        metadata = distribution(repair.NAME).metadata
        def locate_file(self, relative):
            return root.parent / "external" / relative
    monkeypatch.setattr(repair.metadata, "distribution", lambda name:
        ExternalCuSparseLt() if name == repair.NAME else distribution(name))
    with pytest.raises(RuntimeError, match="venv"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (root / repair.INFO / "WHEEL").read_bytes() == original


def test_symlinked_dist_info_is_rejected(wheel_tree):
    root, original, *_ = wheel_tree
    info = root / repair.INFO
    moved = root / "other.dist-info"
    info.rename(moved)
    info.symlink_to(moved, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (moved / "WHEEL").read_bytes() == original


def test_symlinked_library_is_rejected(wheel_tree):
    root, original, *_ = wheel_tree
    lib = root / repair.LIBRARY
    copy = lib.with_name("other.so")
    lib.rename(copy)
    lib.symlink_to(copy)
    with pytest.raises(RuntimeError, match="symlink"):
        repair.normalize_cusparselt_metadata("cu128")
    assert (root / repair.INFO / "WHEEL").read_bytes() == original
