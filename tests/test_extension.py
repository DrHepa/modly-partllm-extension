"""Pure contract/geometry tests. These do not claim model or CUDA inference proof."""
import importlib.util
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import partllm_config as config
import processor
import native_runtime
from geometry import export_partition, flatten_mesh, part_names
from weights import safe_file, validate_weights

spec = importlib.util.spec_from_file_location("partllm_installer", ROOT / "setup.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class ConfigTests(unittest.TestCase):
    def test_manifest_defaults_and_types_match(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(manifest["type"], "process")
        for node in manifest["nodes"]:
            self.assertEqual(node["params_schema"], config.SCHEMAS[node["id"]])
            values = config.normalize_params(node["id"], {})
            self.assertEqual(values, {x["id"]: x["default"] for x in node["params_schema"]})
            self.assertTrue(all(x["type"] in {"int", "float", "select", "string"} for x in node["params_schema"]))
        self.assertEqual(manifest["nodes"][1]["inputs"], ["mesh", "text"])

    def test_invalid_params(self):
        for key, value in [("top_p", float("nan")), ("temperature", math.inf), ("seed", -1),
                           ("seed", "2.5"), ("graph_cut", "maybe"), ("max_new_tokens", True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                config.normalize_params("full-shape", {key: value})
        with self.assertRaises(ValueError):
            config.normalize_params("unknown", {})

    def test_boolean_select_normalization(self):
        self.assertEqual(config.normalize_params("full-shape", {"graph_cut": False})["graph_cut"], "false")

    def test_primary_output_respects_granularity_for_semantic_and_nosem(self):
        exports = [{"selection": name} for name in ["coarse", "general", "nosem_coarse", "nosem_general"]]
        self.assertEqual(processor.select_primary(exports, "general")["selection"], "general")
        self.assertEqual(processor.select_primary(exports[2:], "general")["selection"], "nosem_general")
        self.assertEqual(processor.select_primary([{"selection": "num_parts_3"}], "general")["selection"], "num_parts_3")

    def test_command_arguments_are_not_shell_fragments(self):
        values = config.normalize_params("text-guided", {"part_names": "seat; chair back; $(touch nope)"})
        command = processor.build_command("text-guided", values, Path("/venv/python"), Path("/vendor"),
                                          Path("/models"), Path("/input/input.glb"), Path("/output"), Path("/config"))
        self.assertEqual(command[-3:], ["seat", "chair back", "$(touch nope)"])
        self.assertEqual(command[command.index("--save_mesh") + 1], "true")

    def test_legacy_requires_numeric_parts(self):
        values = config.normalize_params("full-shape", {"prompt_style": "legacy"})
        with self.assertRaisesRegex(ValueError, "requires num_parts"):
            processor.build_command("full-shape", values, *([Path("/a")] * 6))

    def test_setup_forms(self):
        payload = dict(python_exe="/python", ext_dir="/extension", gpu_sm=121)
        self.assertEqual(installer.parse_args(["setup.py", json.dumps(payload)]), payload)
        self.assertEqual(installer.parse_args(["setup.py", "/python", "/extension", "121", "128"])["gpu_sm"], 121)
        for value in [128, "128", "cu128", "12.8"]:
            self.assertEqual(installer.normalize_cuda(value), (12, 8))

    def test_cuda_torch_stack_is_architecture_specific(self):
        self.assertEqual(installer.torch_versions({"machine": "aarch64"}), ("2.9.1", "0.24.1"))
        self.assertEqual(installer.torch_versions({"machine": "x86_64"}), ("2.8.0", "0.23.0"))

    def test_setup_metadata_repair_gate_targets_only_audited_arm_python_abis(self):
        for machine in ["aarch64", "arm64", "x86_64"]:
            for version in [[3, 10], [3, 11], [3, 12], [3, 13]]:
                with self.subTest(machine=machine, version=version), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    python = root / "venv/bin/python"
                    info = {"machine": machine, "version": version}
                    payload = dict(python_exe=str(python), ext_dir=str(root))
                    with patch.object(installer, "__file__", str(root / "setup.py")), \
                            patch.object(installer, "resolve_models_dir", return_value=root / "models"), \
                            patch.object(installer, "ensure_venv", return_value=(python, info)), \
                            patch.object(installer, "checkout_source"), \
                            patch.object(installer, "install_native", return_value={}), \
                            patch.object(installer, "native_environment", return_value={}), \
                            patch.object(installer, "atomic_json"), patch.object(installer, "run") as run:
                        installer.main(["setup.py", json.dumps(payload)])
                    commands = [call.args[0] for call in run.call_args_list]
                    repair_command = [python, root / "cusparselt_metadata.py", "--lane", "cu128"]
                    expected = machine in {"aarch64", "arm64"} and version in [[3, 11], [3, 12]]
                    self.assertEqual(commands.count(repair_command), int(expected))
                    if expected:
                        self.assertLess(commands.index(repair_command), commands.index([python, "-m", "pip", "check"]))

    def test_pyg_source_install_does_not_reuse_unknown_torch_abi_cache(self):
        pip = ["/private/python", "-m", "pip", "install"]
        command = installer.pyg_source_command(pip, "torch-scatter==2.1.2")
        self.assertEqual(command[:4], pip)
        for flag in ["--no-cache-dir", "--force-reinstall", "--no-deps", "--no-build-isolation", "--no-binary=torch-scatter"]:
            self.assertIn(flag, command)
        self.assertEqual(pip, ["/private/python", "-m", "pip", "install"])

    def test_pyg_environment_survives_host_cli_cache_rewrite(self):
        environment = {"PIP_CACHE_DIR": "/shared/cache", "PATH": "/usr/bin"}
        child = installer.pyg_source_environment(environment)
        self.assertEqual(child["PIP_NO_CACHE_DIR"], "1")
        self.assertEqual(child["FORCE_CUDA"], "1")
        self.assertNotIn("PIP_NO_CACHE_DIR", environment)

    def test_models_resolution_requires_matching_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            settings = base / "settings.json"
            settings.write_text(json.dumps(dict(extensionsDir=str(base / "extensions"), modelsDir=str(base / "models"), hfToken="DO-NOT-LOG")))
            with patch.object(config, "settings_candidates", return_value=iter([settings])):
                self.assertEqual(config.resolve_models_dir({}, base / "extensions/ext"), base / "models")
            with patch.object(config, "settings_candidates", return_value=iter([settings])), self.assertRaises(ValueError):
                config.resolve_models_dir({}, base / "other/ext")
            with self.assertRaises(ValueError):
                config.resolve_models_dir({"models_dir": "relative"}, base)

    def test_state_is_root_and_revision_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with self.assertRaises(RuntimeError):
                config.load_state(base)
            state = dict(extension_dir=str(base.resolve()), upstream_revision=config.UPSTREAM_REVISION,
                         weights_revision=config.WEIGHTS_REVISION, ready=True)
            (base / config.STATE_NAME).write_text(json.dumps(state))
            self.assertTrue(config.load_state(base)["ready"])
            state["extension_dir"] = "/other"
            (base / config.STATE_NAME).write_text(json.dumps(state))
            with self.assertRaisesRegex(RuntimeError, "different"):
                config.load_state(base)

    def test_weight_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            safe_file(Path("/tmp/checkpoint"), "../../outside")

    def test_public_checkpoint_ignores_unrelated_global_login(self):
        import weights
        from unittest.mock import Mock
        info = SimpleNamespace(sha=config.WEIGHTS_REVISION,
                               siblings=[SimpleNamespace(rfilename="config.json", size=12, lfs=None)])
        api = Mock()
        api.return_value.model_info.return_value = info
        snapshot = Mock()
        hub = SimpleNamespace(HfApi=api, snapshot_download=snapshot)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": hub}), \
                patch.object(weights, "validate_weights"):
            weights.download_weights(Path(directory))
        api.assert_called_once_with(token=False)
        self.assertFalse(snapshot.call_args.kwargs["token"])


class GeometryTests(unittest.TestCase):
    def test_current_upstream_json_responses_use_semantic_label_not_bbox_json(self):
        response = '{"bbox_aabb": [-1, 0, 0, 1, 1, 1], "label": "Seat"}<|SEG|>\n{"bbox_aabb": [], "label": "Backrest"}<|SEG|>\n<|BG|>'
        self.assertEqual(part_names({"response": response}), ["Seat", "Backrest"])
        self.assertEqual(part_names({"response": "Leg<|SEG|>\nBase<|SEG|>"}), ["Leg", "Base"])

    def setUp(self):
        import numpy as np
        import trimesh
        self.np, self.trimesh = np, trimesh
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def scene_input(self):
        scene = self.trimesh.Scene()
        box = self.trimesh.creation.box()
        scene.add_geometry(box, geom_name="shared", node_name="first")
        transform = self.np.eye(4)
        transform[:3, 3] = [4, 2, -1]
        scene.graph.update(frame_to="second", geometry="shared", matrix=transform)
        source = self.root / "mesh with spaces.glb"
        scene.export(source)
        return source

    def test_instances_and_world_coordinates_preserved(self):
        canonical, transform = flatten_mesh(self.scene_input(), self.root / "stage/input.glb")
        self.assertEqual(transform["input_instances"], 2)
        self.assertEqual(len(canonical.faces), 24)
        self.np.testing.assert_allclose(canonical.bounds, [[-.5, -.5, -1.5], [4.5, 2.5, .5]])

    def create_partition(self):
        canonical, transform = flatten_mesh(self.scene_input(), self.root / "stage/input.glb")
        native = self.root / "native"
        native.mkdir()
        predicted = canonical.copy()
        predicted.vertices = canonical.vertices * transform["scale"] + transform["shift"]
        predicted.export(native / "pred_mesh.ply")
        labels = self.np.repeat([0, 1], 12)
        self.np.save(native / "face_labels.npy", labels)
        (native / "metadata.json").write_text(json.dumps(dict(response="Seat<|SEG|>\nLeg<|SEG|>")))
        return canonical, transform, native

    def test_partition_is_real_submeshes_not_only_colors(self):
        canonical, transform, native = self.create_partition()
        output = self.root / "segmented.glb"
        report = export_partition(canonical, native, output, transform)
        scene = self.trimesh.load(output, process=False)
        self.assertEqual(len(scene.geometry), 2)
        self.assertEqual(sum(len(m.faces) for m in scene.geometry.values()), 24)
        self.np.testing.assert_allclose(scene.bounds, canonical.bounds)
        self.assertEqual([x["faces"] for x in report["parts"]], [12, 12])

    def test_topology_mismatch_fails_closed(self):
        canonical, transform, native = self.create_partition()
        self.np.save(native / "face_labels.npy", self.np.array([0, 1]))
        with self.assertRaisesRegex(RuntimeError, "topology changed"):
            export_partition(canonical, native, self.root / "out.glb", transform)

    def test_invalid_input(self):
        source = self.root / "bad.glb"
        source.write_text("not a mesh")
        with self.assertRaises(Exception):
            flatten_mesh(source, self.root / "output.glb")


class ProtocolTests(unittest.TestCase):
    def invoke(self, payload):
        completed = subprocess.run([sys.executable, str(ROOT / "processor.py")], input=payload + "\n",
                                   text=True, capture_output=True)
        messages = [json.loads(line) for line in completed.stdout.splitlines()]
        terminals = [x for x in messages if x["type"] in {"done", "error"}]
        self.assertEqual(len(terminals), 1)
        return completed, terminals[0]

    def test_invalid_json_is_one_error(self):
        completed, terminal = self.invoke("{")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(terminal["type"], "error")

    def test_unknown_node_is_one_error(self):
        _, terminal = self.invoke(json.dumps(dict(nodeId="unknown")))
        self.assertIn("Unsupported nodeId", terminal["message"])

    def test_each_declared_node_validates_input(self):
        for node in config.SCHEMAS:
            with self.subTest(node=node):
                _, terminal = self.invoke(json.dumps(dict(nodeId=node, input=dict(filePath="/missing.glb"))))
                self.assertIn("existing absolute mesh", terminal["message"])

    def test_progress_uses_host_label_field(self):
        from io import StringIO
        with patch("sys.stdout", new_callable=StringIO) as output:
            processor.emit("progress", percent=3, message="Preparing")
            self.assertEqual(json.loads(output.getvalue()), dict(type="progress", percent=3, label="Preparing"))


class NativePolicyTests(unittest.TestCase):
    def test_native_reuse_probes_and_preserves_failed_state_for_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signature = {"native_policy_revision": 4}
            state = root / ".native-build.json"
            state.write_text(json.dumps(signature))
            failure = subprocess.CalledProcessError(1, ["native_probe.py"])
            with patch.object(installer, "run", side_effect=failure) as run:
                self.assertFalse(installer.reusable_native_build(Path("/python"), root, signature, {}))
            run.assert_called_once()
            self.assertFalse(state.exists())
            self.assertEqual(len(list(root.glob(".native-build.failed-*.json"))), 1)
            state.write_text(json.dumps(signature))
            with patch.object(installer, "run") as run:
                self.assertTrue(installer.reusable_native_build(Path("/python"), root, signature, {}))
            self.assertTrue(state.exists())
            run.assert_called_once()

    def test_source_build_environment_prevents_repair_shadowing_without_leak(self):
        with tempfile.TemporaryDirectory(prefix="native source ") as directory:
            root = Path(directory)
            source = root / "cumm-source"
            installed = root / "installed"
            for path, value in [(source, "source"), (installed, "old-installed")]:
                path.mkdir()
                (path / "binding_fixture.py").write_text(f"ORIGIN = {value!r}\n")
            environment = dict(os.environ, PYTHONPATH=str(installed))
            cumm_environment = native_runtime.source_build_environment(source, environment)
            completed = subprocess.run([sys.executable, "-c", "import binding_fixture;print(binding_fixture.ORIGIN)"],
                                       cwd=root, env=cumm_environment, capture_output=True, text=True, check=True)
            self.assertEqual(completed.stdout.strip(), "source")
            spconv_environment = native_runtime.source_build_environment(root / "spconv-source", environment)
            self.assertNotIn(str(source), spconv_environment["PYTHONPATH"])
            self.assertEqual(environment["PYTHONPATH"], str(installed))

    def test_probe_current_environment_precedes_previous_repair_state(self):
        probe_spec = importlib.util.spec_from_file_location("native_probe", ROOT / "scripts/native_probe.py")
        probe = importlib.util.module_from_spec(probe_spec)
        probe_spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = dict(cuda_home="/old/cuda", architecture_target="8.0+PTX", gpu_sm=80)
            (root / ".native-build.json").write_text(json.dumps(old))
            current = dict(cuda_home="/current/cuda", architecture_target="12.0+PTX", gpu_sm=121)
            environment = native_runtime.native_environment(current, {"PATH": "/usr/bin"})
            with patch.dict(os.environ, environment, clear=True):
                probe.configure_environment(root)
                self.assertEqual(os.environ["CUDA_HOME"], "/current/cuda")
                self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "12.0+PTX")
                self.assertEqual(os.environ["PARTLLM_UTONIA_ALLOW_DOWNLOAD"], "0")
            with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
                probe.configure_environment(root)
                self.assertEqual(os.environ["CUDA_HOME"], "/old/cuda")

    def test_native_environment_shared_build_probe_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            toolkit = Path(directory)
            (toolkit / "lib64").mkdir()
            native = dict(cuda_home=str(toolkit), architecture_target="12.0+PTX", gpu_sm=121)
            environment = native_runtime.native_environment(native, {"PATH": "/old/bin", "LD_LIBRARY_PATH": "/old/lib"})
            self.assertEqual(environment["CUDA_HOME"], str(toolkit))
            self.assertEqual(environment["PATH"], str(toolkit / "bin") + os.pathsep + "/old/bin")
            self.assertEqual(environment["LD_LIBRARY_PATH"], str(toolkit / "lib64") + os.pathsep + "/old/lib")
            self.assertEqual(environment["TORCH_CUDA_ARCH_LIST"], "12.0+PTX")
            self.assertEqual(environment["CUMM_CUDA_ARCH_LIST"], "12.0+PTX")
            self.assertEqual(environment["FLASH_ATTN_CUDA_ARCHS"], "120")
            self.assertEqual(environment["CUMM_DISABLE_JIT"], "1")
            self.assertEqual(environment["SPCONV_DISABLE_JIT"], "1")

    def test_cumm_arch_patch_old_new_marker_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "cumm").mkdir()
            (source / ".modly-source-revision").write_text(native_runtime.CUMM_REVISION)
            common = source / "cumm/common.py"
            original = ("supported_arches = [\n" + native_runtime.CUMM_ARCH_ANCHOR + "]\n"
                        "def get_paths():\n    if True:\n        if True:\n" + native_runtime.CUMM_TOOLKIT_ANCHOR +
                        "            return linux_cuda_root / 'include', linux_cuda_root / 'lib64'\n")
            common.write_text(original)
            digest = hashlib.sha256(original.encode()).hexdigest()
            with patch.object(native_runtime, "CUMM_ORIGINAL_SHA256", digest):
                first = native_runtime.patch_cumm_blackwell(source)
                self.assertIn("'12.0'", common.read_text())
                namespace = {"Path": Path, "os": os}
                exec(common.read_text(), namespace)
                with patch.dict(os.environ, {"CUDA_HOME": str(source / "selected-toolkit")}):
                    self.assertEqual(namespace["get_paths"](), (source / "selected-toolkit/include", source / "selected-toolkit/lib64"))
                self.assertEqual(native_runtime.patch_cumm_blackwell(source), first)
                (source / ".modly-cumm-native-patch.json").unlink()
                # The exact old whitelist-only state is safely upgraded.
                old = original.replace(native_runtime.CUMM_ARCH_ANCHOR, native_runtime.CUMM_ARCH_PATCH)
                common.write_text(old)
                old_marker = source / ".modly-cumm-sm120-patch.json"
                old_marker.write_text(json.dumps(dict(patch="cumm-0.7.11-sm120-whitelist-v1",
                    source_revision=native_runtime.CUMM_REVISION, original_sha256=digest,
                    patched_sha256=hashlib.sha256(old.encode()).hexdigest())))
                self.assertEqual(native_runtime.patch_cumm_blackwell(source), first)
                marker = source / ".modly-cumm-native-patch.json"
                marker.write_text('{}')
                with self.assertRaisesRegex(RuntimeError, "provenance differs"):
                    native_runtime.patch_cumm_blackwell(source)
                marker.unlink()
                common.write_text(common.read_text() + "# unrelated change\n")
                with self.assertRaisesRegex(RuntimeError, "unrelated source modifications"):
                    native_runtime.patch_cumm_blackwell(source)

    def test_cumm_headers_override_unrelated_include_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collision = root / "site-packages/include"
            collision.mkdir(parents=True)
            source = root / "pinned-cumm"
            with self.assertRaisesRegex(RuntimeError, "missing tensorview"):
                native_runtime.verify_cumm_headers(source)
            for sentinel in ["tensorview/gemm/core/params.h", "tensorview/core/all.h"]:
                path = source / "include" / sentinel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("// real header sentinel")
            includes = native_runtime.verify_cumm_headers(source)
            native = dict(cuda_home=str(root), architecture_target="12.0+PTX", gpu_sm=121, cumm_include_path=str(includes))
            environment = native_runtime.native_environment(native, {"CUMM_INCLUDE_PATH": str(collision)})
            self.assertEqual(environment["CUMM_INCLUDE_PATH"], str(includes))

    def test_private_ninja_limits_workers_preserves_arguments_and_exit(self):
        with tempfile.TemporaryDirectory(prefix="native policy ") as directory:
            root = Path(directory)
            executable = root / "venv/bin/ninja"
            executable.parent.mkdir(parents=True)
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGUMENT_LOG"\nexit 7\n')
            executable.chmod(0o755)
            native = dict(cuda_home=str(root), architecture_target="12.0+PTX", gpu_sm=121, max_jobs=2)
            native_runtime.prepare_native_paths(root, native)
            environment = native_runtime.native_environment(native, {"PATH": "/usr/bin", "ARGUMENT_LOG": str(root / "args")})
            completed = subprocess.run([str(Path(native["ninja_directory"]) / "ninja"), "-v", "task with spaces"], env=environment)
            self.assertEqual(completed.returncode, 7)
            self.assertEqual((root / "args").read_text().splitlines(), ["-j", "2", "-v", "task with spaces"])

    def test_old_toolkit_objects_preserved_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "build").mkdir()
            (source / "build/old.o").write_text("wrong toolkit")
            signature = {"native_policy_revision": native_runtime.NATIVE_POLICY_REVISION}
            native_runtime.preserve_old_build(source, signature)
            self.assertFalse((source / "build").exists())
            self.assertEqual(len(list(source.glob("build.before-policy-*/old.o"))), 1)
            (source / "build").mkdir()
            (source / "build/new.o").write_text("correct toolkit")
            native_runtime.preserve_old_build(source, signature)
            self.assertTrue((source / "build/new.o").is_file())
            native_runtime.preserve_old_build(source, dict(signature, pyg_source_build_policy="no-global-wheel-cache-v1"))
            self.assertTrue((source / "build/new.o").is_file())
    def test_cumm_patch_refuses_wrong_revision_or_unrecognized_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "revision"):
                native_runtime.patch_cumm_blackwell(source)
            (source / ".modly-source-revision").write_text(native_runtime.CUMM_REVISION)
            (source / "cumm").mkdir()
            (source / "cumm/common.py").write_text("unexpected source")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                native_runtime.patch_cumm_blackwell(source)


if __name__ == "__main__":
    unittest.main()
