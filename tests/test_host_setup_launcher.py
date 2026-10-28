"""Opt-in checks against exact checkout launchers; no packages are downloaded."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOSTS = [os.environ.get("MODLY_TEST_UPSTREAM_HOST"), os.environ.get("MODLY_TEST_LOCAL_HOST")]


def launcher_source(host):
    dependencies = os.environ.get("MODLY_TEST_DEPENDENCIES_HOST")
    if not host or not dependencies:
        pytest.skip("Set MODLY_TEST_UPSTREAM_HOST/LOCAL_HOST and MODLY_TEST_DEPENDENCIES_HOST")
    script = """
const {createRequire}=require('node:module');
const req=createRequire(process.argv[2]+'/package.json');
const {buildSync}=req('esbuild');
const result=buildSync({entryPoints:[process.argv[1]+'/electron/main/setup-launcher.ts'],bundle:true,platform:'node',format:'cjs',write:false});
const mod={exports:{}};new Function('exports','module',result.outputFiles[0].text)(mod.exports,mod);
process.stdout.write(mod.exports.SETUP_LAUNCHER_SOURCE);
"""
    return subprocess.check_output(["node", "-e", script, host, dependencies], text=True)


@pytest.mark.parametrize("host", HOSTS)
def test_real_launcher_setup_bootstraps_sibling_imports_from_unrelated_cwd(host, tmp_path):
    source = launcher_source(host)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", source, str(ROOT / "setup.py"), "{}"],
                            cwd=tmp_path, env=environment, text=True, capture_output=True)
    assert result.returncode == 1
    assert "Setup requires python_exe and ext_dir" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize("host", HOSTS)
def test_real_launcher_preserves_pyg_no_cache_environment(host, tmp_path):
    source = launcher_source(host)
    capture = tmp_path / "captured.json"
    pip_stub = tmp_path / "pip.py"
    pip_stub.write_text("import json,os,sys;json.dump({'args':sys.argv[2:],'no_cache':os.environ.get('PIP_NO_CACHE_DIR')},open(sys.argv[1],'w'))")
    setup = tmp_path / "setup.py"
    setup.write_text(f"""import sys,runpy,subprocess
sys.path.insert(0,{str(ROOT)!r})
installer=runpy.run_path({str(ROOT / 'setup.py')!r},run_name='partllm_test_installer')
command=[sys.executable,{str(pip_stub)!r},{str(capture)!r}]+installer['pyg_source_command'](['install'],'torch-scatter==2.1.2')
subprocess.run(command,env=installer['pyg_source_environment'](dict(__import__('os').environ)),check=True)
""")
    result = subprocess.run([sys.executable, "-c", source, str(setup), "{}"], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    captured = json.loads(capture.read_text())
    assert "--no-cache-dir" not in captured["args"]  # Host really stripped it.
    assert captured["no_cache"] == "1"
