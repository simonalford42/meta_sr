#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lasr_root="$project_root/LaSR.jl"
venv_dir="${LASR_VENV_DIR:-$project_root/.venv-lasr}"

if [[ ! -f "$lasr_root/experiments/main.py" ]]; then
    git -C "$project_root" submodule update --init LaSR.jl
fi

if [[ ! -x "$venv_dir/bin/python" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.10 "$venv_dir"
    else
        python3.10 -m venv "$venv_dir"
    fi
fi

# Build from an archive so setuptools cannot rewrite tracked egg-info files in
# the pinned submodule.
build_dir=$(mktemp -d /tmp/lasr-build.XXXXXX)
cleanup_build_dir() {
    if [[ -d "$build_dir" ]]; then
        mkdir -p "$HOME/trash"
        mv "$build_dir" "$HOME/trash/$(basename "$build_dir")"
    fi
}
trap cleanup_build_dir EXIT
git -C "$lasr_root" archive HEAD | tar -x -C "$build_dir"

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$venv_dir/bin/python" "$build_dir"
    uv pip install --python "$venv_dir/bin/python" 'scikit-learn==1.5.2'
else
    "$venv_dir/bin/python" -m pip install "$build_dir"
    "$venv_dir/bin/python" -m pip install 'scikit-learn==1.5.2'
fi

# The archived paper branch contains the original author's absolute path in
# pysr/juliapkg.json. Patch only the private virtualenv copy, leaving the
# pinned LaSR submodule pristine.
juliapkg_json=$(find "$venv_dir" -path '*/site-packages/pysr/juliapkg.json' -print -quit)
if [[ -z "$juliapkg_json" ]]; then
    echo "Could not locate the installed pysr/juliapkg.json" >&2
    exit 1
fi

# The archived wheel also omits this generated setuptools_scm file when built
# outside its original checkout context.
pysr_dir=$(dirname "$juliapkg_json")
if [[ ! -f "$pysr_dir/version.py" ]]; then
    printf '%s\n' '__version__ = "0.19.dev0"' > "$pysr_dir/version.py"
fi

juliapkg_tmp=$(mktemp "${juliapkg_json}.XXXXXX")
jq --arg path "$lasr_root/SymbolicRegression.jl" \
    '.packages.SymbolicRegression.path = $path' \
    "$juliapkg_json" > "$juliapkg_tmp"
mv "$juliapkg_tmp" "$juliapkg_json"

export JULIA_DEPOT_PATH="${LASR_JULIA_DEPOT:-$project_root/.julia_depot}"
export PYTHON_JULIACALL_EXE="${PYTHON_JULIACALL_EXE:-$(command -v julia)}"

"$venv_dir/bin/python" -c \
    'from pysr import PySRRegressor; print("LaSR paper environment imported successfully")'

printf '\nEnvironment ready. Activate it with:\n  source %s/bin/activate\n' "$venv_dir"
