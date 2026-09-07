#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$project_root/data/lre_dataset"

if [[ -d "$destination" ]]; then
  echo "LRE data already exists: $destination"
else
  temporary_root="$(mktemp -d)"
  trap 'rm -rf "$temporary_root"' EXIT

  git clone --depth 1 https://github.com/frankniujc/entrainment.git \
    "$temporary_root/entrainment"
  mkdir -p "$project_root/data"
  cp -R "$temporary_root/entrainment/data/lre_dataset" "$destination"
  echo "Downloaded LRE data to $destination"
fi

python_prefix="$(python -c 'import sys; print(sys.prefix)')"
nltk_destination="${NLTK_DATA:-$python_prefix/nltk_data}"
mkdir -p "$nltk_destination"
python -m nltk.downloader -d "$nltk_destination" brown
echo "Brown corpus is available at $nltk_destination"
