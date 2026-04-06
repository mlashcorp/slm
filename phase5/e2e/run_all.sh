#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ubuntu/code/slm"
LAUNCH_OUT="/tmp/phase5_e2e_launch.out"

cd "${ROOT_DIR}"

echo "[phase5:e2e] Step 1/4 download sample data"
bash phase5/e2e/01_download_nanochat_sample.sh

echo "[phase5:e2e] Step 2/4 upload data to R2"
bash phase5/e2e/02_upload_to_r2.sh

echo "[phase5:e2e] Step 3/4 launch Vast smoke job"
bash phase5/e2e/03_launch_vast_smoke.sh | tee "${LAUNCH_OUT}"

echo "[phase5:e2e] Step 4/4 ssh verify"
python3 - <<'PY'
import re
from pathlib import Path

text = Path('/tmp/phase5_e2e_launch.out').read_text()
m = re.search(r"SSH\s*:\s*ssh\s+-p\s+(\d+)\s+root@([^\s]+)", text)
if not m:
    raise SystemExit("Could not parse SSH host/port from launch output")
port, host = m.group(1), m.group(2)
print(f"HOST={host}")
print(f"PORT={port}")
PY

HOST="$(python3 - <<'PY'
import re
from pathlib import Path
text = Path('/tmp/phase5_e2e_launch.out').read_text()
m = re.search(r"SSH\s*:\s*ssh\s+-p\s+(\d+)\s+root@([^\s]+)", text)
print(m.group(2) if m else "")
PY
)"
PORT="$(python3 - <<'PY'
import re
from pathlib import Path
text = Path('/tmp/phase5_e2e_launch.out').read_text()
m = re.search(r"SSH\s*:\s*ssh\s+-p\s+(\d+)\s+root@([^\s]+)", text)
print(m.group(1) if m else "")
PY
)"

if [[ -z "${HOST}" || -z "${PORT}" ]]; then
  echo "Could not determine host/port from launch output" >&2
  exit 1
fi

bash phase5/e2e/04_ssh_verify.sh "${HOST}" "${PORT}"

echo
echo "[phase5:e2e] Done. Remember to terminate the instance when finished:"
echo "cd /home/ubuntu/workspace/code/Opencode-Remote-/ml && set -a && source .env && set +a && python3 launch.py status"
echo "cd /home/ubuntu/workspace/code/Opencode-Remote-/ml && set -a && source .env && set +a && python3 launch.py terminate <instance_id>"
