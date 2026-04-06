# Phase5 Minimal E2E (nanochat + Vast.ai + R2)

This is a minimal smoke flow to validate:

1. Download dataset shards using nanochat code
2. Upload those shards to Cloudflare R2
3. Launch a Vast.ai instance/job via `Opencode-Remote/ml/launch.py`
4. Connect and inspect via SSH

The flow is intentionally small and cheap.

## Prerequisites

- Credentials filled in:
  - `/home/ubuntu/workspace/code/Opencode-Remote-/ml/.env`
- Vast account has your SSH public key added.
- `rclone`, `python3`, and `git` available on this machine.

## Files

- `01_download_nanochat_sample.sh`
- `02_upload_to_r2.sh`
- `03_launch_vast_smoke.sh`
- `04_ssh_verify.sh`
- `remote_nanochat_smoke.sh` (runs on the Vast instance)
- `vast_smoke_config.yaml` (launch config used by `ml/launch.py`)

## Default parameters

- Local nanochat data root: `/home/ubuntu/workspace/data/nanochat-e2e`
- R2 prefix: `r2:ml-datasets/phase5/e2e/nanochat`
- Vast price cap: `$0.50/hr`
- GPU override: `RTX 4090`

## Run order

```bash
bash phase5/e2e/01_download_nanochat_sample.sh
bash phase5/e2e/02_upload_to_r2.sh
bash phase5/e2e/03_launch_vast_smoke.sh
```

Or run all four steps in one command:

```bash
bash phase5/e2e/run_all.sh
```

After launch prints SSH details:

```bash
bash phase5/e2e/04_ssh_verify.sh <host> <port>
```

## Notes

- `01_download_nanochat_sample.sh` defaults to `--num-files 1`, which downloads one train shard plus the validation shard.
- `remote_nanochat_smoke.sh` reads dataset from `/workspace/datasets/phase5/e2e/nanochat/base_data_climbmix` and runs a tiny nanochat training smoke command.
- To terminate instances:

```bash
cd /home/ubuntu/workspace/code/Opencode-Remote-/ml
set -a && source .env && set +a
python3 launch.py status
python3 launch.py terminate <instance_id>
```

## Next step after smoke

Run the first canary in `phase5/canary`:

```bash
bash phase5/canary/launch_canary.sh
```
