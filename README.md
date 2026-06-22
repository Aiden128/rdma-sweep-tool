# RDMA Sweep Tool

`rdma_sweep.py` runs Linux `perftest` sweeps across two machines:

- **server host**: starts the selected `ib_*` binary in server mode
- **client host**: runs the matching client command against the server RDMA address

This is not a local SoftRoCE loopback runner.  Sweep mode requires distinct
server/client SSH hosts and an explicit `server.address` for the RDMA data path.
Loopback addresses such as `127.0.0.1` are rejected.

## Output

Each sweep point records:

- perftest JSON metrics such as `BW_average`, `MsgRate`, and `MsgSize`
- server-side `perf record -g` symbols, when enabled
- server and client CPU snapshots from `/proc/stat`
- server and client host-level memory snapshots from `/proc/meminfo`
- client `/usr/bin/time` output

The SVG report includes bandwidth, message rate, server/client average CPU,
server/client host memory-pressure delta, top server CPU symbols, and server
per-core CPU.

![sample report](examples/qp_scale/chart.svg)

The checked-in SVG is sample report output.  Use your generated `summary.json`
and `result.json` files for real bottleneck conclusions.

## Generic Config

Start from `examples/qp_scale/config.yaml`:

```yaml
test: ib_write_bw
duration: 10

server:
  host: rdma-server.example.com
  address: rdma-server-data.example.com

client:
  host: rdma-client.example.com

perftest:
  dir: /opt/perftest
  tmp_dir: /tmp/rdma_sweep_{run_id}
  json_file: "{tmp_dir}/perftest_out.json"
  time_file: "{tmp_dir}/perftest_time.out"
  server_pid_file: "{tmp_dir}/perftest_server.pid"
  server_log_file: "{tmp_dir}/perftest_server.log"
  perf_data: "{tmp_dir}/perftest_perf.data"
  perf_pid_file: "{tmp_dir}/perftest_perf.pid"
  perf_record: true
  wait_timeout: 30
  default_port: 18515
  env: {}

ssh:
  sudo: true
  connect_timeout: 10
  options: [-o, BatchMode=yes, -o, StrictHostKeyChecking=accept-new]

fixed:
  port: 18515
  msg_size: 64K

sweep:
  - name: qp
    values: [2, 4, 8, 16, 32, 64, 128]
```

`server.host` and `client.host` are SSH targets.  `server.address` is the RDMA
address passed to the client perftest command; it may be different from the SSH
hostname on multi-NIC machines.

The default temporary paths use `{run_id}` so old PID/log/JSON files are not
reused across runs.  Keep that pattern for normal use; fixed PID paths are
supported for compatibility, but they are inherently less safe when another
matching perftest binary is already running.

## Run

Install the local dependency on the controller machine:

```bash
python3 -m pip install pyyaml
```

Run a sweep:

```bash
python3 rdma_sweep.py \
  --config examples/qp_scale/config.yaml \
  --output-dir results/qp_scale
```

Generate or refresh the report:

```bash
python3 rdma_sweep.py --report results/qp_scale
```

Output layout:

```text
results/qp_scale/
  run_config.json
  0001/result.json
  0002/result.json
  ...
  summary.json
  summary.csv
  chart.svg
  chart.pdf    # only when cairosvg is installed
```

## Preflight Checks

Replace the environment variables with your lab hosts:

```bash
export RDMA_SWEEP_SERVER_SSH='server-user@server-host'
export RDMA_SWEEP_CLIENT_SSH='client-user@client-host'
export RDMA_SWEEP_SERVER_ADDR='server-rdma-address'
export RDMA_SWEEP_PERFTEST_DIR='/opt/perftest'
export RDMA_SWEEP_TEST='ib_write_bw'
export RDMA_SWEEP_DEVICE='roce-device-name'
export RDMA_SWEEP_GID_INDEX='gid-index'
export RDMA_SWEEP_FORCE_LINK='Ethernet'
```

`server-rdma-address` must be assigned to the Ethernet netdev backing the RDMA
device, not to an unrelated management network.  Confirm the mapping with
`rdma link`; for example, if it shows `roceP2p1s0f1/1 ... netdev
enP2p1s0f1np1`, configure the IP on `enP2p1s0f1np1` and use
`device: roceP2p1s0f1` in the sweep config.  After assigning the IP, run
`show_gids` and use the RoCEv2 GID index for that IPv4 address.

Check SSH and remote prerequisites:

```bash
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)

for h in "$RDMA_SWEEP_SERVER_SSH" "$RDMA_SWEEP_CLIENT_SSH"; do
  ssh "${SSH_OPTS[@]}" "$h" '
    hostname
    id
    python3 --version
    command -v ss
    test -x /usr/bin/time
    command -v ibv_devices >/dev/null 2>&1 && ibv_devices || true
    command -v rdma >/dev/null 2>&1 && rdma link || true
    command -v show_gids >/dev/null 2>&1 && show_gids || true
  '
done

ssh "${SSH_OPTS[@]}" "$RDMA_SWEEP_SERVER_SSH" \
  "test -x \"$RDMA_SWEEP_PERFTEST_DIR/$RDMA_SWEEP_TEST\" && command -v perf && sudo -n true"

ssh "${SSH_OPTS[@]}" "$RDMA_SWEEP_CLIENT_SSH" \
  "test -x \"$RDMA_SWEEP_PERFTEST_DIR/$RDMA_SWEEP_TEST\" && sudo -n true"

ssh "${SSH_OPTS[@]}" "$RDMA_SWEEP_CLIENT_SSH" \
  "ip route get \"$RDMA_SWEEP_SERVER_ADDR\""

ssh "${SSH_OPTS[@]}" "$RDMA_SWEEP_CLIENT_SSH" \
  "ping -c 3 \"$RDMA_SWEEP_SERVER_ADDR\""
```

If `sudo -n true` fails, either configure passwordless sudo for the required
commands or set `ssh.sudo: false` and disable `perftest.perf_record`.

## Minimal Smoke Config

Generate a local smoke-test YAML without committing lab-specific hosts.  These
values are shell variables so the example does not bake lab policy into the
repository:

```bash
export RDMA_SWEEP_DURATION="${RDMA_SWEEP_DURATION:-2}"
export RDMA_SWEEP_PORT="${RDMA_SWEEP_PORT:-18515}"
export RDMA_SWEEP_MSG_SIZE="${RDMA_SWEEP_MSG_SIZE:-64}"
export RDMA_SWEEP_QP="${RDMA_SWEEP_QP:-1}"
export RDMA_SWEEP_SUDO="${RDMA_SWEEP_SUDO:-true}"
export RDMA_SWEEP_SSH_CONNECT_TIMEOUT="${RDMA_SWEEP_SSH_CONNECT_TIMEOUT:-10}"

cat >/tmp/rdma-sweep-smoke.yaml <<YAML
test: ${RDMA_SWEEP_TEST}
duration: ${RDMA_SWEEP_DURATION}

server:
  host: ${RDMA_SWEEP_SERVER_SSH}
  address: ${RDMA_SWEEP_SERVER_ADDR}

client:
  host: ${RDMA_SWEEP_CLIENT_SSH}

perftest:
  dir: ${RDMA_SWEEP_PERFTEST_DIR}

ssh:
  sudo: ${RDMA_SWEEP_SUDO}
  connect_timeout: ${RDMA_SWEEP_SSH_CONNECT_TIMEOUT}
  options: [-o, BatchMode=yes, -o, StrictHostKeyChecking=accept-new]

fixed:
  port: ${RDMA_SWEEP_PORT}
  msg_size: ${RDMA_SWEEP_MSG_SIZE}
  device: ${RDMA_SWEEP_DEVICE}
  gid_index: ${RDMA_SWEEP_GID_INDEX}
  force_link: ${RDMA_SWEEP_FORCE_LINK}

sweep:
  - name: qp
    values: [${RDMA_SWEEP_QP}]
YAML
```

Run it:

```bash
rm -rf /tmp/rdma-sweep-smoke-results
python3 rdma_sweep.py -c /tmp/rdma-sweep-smoke.yaml -o /tmp/rdma-sweep-smoke-results
python3 rdma_sweep.py --report /tmp/rdma-sweep-smoke-results
```

Verify the expected fields:

```bash
python3 - <<'PY'
import json
from pathlib import Path

out = Path('/tmp/rdma-sweep-smoke-results')
summary = json.loads((out / 'summary.json').read_text())
assert len(summary) == 1, summary
assert not summary[0].get('error'), summary[0].get('error')
for key in ('server_cpu_per_core', 'client_cpu_per_core', 'server_memory', 'client_memory'):
    assert summary[0].get(key), key
svg = (out / 'chart.svg').read_text()
for label in ('Average CPU Utilization', 'Host Memory Pressure Delta', 'Server Per-Core CPU Utilization'):
    assert label in svg, label
print('smoke-ok')
PY
```

## Sweep Parameters

Common YAML keys map to perftest flags:

| Config key | Flag | Description |
| --- | --- | --- |
| `msg_size` | `-s` | message size |
| `qp` | `-q` | queue pairs |
| `tx_depth` | `-t` | TX depth |
| `rx_depth` | `-r` | RX depth |
| `post_list` | `-l` | post list |
| `cq_mod` | `-Q` | completion queue moderation |
| `iters` | `-n` | iterations |
| `port` | `-p` | perftest TCP control port |
| `ib_port` | `-i` | IB device port |
| `inline` | `-I` | inline size |
| `sl` | `-S` | service level |
| `mtu` | `-m` | MTU |
| `tos` | `-T` | type of service |
| `recv_post_list` | `--recv-post-list` | receive post list |
| `cpu_util` | `--cpu_util` | ask perftest to report CPU utilization |
| `device` | `-d` | RDMA device |
| `gid_index` | `-x` | GID index |
| `force_link` | `--force-link` | force link type, for example `Ethernet` |
| `rdma_cm` | `-R` | use RDMA CM QPs |
| `comm_rdma_cm` | `-z` | exchange data through rdma_cm |
| `bind_source_ip` | `--bind_source_ip` | bind connection setup source IP |
| `check_alive` | `--check-alive` | perftest alive checks |
| other keys | `--{name}` | passed through with underscores converted to dashes |

`duration`, `server`, `client`, `perftest`, `ssh`, `report`, and `use_gpu` are
tool configuration keys and are not forwarded as perftest arguments.
