# RDMA Sweep Tool

Run Linux `perftest` sweeps across two hosts and generate an SVG report.

The tool starts the selected `ib_*` binary on the server host, runs the matching
client command against the server RDMA data-plane address, collects host CPU and
memory snapshots from both machines, and writes JSON/CSV/SVG output.

## Requirements

- Python 3.10+
- `pyyaml` on the controller host
- SSH access from the controller host to both test hosts
- `perftest` installed on both test hosts
- `ss`, `/usr/bin/time`, and `/proc` available on both test hosts
- `perf` and sudo access on the server host only when `perftest.perf_record` is enabled

Install the Python dependency:

```bash
python3 -m pip install pyyaml
```

## Configuration

Start from [examples/qp_scale/config.yaml](examples/qp_scale/config.yaml).

```yaml
test: ib_write_bw
duration: 2

server:
  host: rdma-server.example.com
  address: rdma-server-data.example.com

client:
  host: rdma-client.example.com

perftest:
  dir: /opt/perftest
  perf_record: false
  wait_timeout: 30
  default_port: 18515
  fixed:
    port: 18515
    msg_size: 64
    device: roce-device-name
    gid_index: 5
    force_link: Ethernet

ssh:
  sudo: true
  connect_timeout: 10
  options: [-o, BatchMode=yes, -o, StrictHostKeyChecking=accept-new]

fixed:
  server:
    rdma_device: mlx5_0
    netdev: enP2p1s0f1np1
    mtu: 4200
  client:
    rdma_device: mlx5_1
    netdev: enP2p1s0f1np1
    mtu: 4200

sweep:
  - name: qp
    values: [1, 2, 4, 8, 16]
```

`server.host` and `client.host` are SSH targets. `server.address` is the RDMA
data-plane address passed to the client perftest command. The tool rejects
loopback addresses and same-host client/server configurations.

`perftest.dir` must point to the directory containing the selected binary, for
example `/usr/bin` when `ib_write_bw` is installed as `/usr/bin/ib_write_bw`.

`perftest.fixed` contains perftest command-line parameters that are applied to
every run. Put values there when they should become perftest flags, for example
`port`, `msg_size`, `device`, `gid_index`, or `force_link`. A parameter may be
defined in either `perftest.fixed` or `sweep`, not both.

Top-level `fixed` is reserved for RDMA/OS configuration that is held constant
for the run and recorded in `run_config.json`, each `result.json`, and
`summary.json`; the tool does not mutate OS state from this block. Do not put
perftest flags under top-level `fixed`.

Before running a sweep, top-level `fixed.server` and `fixed.client` are checked
over SSH. Supported read-only checks are `rdma_device`, `netdev`, `mtu`,
`address`/`ip`, `operstate`, `rdma_state`, and `sysctl` key/value mappings. A
mismatch aborts before any perftest process is started. Successful preflight
evidence is stored under `fixed_check` in `run_config.json`, each `result.json`,
and `summary.json`. Failed preflight checks write `preflight.json` in the output
directory before exiting.

`fixed.*.mtu` checks the operating-system MTU on `fixed.*.netdev`.
`perftest.fixed.mtu` is different: it maps to perftest `-m`.
`mtu`, `address`/`ip`, and `operstate` checks require `netdev`; `rdma_state`
requires `rdma_device`.

## Run

Preview the perftest combinations without touching either host:

```bash
python3 rdma_sweep.py \
  --config examples/qp_scale/config.yaml \
  --dry-run
```

```bash
python3 rdma_sweep.py \
  --config examples/qp_scale/config.yaml \
  --output-dir results/qp_scale
```

Regenerate the report from an existing result directory:

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

## RoCE Setup Notes

For RoCE, perftest still uses a TCP control connection to exchange QP
information before RDMA traffic starts. Assign the test IP to the Ethernet
netdev backing the RDMA device, then use that IP as `server.address`.

Find the RDMA device to netdev mapping:

```bash
rdma link
```

Example output:

```text
link roceP2p1s0f1/1 state ACTIVE physical_state LINK_UP netdev enP2p1s0f1np1
```

In that case, assign the data-plane IP to `enP2p1s0f1np1` and use
`device: roceP2p1s0f1` under `perftest.fixed` when the device is constant.
Use a `sweep` entry named `device` only when you intentionally want to sweep
across devices. Record the matching `rdma_device`, `netdev`, and expected MTU
under top-level `fixed.server` / `fixed.client` when you want the tool to check
that OS/RDMA state before it starts perftest.

Find the RoCEv2 GID index for the assigned IP:

```bash
show_gids
```

Use that index as `gid_index`.

Basic preflight checks:

```bash
SERVER_SSH=rdma-server.example.com
CLIENT_SSH=rdma-client.example.com
SERVER_ADDR=rdma-server-data.example.com
PERFTEST_DIR=/opt/perftest
TEST=ib_write_bw

ssh "$SERVER_SSH" "test -x $PERFTEST_DIR/$TEST && ss -H -tln >/dev/null"
ssh "$CLIENT_SSH" "test -x $PERFTEST_DIR/$TEST && /usr/bin/time true"
ssh "$CLIENT_SSH" "ip route get $SERVER_ADDR && ping -c 3 $SERVER_ADDR"
```

## Report

The SVG report includes:

- bandwidth
- message rate
- server and client average CPU utilization
- server and client host memory-pressure delta
- server per-core CPU utilization
- top server CPU symbols when `perftest.perf_record` is enabled

Report generation reads an existing result directory. It needs `summary.json`;
`run_config.json` is optional but supplies the report title/subtitle when
present. Report mode does not SSH into the test hosts.

![QP scale report](examples/qp_scale/chart.svg)

The files under [examples/qp_scale](examples/qp_scale) are sanitized output from
a real two-host run. Hostnames, addresses, device name, and port were replaced
with generic values; measured bandwidth, message rate, CPU, and memory values
come from the run.

## Sweep Parameters

Common YAML keys in `perftest.fixed` and `sweep` map to perftest flags:

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

`duration`, `server`, `client`, `fixed`, `perftest`, `ssh`, `report`, and
`use_gpu` are tool configuration keys and are not forwarded as perftest
arguments. Put fixed perftest flags under `perftest.fixed`; put RDMA/OS state
that should be checked and recorded under top-level `fixed`.

## Development Checks

Run the focused fixed-config and report tests:

```bash
python3 -m unittest \
  test_rdma_sweep.TestFixedConfigChecks \
  test_rdma_sweep.TestSweepConfig \
  test_rdma_sweep.TestReportGeneration \
  test_rdma_sweep.TestMainCLI
```

Check syntax, example JSON, and whitespace:

```bash
python3 -m py_compile rdma_config.py rdma_remote.py rdma_sweep.py test_rdma_sweep.py
python3 -m json.tool examples/qp_scale/run_config.json >/dev/null
python3 -m json.tool examples/qp_scale/summary.json >/dev/null
git diff --check
```

Regenerate a report from the sanitized example data:

```bash
python3 rdma_sweep.py --report examples/qp_scale
```
