#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_ROOT="${ROOT_DIR}/artifacts"
TS="$(date +%Y%m%d-%H%M%S)"
ART_DIR="${ART_ROOT}/cloud-flat-2node-${TS}"
STAGE_DIR="${ART_DIR}/stage"

NODE0=""
NODE1=""
SSH_KEY=""
RAM_MB=1024
CORES=1
SYNC_BATCH=1
KEEP_RUNNING=0
WAIT_SECONDS=12
RUN_VM_TEST=0
QEMU_BIN="qemu-system-x86_64"
VM_IMAGE_URL="https://cloud-images.ubuntu.com/minimal/releases/jammy/release/ubuntu-22.04-minimal-cloudimg-amd64.img"
VM_TIMEOUT=240
VM_MEMORY_MB=1024
VM_SMP=2

# Flat topology defaults
GW_PORT=8000
MASTER_PORT=9000
CTRL_PORT=9001
SLAVE_PORT=9005

FAIL_PAT='Address already in use|Segmentation fault|Failed to init|Resource Mismatch|CRASH on OOB access|RX socket create failed|bind .* failed|Operation not permitted'

usage() {
  cat <<'EOF'
Usage:
  bash deploy/cloud_flat_2node_smoke.sh \
    --node0 <user@ip-or-hostname> \
    --node1 <user@ip-or-hostname> \
    [--ssh-key <path>] \
    [--ram-mb 1024] \
    [--cores 1] \
    [--sync-batch 1] \
    [--wait-seconds 12] \
    [--vm-test] \
    [--qemu-bin qemu-system-x86_64] \
    [--vm-image-url <url>] \
    [--vm-timeout 240] \
    [--vm-memory-mb 1024] \
    [--vm-smp 2] \
    [--keep-running]

Example:
  bash deploy/cloud_flat_2node_smoke.sh \
    --node0 ubuntu@10.0.0.11 \
    --node1 ubuntu@10.0.0.12 \
    --ssh-key ~/.ssh/id_rsa \
    --vm-test
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node0) NODE0="${2:-}"; shift 2 ;;
    --node1) NODE1="${2:-}"; shift 2 ;;
    --ssh-key) SSH_KEY="${2:-}"; shift 2 ;;
    --ram-mb) RAM_MB="${2:-}"; shift 2 ;;
    --cores) CORES="${2:-}"; shift 2 ;;
    --sync-batch) SYNC_BATCH="${2:-}"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="${2:-}"; shift 2 ;;
    --vm-test) RUN_VM_TEST=1; shift 1 ;;
    --qemu-bin) QEMU_BIN="${2:-}"; shift 2 ;;
    --vm-image-url) VM_IMAGE_URL="${2:-}"; shift 2 ;;
    --vm-timeout) VM_TIMEOUT="${2:-}"; shift 2 ;;
    --vm-memory-mb) VM_MEMORY_MB="${2:-}"; shift 2 ;;
    --vm-smp) VM_SMP="${2:-}"; shift 2 ;;
    --keep-running) KEEP_RUNNING=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${NODE0}" || -z "${NODE1}" ]]; then
  echo "[ERROR] --node0 and --node1 are required."
  usage
  exit 1
fi

if ! [[ "${RAM_MB}" =~ ^[0-9]+$ ]] || [[ "${RAM_MB}" -lt 512 ]]; then
  echo "[ERROR] --ram-mb must be integer >= 512"
  exit 1
fi
if ! [[ "${CORES}" =~ ^[0-9]+$ ]] || [[ "${CORES}" -lt 1 ]]; then
  echo "[ERROR] --cores must be integer >= 1"
  exit 1
fi
if ! [[ "${SYNC_BATCH}" =~ ^[0-9]+$ ]] || [[ "${SYNC_BATCH}" -lt 1 ]]; then
  echo "[ERROR] --sync-batch must be integer >= 1"
  exit 1
fi
if ! [[ "${WAIT_SECONDS}" =~ ^[0-9]+$ ]] || [[ "${WAIT_SECONDS}" -lt 3 ]]; then
  echo "[ERROR] --wait-seconds must be integer >= 3"
  exit 1
fi
if ! [[ "${VM_TIMEOUT}" =~ ^[0-9]+$ ]] || [[ "${VM_TIMEOUT}" -lt 30 ]]; then
  echo "[ERROR] --vm-timeout must be integer >= 30"
  exit 1
fi
if ! [[ "${VM_MEMORY_MB}" =~ ^[0-9]+$ ]] || [[ "${VM_MEMORY_MB}" -lt 512 ]]; then
  echo "[ERROR] --vm-memory-mb must be integer >= 512"
  exit 1
fi
if ! [[ "${VM_SMP}" =~ ^[0-9]+$ ]] || [[ "${VM_SMP}" -lt 1 ]]; then
  echo "[ERROR] --vm-smp must be integer >= 1"
  exit 1
fi

mkdir -p "${ART_DIR}" "${STAGE_DIR}/bin" "${STAGE_DIR}/conf"
REMOTE_ROOT="/tmp/wavevm-flat-2node-${TS}"

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o BatchMode=yes
  -o ConnectTimeout=10
)
if [[ -n "${SSH_KEY}" ]]; then
  SSH_OPTS+=(-i "${SSH_KEY}")
fi

remote_host_from_target() {
  local t="$1"
  if [[ "${t}" == *"@"* ]]; then
    echo "${t#*@}"
  else
    echo "${t}"
  fi
}

NODE0_HOST="$(remote_host_from_target "${NODE0}")"
NODE1_HOST="$(remote_host_from_target "${NODE1}")"
RAM_GB=$(( (RAM_MB + 1023) / 1024 ))

echo "[INFO] Building binaries locally..."
make -C "${ROOT_DIR}/gateway_service" >/dev/null
make -C "${ROOT_DIR}/slave_daemon" >/dev/null
make -C "${ROOT_DIR}/master_core" -f Makefile_User >/dev/null

cp "${ROOT_DIR}/gateway_service/wavevm_gateway" "${STAGE_DIR}/bin/"
cp "${ROOT_DIR}/master_core/wavevm_node_master" "${STAGE_DIR}/bin/"
cp "${ROOT_DIR}/slave_daemon/wavevm_node_slave" "${STAGE_DIR}/bin/"
cp "${ROOT_DIR}/deploy/sysctl_check.sh" "${STAGE_DIR}/bin/"

cat > "${STAGE_DIR}/conf/logical_topology.txt" <<EOF
NODE 0 127.0.0.1 ${GW_PORT} ${CORES} ${RAM_GB}
NODE 1 127.0.0.1 ${GW_PORT} ${CORES} ${RAM_GB}
EOF

cat > "${STAGE_DIR}/conf/real_routes.txt" <<EOF
ROUTE 0 1 ${NODE0_HOST} ${MASTER_PORT}
ROUTE 1 1 ${NODE1_HOST} ${MASTER_PORT}
EOF

cat > "${STAGE_DIR}/bin/start_node.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
NODE_ID="$2"
RAM_MB="$3"
CORES="$4"
SYNC_BATCH="$5"
GW_PORT="$6"
MASTER_PORT="$7"
CTRL_PORT="$8"
SLAVE_PORT="$9"

mkdir -p "${ROOT}/logs" "${ROOT}/run"

cd "${ROOT}"

nohup "${ROOT}/bin/wavevm_gateway" \
  "${GW_PORT}" 127.0.0.1 "${MASTER_PORT}" "${ROOT}/conf/real_routes.txt" "${CTRL_PORT}" \
  > "${ROOT}/logs/gateway.log" 2>&1 &
echo $! > "${ROOT}/run/gateway.pid"

nohup env WVM_SHM_FILE="/wavevm_flat_node${NODE_ID}" \
  "${ROOT}/bin/wavevm_node_master" \
  "${RAM_MB}" "${MASTER_PORT}" "${ROOT}/conf/logical_topology.txt" "${NODE_ID}" "${CTRL_PORT}" "${SLAVE_PORT}" "${SYNC_BATCH}" \
  > "${ROOT}/logs/master.log" 2>&1 &
echo $! > "${ROOT}/run/master.pid"

sleep 2

nohup env WVM_SHM_FILE="/wavevm_flat_node${NODE_ID}" \
  "${ROOT}/bin/wavevm_node_slave" \
  "${SLAVE_PORT}" "${CORES}" "${RAM_MB}" "${NODE_ID}" "${CTRL_PORT}" \
  > "${ROOT}/logs/slave.log" 2>&1 &
echo $! > "${ROOT}/run/slave.pid"
EOF

cat > "${STAGE_DIR}/bin/stop_node.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
for n in slave master gateway; do
  if [[ -f "${ROOT}/run/${n}.pid" ]]; then
    pid="$(cat "${ROOT}/run/${n}.pid" || true)"
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  fi
done
EOF

chmod +x "${STAGE_DIR}/bin/"*.sh

ssh_run() {
  local target="$1"
  shift
  ssh "${SSH_OPTS[@]}" "${target}" "$@"
}

ship_stage() {
  local target="$1"
  echo "[INFO] Shipping stage to ${target} ..."
  tar -C "${STAGE_DIR}" -cf - . | ssh_run "${target}" "rm -rf '${REMOTE_ROOT}' && mkdir -p '${REMOTE_ROOT}' && tar -C '${REMOTE_ROOT}' -xf -"
}

start_remote_stack() {
  local target="$1"
  local node_id="$2"
  echo "[INFO] Starting stack on ${target} (node_id=${node_id}) ..."
  ssh_run "${target}" "bash '${REMOTE_ROOT}/bin/start_node.sh' '${REMOTE_ROOT}' '${node_id}' '${RAM_MB}' '${CORES}' '${SYNC_BATCH}' '${GW_PORT}' '${MASTER_PORT}' '${CTRL_PORT}' '${SLAVE_PORT}'"
}

stop_remote_stack() {
  local target="$1"
  ssh_run "${target}" "bash '${REMOTE_ROOT}/bin/stop_node.sh' '${REMOTE_ROOT}'" || true
}

collect_logs() {
  local target="$1"
  local tag="$2"
  for n in gateway master slave; do
    ssh_run "${target}" "tail -n 300 '${REMOTE_ROOT}/logs/${n}.log' 2>/dev/null || true" > "${ART_DIR}/${tag}_${n}.log" || true
  done
  ssh_run "${target}" "for n in gateway master slave; do p=\$(cat '${REMOTE_ROOT}/run/'\"\${n}\"'.pid' 2>/dev/null || true); if [[ -n \"\${p}\" ]]; then ps -p \"\${p}\" -o pid,comm,state --no-headers || true; fi; done" > "${ART_DIR}/${tag}_ps.log" || true
  ssh_run "${target}" "ss -lun | grep -E '(:${GW_PORT}|:${MASTER_PORT}|:${CTRL_PORT}|:${SLAVE_PORT})' || true" > "${ART_DIR}/${tag}_ports.log" || true
  if [[ "${RUN_VM_TEST}" -eq 1 && "${tag}" == "node0" ]]; then
    ssh_run "${target}" "tail -n 300 '${REMOTE_ROOT}/logs/vm_qemu.log' 2>/dev/null || true" > "${ART_DIR}/${tag}_vm_qemu.log" || true
    ssh_run "${target}" "tail -n 300 '${REMOTE_ROOT}/logs/vm_console.log' 2>/dev/null || true" > "${ART_DIR}/${tag}_vm_console.log" || true
    ssh_run "${target}" "cat '${REMOTE_ROOT}/run/vm_qemu.exit' 2>/dev/null || true" > "${ART_DIR}/${tag}_vm_qemu.exit" || true
  fi
}

verify_logs() {
  local ok=1
  if grep -Eiq "${FAIL_PAT}" "${ART_DIR}"/node*_*.log; then
    echo "[ERROR] Failure signature found in logs."
    ok=0
  fi

  if ! grep -q "Listening on 0.0.0.0:${SLAVE_PORT}" "${ART_DIR}/node0_slave.log"; then
    echo "[ERROR] node0 slave did not show listening evidence."
    ok=0
  fi
  if ! grep -q "Listening on 0.0.0.0:${SLAVE_PORT}" "${ART_DIR}/node1_slave.log"; then
    echo "[ERROR] node1 slave did not show listening evidence."
    ok=0
  fi

  if ! grep -q "New neighbor discovered: 1" "${ART_DIR}/node0_master.log"; then
    echo "[ERROR] node0 master missing neighbor discovery for node1."
    ok=0
  fi
  if ! grep -q "New neighbor discovered: 0" "${ART_DIR}/node1_master.log"; then
    echo "[ERROR] node1 master missing neighbor discovery for node0."
    ok=0
  fi

  if [[ "${ok}" -ne 1 ]]; then
    return 1
  fi
  return 0
}

run_vm_test() {
  local target="$1"
  echo "[INFO] Running VM test on ${target} ..."
  ssh "${SSH_OPTS[@]}" "${target}" bash -s -- \
    "${REMOTE_ROOT}" "${QEMU_BIN}" "${VM_IMAGE_URL}" "${VM_TIMEOUT}" "${VM_MEMORY_MB}" "${VM_SMP}" <<'EOF'
set -euo pipefail
ROOT="$1"
QEMU_BIN="$2"
VM_IMAGE_URL="$3"
VM_TIMEOUT="$4"
VM_MEMORY_MB="$5"
VM_SMP="$6"

IMG_PATH="${ROOT}/vm/ubuntu-minimal-amd64.img"
LOG_QEMU="${ROOT}/logs/vm_qemu.log"
LOG_CONSOLE="${ROOT}/logs/vm_console.log"
EXIT_FILE="${ROOT}/run/vm_qemu.exit"

mkdir -p "${ROOT}/vm" "${ROOT}/logs" "${ROOT}/run"
: > "${LOG_QEMU}"
: > "${LOG_CONSOLE}"

if [[ "${QEMU_BIN}" == */* ]]; then
  if [[ ! -x "${QEMU_BIN}" ]]; then
    echo "[ERR] qemu binary not executable: ${QEMU_BIN}" > "${LOG_QEMU}"
    echo "127" > "${EXIT_FILE}"
    exit 0
  fi
else
  if ! command -v "${QEMU_BIN}" >/dev/null 2>&1; then
    echo "[ERR] qemu binary not found in PATH: ${QEMU_BIN}" > "${LOG_QEMU}"
    echo "127" > "${EXIT_FILE}"
    exit 0
  fi
fi

if ! "${QEMU_BIN}" -accel help 2>/dev/null | grep -qw wavevm; then
  echo "[ERR] qemu binary has no wavevm accel: ${QEMU_BIN}" > "${LOG_QEMU}"
  echo "126" > "${EXIT_FILE}"
  exit 0
fi

if [[ ! -f "${IMG_PATH}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 10 "${VM_IMAGE_URL}" -o "${IMG_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${IMG_PATH}" "${VM_IMAGE_URL}"
  else
    echo "[ERR] neither curl nor wget found for image download" > "${LOG_QEMU}"
    echo "125" > "${EXIT_FILE}"
    exit 0
  fi
fi

if [[ ! -s "${IMG_PATH}" ]]; then
  echo "[ERR] vm image missing/empty: ${IMG_PATH}" > "${LOG_QEMU}"
  echo "124" > "${EXIT_FILE}"
  exit 0
fi

MASTER_PID="$(cat "${ROOT}/run/master.pid" 2>/dev/null || true)"
if [[ -z "${MASTER_PID}" || ! -r "/proc/${MASTER_PID}/environ" ]]; then
  echo "[ERR] master pid unavailable; cannot read WVM_ENV_SOCK_PATH" > "${LOG_QEMU}"
  echo "123" > "${EXIT_FILE}"
  exit 0
fi

WVM_ENV_SOCK_PATH="$(tr '\0' '\n' < "/proc/${MASTER_PID}/environ" | awk -F= '$1=="WVM_ENV_SOCK_PATH"{print $2; exit}')"
if [[ -z "${WVM_ENV_SOCK_PATH}" ]]; then
  echo "[ERR] WVM_ENV_SOCK_PATH not found in master environ" > "${LOG_QEMU}"
  echo "122" > "${EXIT_FILE}"
  exit 0
fi

set +e
timeout "${VM_TIMEOUT}"s env WVM_ENV_SOCK_PATH="${WVM_ENV_SOCK_PATH}" "${QEMU_BIN}" \
  -name WVM-Flat-VMTest \
  -m "${VM_MEMORY_MB}" \
  -smp "${VM_SMP}" \
  -accel wavevm \
  -drive "file=${IMG_PATH},if=virtio,format=qcow2,cache=none,aio=threads" \
  -netdev user,id=n0 -device virtio-net-pci,netdev=n0 \
  -display none -serial "file:${LOG_CONSOLE}" -monitor none -no-reboot \
  >> "${LOG_QEMU}" 2>&1
rc=$?
set -e
echo "${rc}" > "${EXIT_FILE}"
EOF
}

verify_vm_logs() {
  local ok=1
  if [[ ! -s "${ART_DIR}/node0_vm_qemu.log" ]]; then
    echo "[ERROR] VM qemu log missing on node0."
    ok=0
  fi
  if grep -Eiq "has no wavevm accel|qemu binary not found|qemu binary not executable|WVM_ENV_SOCK_PATH not found|invalid accelerator" "${ART_DIR}/node0_vm_qemu.log"; then
    echo "[ERROR] VM test failed before boot due to qemu/accel/env issue."
    ok=0
  fi
  if ! grep -Eiq "Ubuntu|cloud-init|login:" "${ART_DIR}/node0_vm_console.log"; then
    echo "[ERROR] VM console missing expected boot evidence."
    ok=0
  fi
  if [[ "${ok}" -ne 1 ]]; then
    return 1
  fi
  return 0
}

echo "[INFO] Probing SSH connectivity..."
ssh_run "${NODE0}" "echo '[SSH] node0 ok'" >/dev/null
ssh_run "${NODE1}" "echo '[SSH] node1 ok'" >/dev/null

ship_stage "${NODE0}"
ship_stage "${NODE1}"

start_remote_stack "${NODE0}" 0
start_remote_stack "${NODE1}" 1

echo "[INFO] Waiting ${WAIT_SECONDS}s for convergence..."
sleep "${WAIT_SECONDS}"

if [[ "${RUN_VM_TEST}" -eq 1 ]]; then
  run_vm_test "${NODE0}"
fi

collect_logs "${NODE0}" "node0"
collect_logs "${NODE1}" "node1"

RESULT="PASS"
REASON="flat_2node_smoke_ok"
if ! verify_logs; then
  RESULT="FAIL"
  REASON="startup_or_convergence_evidence_missing"
fi
if [[ "${RESULT}" == "PASS" && "${RUN_VM_TEST}" -eq 1 ]]; then
  if ! verify_vm_logs; then
    RESULT="FAIL"
    REASON="vm_boot_evidence_missing_or_accel_invalid"
  else
    REASON="flat_2node_vm_test_ok"
  fi
fi

if [[ "${KEEP_RUNNING}" -eq 0 ]]; then
  echo "[INFO] Stopping remote stacks (default behavior)."
  stop_remote_stack "${NODE0}"
  stop_remote_stack "${NODE1}"
else
  echo "[INFO] --keep-running set; processes remain active on remote nodes."
fi

{
  echo "STATUS=${RESULT}"
  echo "REASON=${REASON}"
  echo "NODE0=${NODE0}"
  echo "NODE1=${NODE1}"
  echo "REMOTE_ROOT=${REMOTE_ROOT}"
  echo "RUN_VM_TEST=${RUN_VM_TEST}"
} > "${ART_DIR}/result.summary"

echo "[INFO] Summary:"
cat "${ART_DIR}/result.summary"
echo "[INFO] Artifacts: ${ART_DIR}"

if [[ "${RESULT}" != "PASS" ]]; then
  exit 1
fi
