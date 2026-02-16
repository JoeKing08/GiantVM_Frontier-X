#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_ROOT="${ROOT_DIR}/artifacts"
mkdir -p "${ART_ROOT}"
TMPD="${ART_ROOT}/modea-smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${TMPD}"

cleanup() {
  set +e
  if [[ -n "${MASTER_PID:-}" ]]; then kill "${MASTER_PID}" 2>/dev/null || true; fi
  if [[ -n "${SLAVE_PID:-}" ]]; then kill "${SLAVE_PID}" 2>/dev/null || true; fi
  wait "${MASTER_PID:-}" "${SLAVE_PID:-}" 2>/dev/null || true
  sudo rmmod wavevm 2>/dev/null || true
  echo "[INFO] logs are in: ${TMPD}"
}
trap cleanup EXIT

if [[ ! -e /dev/kvm ]]; then
  echo "[ERROR] /dev/kvm is missing; Mode A test requires virtualization."
  exit 1
fi

KREL="$(uname -r)"
if [[ ! -d "/lib/modules/${KREL}/build" ]]; then
  echo "[ERROR] kernel headers missing: /lib/modules/${KREL}/build"
  exit 1
fi

echo "[INFO] building wavevm.ko ..."
make -C "/lib/modules/${KREL}/build" M="${ROOT_DIR}/master_core" modules > "${TMPD}/kbuild.log" 2>&1

echo "[INFO] loading wavevm.ko ..."
sudo insmod "${ROOT_DIR}/master_core/wavevm.ko"
if [[ ! -e /dev/wavevm ]]; then
  echo "[ERROR] /dev/wavevm not found after insmod"
  exit 1
fi

SWARM_CFG="${TMPD}/swarm_modea.conf"
cat > "${SWARM_CFG}" <<'EOF'
NODE 0 127.0.0.1 29100 1 1
EOF

echo "[INFO] starting Mode A slave..."
(
  cd "${ROOT_DIR}" && \
  WVM_SHM_FILE=/dev/shm/wvm_modea_slave ./slave_daemon/wavevm_node_slave 29105 1 1024 0 29101
) > "${TMPD}/slave.log" 2>&1 &
SLAVE_PID=$!

echo "[INFO] starting Mode A master..."
(
  cd "${ROOT_DIR}" && \
  WVM_INSTANCE_ID=modea WVM_SHM_FILE=/dev/shm/wvm_modea_master \
    ./master_core/wavevm_node_master 1024 29100 "${SWARM_CFG}" 0 29101 29105 1
) > "${TMPD}/master.log" 2>&1 &
MASTER_PID=$!

sleep 8
ps -p "${MASTER_PID},${SLAVE_PID}" -o pid,comm,state --no-headers > "${TMPD}/ps.log" || true

for p in "${MASTER_PID}" "${SLAVE_PID}"; do
  if ! kill -0 "${p}" 2>/dev/null; then
    echo "[ERROR] process ${p} exited unexpectedly"
    tail -n 120 "${TMPD}/master.log" || true
    tail -n 120 "${TMPD}/slave.log" || true
    sudo dmesg | tail -n 120 > "${TMPD}/dmesg_tail.log" || true
    exit 1
  fi
done

if grep -qE "Segmentation fault|Failed to init|Resource Mismatch|CRASH on OOB access|bind .* failed" "${TMPD}"/*.log; then
  echo "[ERROR] failure signature detected in Mode A logs"
  tail -n 120 "${TMPD}/master.log" || true
  tail -n 120 "${TMPD}/slave.log" || true
  sudo dmesg | tail -n 120 > "${TMPD}/dmesg_tail.log" || true
  exit 1
fi

sudo dmesg | tail -n 120 > "${TMPD}/dmesg_tail.log" || true
echo "[INFO] Mode A smoke passed."
