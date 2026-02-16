#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_ROOT="${ROOT_DIR}/artifacts"
mkdir -p "${ART_ROOT}"
TMPD="${ART_ROOT}/modeb-smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${TMPD}"

cleanup() {
  set +e
  if [[ -n "${M1_PID:-}" ]]; then kill "${M1_PID}" 2>/dev/null || true; fi
  if [[ -n "${M2_PID:-}" ]]; then kill "${M2_PID}" 2>/dev/null || true; fi
  if [[ -n "${S1_PID:-}" ]]; then kill "${S1_PID}" 2>/dev/null || true; fi
  if [[ -n "${S2_PID:-}" ]]; then kill "${S2_PID}" 2>/dev/null || true; fi
  wait "${M1_PID:-}" "${M2_PID:-}" "${S1_PID:-}" "${S2_PID:-}" 2>/dev/null || true
  echo "[INFO] logs are in: ${TMPD}"
}
trap cleanup EXIT

SWARM_CFG="${TMPD}/swarm.conf"
cat > "${SWARM_CFG}" <<'EOF'
NODE 0 127.0.0.1 19100 1 1
NODE 1 127.0.0.1 19200 1 1
EOF

echo "[INFO] starting two slave instances..."
(
  cd "${ROOT_DIR}" && \
  WVM_SHM_FILE=/dev/shm/wvm_slave_ci_1 ./slave_daemon/wavevm_node_slave 19105 1 1024 0 19101
) > "${TMPD}/slave1.log" 2>&1 &
S1_PID=$!
(
  cd "${ROOT_DIR}" && \
  WVM_SHM_FILE=/dev/shm/wvm_slave_ci_2 ./slave_daemon/wavevm_node_slave 19205 1 1024 1 19201
) > "${TMPD}/slave2.log" 2>&1 &
S2_PID=$!

echo "[INFO] starting two master instances..."
(
  cd "${ROOT_DIR}" && \
  WVM_INSTANCE_ID=1 WVM_SHM_FILE=/dev/shm/wvm_master_ci_1 \
    ./master_core/wavevm_node_master 1024 19100 "${SWARM_CFG}" 0 19101 19105 1
) > "${TMPD}/master1.log" 2>&1 &
M1_PID=$!
(
  cd "${ROOT_DIR}" && \
  WVM_INSTANCE_ID=2 WVM_SHM_FILE=/dev/shm/wvm_master_ci_2 \
    ./master_core/wavevm_node_master 1024 19200 "${SWARM_CFG}" 1 19201 19205 1
) > "${TMPD}/master2.log" 2>&1 &
M2_PID=$!

sleep 8

echo "[INFO] process snapshot:"
ps -p "${S1_PID},${S2_PID},${M1_PID},${M2_PID}" -o pid,comm,state --no-headers || true

if grep -q "Operation not permitted" "${TMPD}"/master*.log; then
  echo "::warning::Runner blocks required UDP socket operations; skipping smoke assertions."
  exit 0
fi

for p in "${S1_PID}" "${S2_PID}" "${M1_PID}" "${M2_PID}"; do
  if ! kill -0 "${p}" 2>/dev/null; then
    echo "[ERROR] process ${p} exited unexpectedly"
    tail -n 80 "${TMPD}/master1.log" || true
    tail -n 80 "${TMPD}/master2.log" || true
    tail -n 80 "${TMPD}/slave1.log" || true
    tail -n 80 "${TMPD}/slave2.log" || true
    exit 1
  fi
done

if grep -qE "Address already in use|Segmentation fault|Failed to init|bind .* failed|Resource Mismatch|CRASH on OOB access" "${TMPD}"/*.log; then
  echo "[ERROR] failure signature detected"
  tail -n 80 "${TMPD}/master1.log" || true
  tail -n 80 "${TMPD}/master2.log" || true
  tail -n 80 "${TMPD}/slave1.log" || true
  tail -n 80 "${TMPD}/slave2.log" || true
  exit 1
fi

echo "[INFO] Mode B multi-instance smoke passed."
