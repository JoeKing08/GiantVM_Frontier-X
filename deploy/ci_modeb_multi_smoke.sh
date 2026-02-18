#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_ROOT="${ROOT_DIR}/artifacts"
mkdir -p "${ART_ROOT}"
TMPD="${ART_ROOT}/modeb-smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${TMPD}"
SUMMARY_FILE="${TMPD}/result.summary"

write_summary() {
  local status="$1"
  local reason="$2"
  {
    echo "STATUS=${status}"
    echo "REASON=${reason}"
  } > "${SUMMARY_FILE}"
}

fail() {
  local reason="$1"
  echo "[ERROR] ${reason}"
  write_summary "FAIL" "${reason}"
  tail -n 80 "${TMPD}/master1.log" 2>/dev/null || true
  tail -n 80 "${TMPD}/master2.log" 2>/dev/null || true
  tail -n 80 "${TMPD}/slave1.log" 2>/dev/null || true
  tail -n 80 "${TMPD}/slave2.log" 2>/dev/null || true
  exit 1
}

warn() {
  local reason="$1"
  echo "[WARN] ${reason}"
}

if command -v stdbuf >/dev/null 2>&1; then
  LINEBUF=(stdbuf -oL -eL)
else
  LINEBUF=()
fi

cleanup() {
  local rc=$?
  set +e
  if [[ -n "${M1_PID:-}" ]]; then kill "${M1_PID}" 2>/dev/null || true; fi
  if [[ -n "${M2_PID:-}" ]]; then kill "${M2_PID}" 2>/dev/null || true; fi
  if [[ -n "${S1_PID:-}" ]]; then kill "${S1_PID}" 2>/dev/null || true; fi
  if [[ -n "${S2_PID:-}" ]]; then kill "${S2_PID}" 2>/dev/null || true; fi
  wait "${M1_PID:-}" "${M2_PID:-}" "${S1_PID:-}" "${S2_PID:-}" 2>/dev/null || true
  if [[ ! -f "${SUMMARY_FILE}" ]]; then
    if [[ "${rc}" -eq 0 ]]; then
      write_summary "PASS" "completed_without_explicit_summary"
    else
      write_summary "FAIL" "aborted_unexpectedly_rc_${rc}"
    fi
  fi
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
  exec env WVM_SHM_FILE=/wvm_slave_ci_1 \
    "${LINEBUF[@]}" ./slave_daemon/wavevm_node_slave 19105 1 1024 0 19101
) > "${TMPD}/slave1.log" 2>&1 &
S1_PID=$!
(
  cd "${ROOT_DIR}" && \
  exec env WVM_SHM_FILE=/wvm_slave_ci_2 \
    "${LINEBUF[@]}" ./slave_daemon/wavevm_node_slave 19205 1 1024 1 19201
) > "${TMPD}/slave2.log" 2>&1 &
S2_PID=$!

echo "[INFO] starting two master instances..."
(
  cd "${ROOT_DIR}" && \
  exec env WVM_INSTANCE_ID=1 WVM_SHM_FILE=/wvm_master_ci_1 \
    "${LINEBUF[@]}" ./master_core/wavevm_node_master 1024 19100 "${SWARM_CFG}" 0 19101 19105 1
) > "${TMPD}/master1.log" 2>&1 &
M1_PID=$!
(
  cd "${ROOT_DIR}" && \
  exec env WVM_INSTANCE_ID=2 WVM_SHM_FILE=/wvm_master_ci_2 \
    "${LINEBUF[@]}" ./master_core/wavevm_node_master 1024 19200 "${SWARM_CFG}" 1 19201 19205 1
) > "${TMPD}/master2.log" 2>&1 &
M2_PID=$!

sleep 8

echo "[INFO] process snapshot:"
ps -p "${S1_PID},${S2_PID},${M1_PID},${M2_PID}" -o pid,comm,state --no-headers || true

if grep -q "Operation not permitted" "${TMPD}"/*.log; then
  fail "environment_blocked_udp_socket_operations"
fi

if grep -qE "Address already in use|Segmentation fault|Failed to init|bind .* failed|Resource Mismatch|CRASH on OOB access|RX socket create failed" "${TMPD}"/*.log; then
  fail "failure_signature_detected"
fi

# CI runner may reclaim or respawn workers; use startup evidence instead of long-lived PID checks.
ready=0
for _ in $(seq 1 12); do
  if grep -q "Listening on 0.0.0.0:19105" "${TMPD}/slave1.log" \
    && grep -q "Listening on 0.0.0.0:19205" "${TMPD}/slave2.log" \
    && grep -q "New neighbor discovered: 1" "${TMPD}/master1.log" \
    && grep -q "New neighbor discovered: 0" "${TMPD}/master2.log"; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
  fail "startup_evidence_incomplete"
fi

congestion_count="$(grep -hc 'Severe Congestion' "${TMPD}"/master*.log 2>/dev/null | awk '{s+=$1} END{print s+0}' || true)"
if [[ "${congestion_count}" -gt 0 ]]; then
  warn "gateway_severe_congestion_detected_count=${congestion_count}"
fi

write_summary "PASS" "modeb_multi_instance_smoke_startup_evidence_ok"
echo "[INFO] Mode B multi-instance smoke passed."
