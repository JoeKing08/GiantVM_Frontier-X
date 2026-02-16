#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_ROOT="${ROOT_DIR}/artifacts"
mkdir -p "${ART_ROOT}"
TMPD="${ART_ROOT}/modea-smoke-$(date +%Y%m%d-%H%M%S)"
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
  tail -n 120 "${TMPD}/master.log" 2>/dev/null || true
  tail -n 120 "${TMPD}/slave.log" 2>/dev/null || true
  sudo dmesg | tail -n 120 > "${TMPD}/dmesg_tail.log" 2>/dev/null || true
  exit 1
}

warn() {
  local reason="$1"
  echo "[WARN] ${reason}"
}

cleanup() {
  local rc=$?
  set +e
  if [[ -n "${MASTER_PID:-}" ]]; then kill "${MASTER_PID}" 2>/dev/null || true; fi
  if [[ -n "${SLAVE_PID:-}" ]]; then kill "${SLAVE_PID}" 2>/dev/null || true; fi
  wait "${MASTER_PID:-}" "${SLAVE_PID:-}" 2>/dev/null || true
  sudo rmmod wavevm 2>/dev/null || true
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

if ! sudo -n true >/dev/null 2>&1; then
  fail "sudo_passwordless_required_for_modea_smoke"
fi

if [[ ! -e /dev/kvm ]]; then
  fail "kvm_missing_modea_requires_dev_kvm"
fi

KREL="$(uname -r)"
if [[ ! -d "/lib/modules/${KREL}/build" ]]; then
  fail "kernel_headers_missing_/lib/modules/${KREL}/build"
fi

echo "[INFO] building wavevm.ko ..."
if ! make -C "/lib/modules/${KREL}/build" M="${ROOT_DIR}/master_core" modules > "${TMPD}/kbuild.log" 2>&1; then
  fail "kernel_module_build_failed"
fi

echo "[INFO] loading wavevm.ko ..."
if ! sudo insmod "${ROOT_DIR}/master_core/wavevm.ko"; then
  fail "insmod_wavevm_ko_failed"
fi
if [[ ! -e /dev/wavevm ]]; then
  fail "dev_wavevm_missing_after_insmod"
fi
sudo chmod 666 /dev/wavevm
echo "[INFO] /dev/wavevm perms: $(ls -l /dev/wavevm)"

SWARM_CFG="${TMPD}/swarm_modea.conf"
cat > "${SWARM_CFG}" <<'EOF'
NODE 0 127.0.0.1 29100 1 1
EOF

echo "[INFO] starting Mode A slave..."
(
  cd "${ROOT_DIR}" && \
  WVM_SHM_FILE=/wvm_modea_slave exec ./slave_daemon/wavevm_node_slave 29105 1 1024 0 29101
) > "${TMPD}/slave.log" 2>&1 &
SLAVE_PID=$!

echo "[INFO] starting Mode A master..."
(
  cd "${ROOT_DIR}" && \
  WVM_INSTANCE_ID=modea WVM_SHM_FILE=/wvm_modea_master exec \
    ./master_core/wavevm_node_master 1024 29100 "${SWARM_CFG}" 0 29101 29105 1
) > "${TMPD}/master.log" 2>&1 &
MASTER_PID=$!

sleep 8
ps -p "${MASTER_PID},${SLAVE_PID}" -o pid,comm,state --no-headers > "${TMPD}/ps.log" || true

if grep -qE "Segmentation fault|Failed to init|Resource Mismatch|CRASH on OOB access|bind .* failed|Operation not permitted|RX socket create failed" "${TMPD}"/*.log; then
  fail "failure_signature_detected_in_modea_logs"
fi

ready=0
for _ in $(seq 1 12); do
  if grep -q "Listening on 0.0.0.0:29105" "${TMPD}/slave.log"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  fail "modea_slave_startup_evidence_missing"
fi

if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
  warn "modea_master_not_alive_after_probe_window"
fi

sudo dmesg | tail -n 120 > "${TMPD}/dmesg_tail.log" || true
write_summary "PASS" "modea_smoke_startup_evidence_ok"
echo "[INFO] Mode A smoke passed."
