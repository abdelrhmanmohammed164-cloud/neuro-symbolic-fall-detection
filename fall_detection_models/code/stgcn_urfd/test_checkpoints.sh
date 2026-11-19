#!/bin/bash
# Recovery tests for preprocess.py on DISPOSABLE test data (never on the real project folder).
# Usage: bash test_checkpoints.sh <test_root> <reference_root> [device]
# <reference_root> must hold the same videos, processed once without interruption.
R="$1"; REF="$2"; DEV="${3:-cpu}"
cd "$(dirname "$0")"
P="python3 preprocess.py --root $R --chunk_size 20 --device $DEV"
LOG="$R/logs/processing.log"
q() { grep -v -i -E "warn|xformers|onnxruntime backend|VerifyEachNode|GetCapability" ; }
hr() { echo; echo "################ $*"; }
CK() { ls "$R"/checkpoints/cfg_*/"$1" 2>/dev/null | tr '\n' ' '; echo; }

hr "T1  stop in the middle of a video (process killed after its 1st committed chunk) - fall-91"
PREPROC_TEST_KILL_AFTER_CHUNK=1 $P --resume --videos fall-91 2>&1 | q | tail -3; echo "exit code: ${PIPESTATUS[0]}"
$P --status --videos fall-91 2>&1 | q | tail -3
echo "chunk files: $(CK fall-91)"

hr "T2  restart -> chunk-level resume (killed again after one more chunk)"
PREPROC_TEST_KILL_AFTER_CHUNK=1 $P --resume --videos fall-91 2>&1 | q | grep -E "chunk|TEST"; echo "exit code: ${PIPESTATUS[0]}"
echo "times chunk 0 was computed so far: $(grep -c 'fall-91 chunk 0 \[0, 20) saved' $LOG)  (must be 1)"
$P --status --videos fall-91 2>&1 | q | grep fall-91

hr "T5  corrupt a checkpoint (overwrite 4 KB in the middle of fall-91 chunk_0000.npz)"
F=$(ls "$R"/checkpoints/cfg_*/fall-91/chunk_0000.npz)
python3 -c "import os; f=open('$F','r+b'); f.seek(os.path.getsize('$F')//2); f.write(os.urandom(4096)); f.close()"
$P --validate-checkpoints --videos fall-91 2>&1 | q; echo "validate exit code: ${PIPESTATUS[0]} (1 = problem found)"
$P --resume --videos fall-91 2>&1 | q | grep -E "fall-91" | grep -v "models loaded"
echo "chunk files after recovery: $(CK fall-91)"

hr "T3  interrupt WHILE writing a checkpoint (fall-92, crash in the middle of the 2nd chunk write)"
PREPROC_TEST_CRASH_IN_WRITE=2 $P --resume --videos fall-92 2>&1 | q | grep -E "chunk|TEST"; echo "exit code: ${PIPESTATUS[0]}"
echo "chunk files: $(CK fall-92)"

hr "T4  incomplete checkpoint is detected, then recovered"
$P --validate-checkpoints --videos fall-92 2>&1 | q; echo "validate exit code: ${PIPESTATUS[0]}"
$P --resume --videos fall-92 2>&1 | q | grep -E "fall-92" | grep -v "models loaded"
echo "chunk files: $(CK fall-92)"

hr "T4b crash AFTER the atomic rename but BEFORE the manifest commit (fall-93 chunk 0)"
PREPROC_TEST_CRASH_BEFORE_COMMIT=0 $P --resume --videos fall-93 2>&1 | q | grep -E "TEST"; echo "exit code: ${PIPESTATUS[0]}"
$P --status --videos fall-93 2>&1 | q | grep fall-93
echo "chunk files: $(CK fall-93)"
$P --resume --videos fall-93 2>&1 | q | grep -E "fall-93" | grep -v "models loaded"

hr "T1b real SIGKILL (kill -9) while processing fall-94"
$P --resume --videos fall-94 > /dev/null 2>&1 &
PID=$!
until grep -q 'fall-94 chunk 0 \[0, 20) saved' "$LOG" 2>/dev/null; do sleep 2; done
sleep 5; kill -9 $PID; wait $PID 2>/dev/null; echo "killed pid $PID with SIGKILL"
$P --status --videos fall-94 2>&1 | q | grep fall-94
echo "chunk files: $(CK fall-94)"
$P --resume --videos fall-94 2>&1 | q | grep -E "fall-94" | grep -v "models loaded"

hr "T6  completed videos are skipped"
$P --resume 2>&1 | q | grep -E "skip|done"

hr "T7  configuration change invalidates checkpoints (det_thr 0.1 -> 0.2)"
OLD=$(ls "$R"/processed_data/cfg_*/urfd_npy/fall-92.npy | head -1); SHA_BEFORE=$(sha256sum "$OLD" | cut -c1-16)
$P --det_thr 0.2 --status 2>&1 | q | tail -3
$P --det_thr 0.2 --resume --videos fall-92 2>&1 | q | grep -E "fall-92.*(reused|COMPLETED)|=== resume"
echo "old output $OLD sha before ${SHA_BEFORE} after $(sha256sum "$OLD" | cut -c1-16) (must be equal)"
echo "config dirs: $(ls "$R"/checkpoints | grep cfg_ | tr '\n' ' ')"

hr "T8  no duplicated / skipped frames, outputs identical to the uninterrupted reference"
python3 - "$R" "$REF" <<'EOF'
import sys, glob, os, numpy as np
R, REF = sys.argv[1:]
ok = True
ref_dir = glob.glob(f"{REF}/processed_data/cfg_*/urfd_npy")[0]
run_dir = os.path.join(R, "processed_data", os.path.basename(os.path.dirname(ref_dir)), "urfd_npy")
for seq in ["fall-91", "fall-92", "fall-93", "fall-94"]:
    res = []
    for suf in ["", "_anchor", "_neck_depth", "_raw_coco17", "_timestamps"]:
        a, b = np.load(f"{run_dir}/{seq}{suf}.npy"), np.load(f"{ref_dir}/{seq}{suf}.npy")
        eq = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        ok &= eq; res.append(f"{suf or 'skeleton'}:{'=' if eq else 'DIFF'}")
    chunks = sorted(glob.glob(f"{R}/checkpoints/{os.path.basename(os.path.dirname(ref_dir))}/{seq}/chunk_*.npz"))
    idx = np.concatenate([np.load(c)["frame_idx"] for c in chunks])
    T = np.load(f"{run_dir}/{seq}.npy").shape[1]
    cov = np.array_equal(idx, np.arange(T))
    ok &= cov
    print(f"{seq}: T={T} frames covered exactly once: {cov} | vs reference {' '.join(res)}")
print("T8 RESULT:", "PASS" if ok else "FAIL")
EOF
