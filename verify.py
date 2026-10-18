"""Verify the package: stored output intact, every generated table reproduced, and the
user-facing module ccpa.py in agreement with the simulation engine.

    python verify.py               # all three checks
    python verify.py --data        # stored output against MANIFEST.sha256 only
    python verify.py --manifest    # rewrite MANIFEST.sha256 (after a deliberate re-run)
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "MANIFEST.sha256")
ROOTS = ("output", os.path.join("ldv_scope", "output"), "data")
SKIP = {"rv_window_store", "rv_logs"}   # gitignored: not shipped


def files():
    for root in ROOTS:
        for d, sub, fs in os.walk(os.path.join(HERE, root)):
            sub[:] = sorted(s for s in sub if s not in SKIP)
            for f in sorted(fs):
                if not f.startswith("."):
                    p = os.path.join(d, f)
                    yield os.path.relpath(p, HERE)


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(HERE, p), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    with open(MANIFEST, "w") as fh:
        for p in files():
            fh.write(f"{sha(p)}  {p}\n")
    print(f"wrote {MANIFEST}")


def check_data():
    bad = 0
    listed = {}
    for line in open(MANIFEST):
        digest, p = line.rstrip("\n").split("  ", 1)
        listed[p] = digest
    for p, digest in listed.items():
        if not os.path.exists(os.path.join(HERE, p)):
            print("missing  ", p); bad += 1
        elif sha(p) != digest:
            print("modified ", p); bad += 1
    for p in files():
        if p not in listed:
            print("unlisted ", p)
    print("stored output intact" if not bad else f"{bad} file(s) differ from the manifest")
    return bad == 0


if __name__ == "__main__":
    if "--manifest" in sys.argv:
        write_manifest(); sys.exit(0)
    ok = check_data()
    if "--data" not in sys.argv:
        for cmd in ([sys.executable, "build_tables.py", "--check"],
                    [sys.executable, os.path.join("tests", "test_ccpa.py")]):
            r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or [""])[-1]
            print(last)
            if r.returncode != 0:
                err = (r.stderr.strip().splitlines() or ["(no error message)"])[-1]
                print(f"  FAILED: {' '.join(cmd[1:])}: {err}")
            ok &= (r.returncode == 0 and ("MISMATCH" not in r.stdout))
    sys.exit(0 if ok else 1)
