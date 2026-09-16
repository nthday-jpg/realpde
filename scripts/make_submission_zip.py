#!/usr/bin/env python3
"""Assemble a self-contained Codabench submission zip from a lean variant folder.

Usage:
    python scripts/make_submission_zip.py submission_v1
    python scripts/make_submission_zip.py submission_v2 --with-model /path/to/sim_real_cno.pth
    python scripts/make_submission_zip.py submission_v1 --out dist/custom.zip

What it does:
  1. Copies ``submissions/<variant>/`` contents (``submission.py``,
     ``policy.yaml``, ``model.pth`` if present) into a temp staging dir.
  2. Injects shared files at pack time (never stored in the variant folder):
     - ``ttt_model.py`` (always; submissions fall back without it, but the
       base class documents the contract),
     - ``load_baseline.py`` + ``realpde/`` package tree (only when
       ``submission.py`` imports ``load_baseline`` — detected by text search).
  3. Zips the staging *contents* so ``submission.py`` is at the zip root
     (Codabench requirement), writes ``dist/<variant>.zip``.
  4. Asserts ``submission.py`` at root and extracted size < 256 MB.

``model.pth`` resolution order: variant folder > ``--with-model`` > absent
(TinyForecaster fallback; fine for smoke tests, not for leaderboard).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

SIZE_CAP_MB = 256

REPO_ROOT = Path(__file__).resolve().parent.parent


def needs_load_baseline(submission_text: str) -> bool:
    return "load_baseline" in submission_text


def copy_tree_clean(src: Path, dst: Path) -> None:
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def apply_policy_overrides(stage: Path, overrides: list[str]) -> None:
    """Apply KEY=VALUE overrides to staged policy.yaml (flat k:v lines only)."""
    if not overrides:
        return
    policy = stage / "policy.yaml"
    if not policy.exists():
        sys.exit("[error] --set given but variant has no policy.yaml")
    kv: dict[str, str] = {}
    for item in overrides:
        if "=" not in item:
            sys.exit(f"[error] --set expects KEY=VALUE, got {item!r}")
        k, v = (s.strip() for s in item.split("=", 1))
        kv[k] = v
    lines = policy.read_text(encoding="utf-8").splitlines()
    seen = set()
    out: list[str] = []
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if stripped and ":" in stripped and not line.startswith((" ", "\t")):
            key = stripped.split(":", 1)[0].strip()
            if key in kv:
                out.append(f"{key}: {kv[key]}")
                seen.add(key)
                continue
        out.append(line)
    for key, val in kv.items():
        if key not in seen:
            out.append(f"{key}: {val}")
    policy.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[pack] policy overrides: {kv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variant", help="e.g. submission_v1 (under submissions/)")
    ap.add_argument("--with-model", default=None,
                    help="checkpoint to stage as model.pth (variant folder wins)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a flat policy.yaml key in the staged zip "
                         "(repeatable, e.g. --set base_model=cno)")
    ap.add_argument("--out", default=None, help="output zip path (default dist/<variant>.zip)")
    args = ap.parse_args()

    variant_dir = REPO_ROOT / "submissions" / args.variant
    if not variant_dir.is_dir():
        sys.exit(f"[error] no such variant dir: {variant_dir}")
    submission_py = variant_dir / "submission.py"
    if not submission_py.exists():
        sys.exit(f"[error] {variant_dir} has no submission.py")

    text = submission_py.read_text(encoding="utf-8", errors="replace")
    want_baseline = needs_load_baseline(text)

    out = Path(args.out) if args.out else (REPO_ROOT / "dist" / f"{args.variant}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mkzip_") as tmp:
        stage = Path(tmp) / "stage"
        stage.mkdir()

        # 1) variant files
        for item in sorted(variant_dir.iterdir()):
            if item.name in ("__pycache__",) or item.suffix in (".pyc", ".pyo"):
                continue
            if item.is_dir():
                copy_tree_clean(item, stage / item.name)
            else:
                shutil.copy2(item, stage / item.name)

        # 2a) ttt_model.py (always)
        ttt = REPO_ROOT / "ttt_model.py"
        if ttt.exists() and not (stage / "ttt_model.py").exists():
            shutil.copy2(ttt, stage / "ttt_model.py")

        # 2b) load_baseline + realpde tree (only if imported)
        if want_baseline:
            lb = REPO_ROOT / "load_baseline.py"
            if not lb.exists():
                sys.exit("[error] submission imports load_baseline but load_baseline.py missing")
            shutil.copy2(lb, stage / "load_baseline.py")
            src_pkg = REPO_ROOT / "src" / "realpde"
            if not src_pkg.is_dir():
                sys.exit("[error] submission imports load_baseline but src/realpde missing")
            copy_tree_clean(src_pkg, stage / "realpde")
            print("[pack] injected load_baseline.py + realpde/ package")
        else:
            print("[pack] load_baseline not imported — skipping realpde/ tree")

        # 3) model.pth resolution
        staged_model = stage / "model.pth"
        if not staged_model.exists() and args.with_model:
            src_ckpt = Path(args.with_model)
            if not src_ckpt.exists():
                sys.exit(f"[error] --with-model not found: {src_ckpt}")
            shutil.copy2(src_ckpt, staged_model)
            print(f"[pack] staged {src_ckpt.name} as model.pth")
        if staged_model.exists():
            print(f"[pack] model.pth: {staged_model.stat().st_size / 1e6:.1f} MB")
        else:
            print("[pack] no model.pth — TinyForecaster fallback (smoke-test only)")

        # 3b) policy overrides (e.g. base_model=cno when staged as model.pth)
        apply_policy_overrides(stage, args.set)

        # 4) zip staging contents (submission.py at root)
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for item in sorted(stage.rglob("*")):
                if item.is_file():
                    zf.write(item, item.relative_to(stage).as_posix())

        # 5) verify
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            if "submission.py" not in names:
                sys.exit("[error] submission.py not at zip root: " + ", ".join(names[:10]))
            total = sum(i.file_size for i in zf.infolist())
        print(f"[pack] wrote {out} ({out.stat().st_size / 1e6:.1f} MB compressed, "
              f"{total / 1e6:.1f} MB extracted)")
        print("[pack] contents:")
        with zipfile.ZipFile(out) as zf:
            for i in sorted(zf.infolist(), key=lambda e: e.filename)[:30]:
                print(f"    {i.file_size / 1e6:7.2f} MB  {i.filename}")
        if total > SIZE_CAP_MB * 1e6:
            sys.exit(f"[error] extracted size exceeds {SIZE_CAP_MB} MB cap")
        print("[pack] OK")


if __name__ == "__main__":
    main()
