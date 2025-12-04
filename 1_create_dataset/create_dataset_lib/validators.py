import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, TypedDict

logger = logging.getLogger(__name__)


class ValidationResult(TypedDict):
    available: bool
    output: str


class KBValidationResult(TypedDict):
    links: Dict[str, List[str]]
    rot_pairs: Dict[str, bool]
    markdownlint: ValidationResult
    mermaid: ValidationResult


def check_links_in_kb(kb_dir: str) -> List[str]:
    """Check for broken relative links in md files under kb_dir.

    Returns list of broken links as strings.
    """
    broken = []
    kb_path = Path(kb_dir)
    md_files = list(kb_path.glob("*.md"))
    existing = {p.name for p in md_files}
    link_re = re.compile(r"\[[^\]]+\]\((\.\/[^)]+)\)")
    for file in md_files:
        text = file.read_text(encoding="utf-8")
        for match in link_re.finditer(text):
            target = match.group(1).lstrip("./")
            if target not in existing:
                broken.append(f"{file.name} -> {target}")
    return broken


def check_rot_pairs(kb_dir: str, expected_pairs: int) -> bool:
    # simple check: count files with _v in filename as rot pairs
    files = list(Path(kb_dir).glob("*.md"))
    versioned = [f for f in files if "_v" in f.stem]
    # Each pair contributes 2 files; hence number of pairs is len(versioned) / 2
    pairs = len(versioned) // 2
    return pairs >= expected_pairs


# (Deprecated) summary-style validate_kb was removed in favor of the richer
# `validate_kb` result structure below. If a simple summary is needed, use
# `check_links_in_kb` and `check_rot_pairs` directly.


def run_markdownlint(kb_dir: str) -> ValidationResult:
    """If `markdownlint` is available on PATH, run it on kb_dir and return results.

    Returns a dict with keys: 'available' True/False, 'output' str
    """
    if shutil.which("markdownlint") is None:
        return {"available": False, "output": "markdownlint not found"}
    try:
        proc = subprocess.run(
            ["markdownlint", str(kb_dir)], capture_output=True, text=True
        )
        return {"available": True, "output": proc.stdout + proc.stderr}
    except Exception as e:
        return {"available": True, "output": f"markdownlint run failed: {e}"}


def run_mermaid_validation(kb_dir: str) -> ValidationResult:
    """If `mmdc` (mermaid CLI) is available on PATH, try to validate Mermaid code blocks.

    This is a simple check: extract mermaid code blocks to temporary files and call `mmdc` to render e.g., png.
    """
    if shutil.which("mmdc") is None:
        return {"available": False, "output": "mmdc not found"}
    results = []
    kb_path = Path(kb_dir)
    md_files = list(kb_path.glob("*.md"))
    mermaid_re = re.compile(r"```mermaid\n(.*?)\n```", re.S)
    for file in md_files:
        text = file.read_text(encoding="utf-8")
        for idx, match in enumerate(mermaid_re.findall(text)):
            with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False) as tmp:
                tmp.write(match)
                tmp.flush()
                tmp_name = tmp.name
            out_png = tmp_name + ".png"
            try:
                proc = subprocess.run(
                    ["mmdc", "-i", tmp_name, "-o", out_png],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    results.append(
                        f"{file.name} mermaid block {idx}: FAILED - {proc.stderr}"
                    )
                else:
                    results.append(f"{file.name} mermaid block {idx}: OK")
            except Exception as e:
                results.append(f"{file.name} mermaid block {idx}: ERROR - {e}")
            finally:
                try:
                    os.unlink(tmp_name)
                    if os.path.exists(out_png):
                        os.unlink(out_png)
                except Exception:
                    pass
    return {"available": True, "output": "\n".join(results)}


def validate_kb(kb_dir: str, expected_rot_pairs: int = 10) -> KBValidationResult:
    res: KBValidationResult = {
        "links": {"broken": check_links_in_kb(kb_dir)},
        "rot_pairs": {"ok": check_rot_pairs(kb_dir, expected_rot_pairs)},
        "markdownlint": run_markdownlint(kb_dir),
        "mermaid": run_mermaid_validation(kb_dir),
    }
    logger.info(
        "Validation summary for %s: broken=%s, rot_ok=%s, markdownlint=%s, mermaid=%s",
        kb_dir,
        res["links"]["broken"],
        res["rot_pairs"]["ok"],
        res["markdownlint"]["available"],
        res["mermaid"]["available"],
    )
    return res
