"""Validate experiment outputs."""

import json
import sys
from pathlib import Path


def validate_result_file(filepath: Path, experiment: str) -> bool:
    """Validate a result JSONL file."""
    if not filepath.exists():
        print(f"❌ {filepath} does not exist")
        return False

    with open(filepath, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        print(f"❌ {filepath} is empty")
        return False

    print(f"✓ {filepath} has {len(lines)} results")

    # Validate format
    for i, line in enumerate(lines[:5]):  # Check first 5
        try:
            result = json.loads(line)
            assert "query_id" in result
            assert "experiment" in result
            assert result["experiment"] == experiment
            assert "llm_answer" in result
            assert "retrieved_chunks" in result
            if experiment == "e4":
                assert "reasoning_steps" in result
        except Exception as e:
            print(f"❌ Line {i + 1} invalid: {e}")
            return False

    print(f"✓ Format validation passed for {filepath}")
    return True


if __name__ == "__main__":
    kb_dir = Path("output/kb")

    all_valid = True
    all_valid &= validate_result_file(kb_dir / "e2_standard_rag.jsonl", "e2")
    all_valid &= validate_result_file(kb_dir / "e3_filtered_rag.jsonl", "e3")
    all_valid &= validate_result_file(kb_dir / "e4_reasoning_rag.jsonl", "e4")

    if all_valid:
        print("\n✅ All output files are valid")
        sys.exit(0)
    else:
        print("\n❌ Some output files are invalid")
        sys.exit(1)
