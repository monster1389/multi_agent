"""Code sandbox — execute generated Python solutions against LeetCode tests."""

import subprocess
import tempfile
import os
import sys
from pathlib import Path


EXECUTION_TIMEOUT = 10  # seconds per test run


def execute_code(
    code: str,
    test_code: str,
    entry_point: str,
) -> tuple[bool, str]:
    """Execute generated code against test cases in a subprocess sandbox.

    Args:
        code: The generated Python solution (class Solution with methods).
        test_code: The test harness (def check(candidate): ...).
        entry_point: How to instantiate the solution, e.g. "Solution().twoSum".

    Returns:
        (passed, output) where passed=True if all assertions succeed,
        and output contains stdout/stderr.
    """
    # Build the complete test script
    # The entry_point is something like "Solution().twoSum"
    # We wrap in try/except to catch failures gracefully
    script = f"""\
import sys
import traceback

{code}

{test_code}

if __name__ == "__main__":
    try:
        candidate = {entry_point}
        check(candidate)
        print("__ALL_TESTS_PASSED__")
    except AssertionError as e:
        print(f"ASSERTION_ERROR: {{e}}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"RUNTIME_ERROR: {{e}}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
"""

    # Write to a temporary file and execute
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="leetcode_test_")
        os.close(fd)

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(script)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
        )

        passed = result.returncode == 0 and "__ALL_TESTS_PASSED__" in result.stdout
        output = result.stdout + result.stderr

        if result.returncode != 0 and not result.stderr.strip():
            output += f"\n[Process exited with code {result.returncode}]"

        return passed, output.strip()

    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: execution exceeded {EXECUTION_TIMEOUT}s"
    except Exception as e:
        return False, f"SANDBOX_ERROR: {e}"
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def run_all_tests(
    code: str,
    problem: dict,
) -> tuple[int, int, str]:
    """Run the full test suite for a problem against the generated code.

    Args:
        code: The generated solution code.
        problem: Problem dict with 'test' and 'entry_point' keys.

    Returns:
        (passed_count, total_count, output_log).
        Since the test harness runs all assertions as a single check()
        call, total_count is always 1 and passed_count is 0 or 1.
    """
    test_code = problem.get("test", "")
    entry_point = problem.get("entry_point", "")

    if not test_code or not entry_point:
        return 0, 1, "MISSING_TEST_OR_ENTRY_POINT"

    passed, output = execute_code(code, test_code, entry_point)
    return (1 if passed else 0, 1, output)
