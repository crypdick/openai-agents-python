#!/usr/bin/env python3
"""
Script to test all examples in the examples/ directory with and without Ray backend.
Reports which examples passed/failed and timing differences.
"""

import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

# Files/directories to exclude
EXCLUDED_PATTERNS = [
    "__init__.py",
    "server.py",
    "twilio_handler.py",
    "util.py",
    "printer.py",
    "manager.py",
    "my_workflow.py",
    "app.js",
    "*.ipynb",
]

# Directories to skip entirely
EXCLUDED_DIRS = [
    "realtime/app",
    "realtime/twilio",
    "realtime/twilio_sip",
    "mcp/streamablehttp_example",
    "mcp/streamablehttp_custom_client_example",
    "mcp/sse_example",
    "mcp/prompt_server",
    "voice/static",
    "voice/streamed",
]


class TestResult(NamedTuple):
    """Result of running a test."""

    success: bool
    duration: float
    error: str | None = None


def should_exclude_file(file_path: Path) -> bool:
    """Check if a file should be excluded from testing."""
    # Check if file matches excluded patterns
    if any(pattern in file_path.name for pattern in EXCLUDED_PATTERNS):
        return True

    # Check if file is in excluded directory
    parts = file_path.parts
    for excluded_dir in EXCLUDED_DIRS:
        excluded_parts = tuple(excluded_dir.split("/"))
        if excluded_parts == parts[-len(excluded_parts) :]:
            return True

    return False


def find_example_files() -> list[Path]:
    """Find all Python example files that should be tested."""
    examples_dir = Path("examples")
    example_files = []

    for py_file in examples_dir.rglob("*.py"):
        if should_exclude_file(py_file):
            continue
        example_files.append(py_file)

    return sorted(example_files)


def run_example(
    example_path: Path, use_ray: bool, timeout: int = 300
) -> TestResult:
    """Run an example script and return the result."""
    # Prepare command
    cmd = ["uv", "run", "--active", str(example_path)]

    # Prepare environment
    env = os.environ.copy()
    if use_ray:
        env["RAY_BACKEND"] = "1"
    else:
        # Ensure RAY_BACKEND is not set when not using Ray
        env.pop("RAY_BACKEND", None)

    start_time = time.time()
    try:
        # Run with input "test" piped via stdin
        result = subprocess.run(
            cmd,
            input="test\n",
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        duration = time.time() - start_time

        if result.returncode == 0:
            return TestResult(success=True, duration=duration)
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            # Truncate long error messages
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
            return TestResult(
                success=False, duration=duration, error=error_msg
            )
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return TestResult(
            success=False, duration=duration, error=f"Timeout after {timeout}s"
        )
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            success=False, duration=duration, error=str(e)
        )


def main():
    """Main function to run all tests."""
    example_files = find_example_files()
    print(f"Found {len(example_files)} example files to test\n")

    results: dict[str, dict[str, TestResult]] = defaultdict(dict)

    # Test each example with both backends
    for i, example_path in enumerate(example_files, 1):
        rel_path = str(example_path.relative_to(Path("examples")))
        print(f"[{i}/{len(example_files)}] Testing {rel_path}...")

        # Test without Ray backend
        print(f"  Running without Ray backend...", end=" ", flush=True)
        result_no_ray = run_example(example_path, use_ray=False)
        results[rel_path]["no_ray"] = result_no_ray
        status = "✓" if result_no_ray.success else "✗"
        print(f"{status} ({result_no_ray.duration:.2f}s)")

        # Test with Ray backend
        print(f"  Running with Ray backend...", end=" ", flush=True)
        result_ray = run_example(example_path, use_ray=True)
        results[rel_path]["ray"] = result_ray
        status = "✓" if result_ray.success else "✗"
        print(f"{status} ({result_ray.duration:.2f}s)")

        print()

    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80 + "\n")

    # Group results by status
    passed_both = []
    failed_both = []
    passed_no_ray_only = []
    passed_ray_only = []
    failed_no_ray_only = []
    failed_ray_only = []

    for rel_path, test_results in results.items():
        no_ray = test_results["no_ray"]
        ray = test_results["ray"]

        if no_ray.success and ray.success:
            passed_both.append((rel_path, no_ray, ray))
        elif not no_ray.success and not ray.success:
            failed_both.append((rel_path, no_ray, ray))
        elif no_ray.success and not ray.success:
            passed_no_ray_only.append((rel_path, no_ray, ray))
        elif not no_ray.success and ray.success:
            passed_ray_only.append((rel_path, no_ray, ray))

    # Print results
    print(f"✓ Passed with both backends: {len(passed_both)}")
    print(f"✗ Failed with both backends: {len(failed_both)}")
    print(f"✓ Passed only without Ray: {len(passed_no_ray_only)}")
    print(f"✓ Passed only with Ray: {len(passed_ray_only)}\n")

    # Detailed failure report
    if failed_both:
        print("FAILED WITH BOTH BACKENDS:")
        print("-" * 80)
        for rel_path, no_ray, ray in failed_both:
            print(f"\n{rel_path}:")
            print(f"  No Ray: {no_ray.error or 'Unknown error'}")
            print(f"  Ray:    {ray.error or 'Unknown error'}")
        print()

    if passed_no_ray_only:
        print("PASSED ONLY WITHOUT RAY:")
        print("-" * 80)
        for rel_path, no_ray, ray in passed_no_ray_only:
            print(f"\n{rel_path}:")
            print(f"  Ray error: {ray.error or 'Unknown error'}")
        print()

    if passed_ray_only:
        print("PASSED ONLY WITH RAY:")
        print("-" * 80)
        for rel_path, no_ray, ray in passed_ray_only:
            print(f"\n{rel_path}:")
            print(f"  No Ray error: {no_ray.error or 'Unknown error'}")
        print()

    # Speed comparison
    print("=" * 80)
    print("SPEED COMPARISON (Ray vs No Ray)")
    print("=" * 80 + "\n")

    speed_comparisons = []
    for rel_path, test_results in results.items():
        no_ray = test_results["no_ray"]
        ray = test_results["ray"]

        if no_ray.success and ray.success:
            speedup = no_ray.duration / ray.duration if ray.duration > 0 else 0
            speed_comparisons.append(
                (rel_path, no_ray.duration, ray.duration, speedup)
            )

    # Sort by speedup (largest first)
    speed_comparisons.sort(key=lambda x: x[3], reverse=True)

    print(f"{'Example':<50} {'No Ray':<12} {'Ray':<12} {'Speedup':<10}")
    print("-" * 80)
    for rel_path, no_ray_time, ray_time, speedup in speed_comparisons:
        speedup_str = f"{speedup:.2f}x"
        if speedup < 1:
            speedup_str = f"{1/speedup:.2f}x slower"
        print(
            f"{rel_path:<50} {no_ray_time:>10.2f}s {ray_time:>10.2f}s {speedup_str:>10}"
        )

    # Summary statistics
    if speed_comparisons:
        avg_speedup = sum(x[3] for x in speed_comparisons) / len(speed_comparisons)
        faster_count = sum(1 for x in speed_comparisons if x[3] > 1)
        slower_count = sum(1 for x in speed_comparisons if x[3] < 1)
        print("\n" + "-" * 80)
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Ray faster: {faster_count} examples")
        print(f"Ray slower: {slower_count} examples")
        print(f"No difference: {len(speed_comparisons) - faster_count - slower_count} examples")

    # Exit code
    total_failures = (
        len(failed_both)
        + len(passed_no_ray_only)
        + len(passed_ray_only)
    )
    if total_failures > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

