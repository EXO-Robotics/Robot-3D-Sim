from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmark"
ALIASES_PATH = BENCHMARK_ROOT / "registry" / "aliases.json"
ROBOTS_ROOT = BENCHMARK_ROOT / "robots"
CONTROLLERS_ROOT = BENCHMARK_ROOT / "controllers"
COURSES_ROOT = BENCHMARK_ROOT / "courses"
RESET_PROFILES_ROOT = BENCHMARK_ROOT / "reset_profiles"
QUALIFICATION_SPEC_PATH = BENCHMARK_ROOT / "qualification" / "foundation-v1.spec.json"
UV_LOCK_PATH = REPOSITORY_ROOT / "uv.lock"
