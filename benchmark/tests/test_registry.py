from exo_bench.registry import ArtifactRegistry


def test_aliases_resolve_to_hash_bound_pair() -> None:
    registry = ArtifactRegistry()
    artifacts = registry.resolve("h1", "baseline", "basic-walk-stop")
    assert artifacts.robot_id == "exo.h1.v1"
    assert artifacts.controller_id == "exo.h1.velocity.v1"
    assert artifacts.course_ref == "BASIC-003@1.0.0"
    assert artifacts.robot["status"] == "qualified"
    assert artifacts.controller["status"] == "qualified"


def test_pinned_artifact_hashes_match() -> None:
    registry = ArtifactRegistry()
    artifacts = registry.resolve("h1", "baseline", "BASIC-001")
    hashes = registry.verify_hashes(artifacts)
    assert hashes["model_tree_sha256"] == "12c7d68fabc0f24e7ce5ed46697cc41ccdce41eebe464b87201179a3b0a0136a"
    assert hashes["policy_sha256"] == "44a0fbceb81f3877833ae9a398d039bea1759cb0d3c8188181013885f70589eb"

