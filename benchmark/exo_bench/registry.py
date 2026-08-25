from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import require_sha256, sha256_file, sha256_tree
from .paths import ALIASES_PATH, CONTROLLERS_ROOT, COURSES_ROOT, RESET_PROFILES_ROOT, ROBOTS_ROOT


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class ResolvedArtifacts:
    robot_id: str
    controller_id: str
    course_ref: str
    robot_dir: Path
    controller_dir: Path
    course_path: Path
    reset_profile_path: Path
    robot: dict[str, Any]
    controller: dict[str, Any]
    course: dict[str, Any]
    reset_profile_id: str
    reset_profile: dict[str, Any]


class ArtifactRegistry:
    def __init__(self) -> None:
        self.aliases = _read_json(ALIASES_PATH)

    def resolve(
        self,
        robot: str,
        controller: str,
        course: str,
        reset_profile: str = "basic",
    ) -> ResolvedArtifacts:
        robot_id = self.aliases["robots"].get(robot, robot)
        controller_id = self.aliases["controllers"].get(controller, controller)
        course_ref = self.aliases["courses"].get(course, course)
        reset_profile_id = self.aliases["reset_profiles"].get(reset_profile, reset_profile)
        course_id, separator, version = course_ref.partition("@")
        robot_dir = ROBOTS_ROOT / robot_id
        controller_dir = CONTROLLERS_ROOT / controller_id
        course_path = COURSES_ROOT / f"{course_id}.json"
        reset_profile_path = RESET_PROFILES_ROOT / f"{reset_profile_id}.json"
        resolved = ResolvedArtifacts(
            robot_id=robot_id,
            controller_id=controller_id,
            course_ref=course_ref,
            robot_dir=robot_dir,
            controller_dir=controller_dir,
            course_path=course_path,
            reset_profile_path=reset_profile_path,
            robot=_read_json(robot_dir / "manifest.json"),
            controller=_read_json(controller_dir / "manifest.json"),
            course=_read_json(course_path),
            reset_profile_id=reset_profile_id,
            reset_profile=_read_json(reset_profile_path),
        )
        if resolved.robot["robot_id"] != robot_id:
            raise ValueError(f"Robot manifest ID mismatch: {robot_id}")
        if resolved.controller["controller_id"] != controller_id:
            raise ValueError(f"Controller manifest ID mismatch: {controller_id}")
        if resolved.controller["robot_id"] != robot_id:
            raise ValueError(f"Controller {controller_id} is not bound to {robot_id}")
        if separator and resolved.course["version"] != version:
            raise ValueError(f"Course version mismatch: requested {course_ref}")
        if resolved.reset_profile["profile_id"] != reset_profile_id:
            raise ValueError(f"Reset profile ID mismatch: {reset_profile_id}")
        return resolved

    @staticmethod
    def verify_hashes(artifacts: ResolvedArtifacts) -> dict[str, str]:
        scene = artifacts.robot_dir / artifacts.robot["model"]["entrypoint"]
        model_xml = artifacts.robot_dir / "model" / "h1.xml"
        policy = artifacts.controller_dir / artifacts.controller["policy"]["path"]
        upstream_config = artifacts.controller_dir / artifacts.controller["upstream_config"]["path"]
        actual = {
            "scene_sha256": sha256_file(scene),
            "model_xml_sha256": sha256_file(model_xml),
            "model_tree_sha256": sha256_tree(artifacts.robot_dir / "model"),
            "policy_sha256": sha256_file(policy),
            "controller_config_sha256": sha256_file(upstream_config),
            "course_sha256": sha256_file(artifacts.course_path),
            "reset_profile_sha256": sha256_file(artifacts.reset_profile_path),
        }
        expected = {
            "scene_sha256": artifacts.robot["model"]["entrypoint_sha256"],
            "model_xml_sha256": artifacts.robot["model"]["model_xml_sha256"],
            "policy_sha256": artifacts.controller["policy"]["sha256"],
            "controller_config_sha256": artifacts.controller["upstream_config"]["sha256"],
        }
        for name, expected_hash in expected.items():
            require_sha256(expected_hash, name)
            if actual[name] != expected_hash:
                raise ValueError(f"Artifact hash mismatch for {name}: {actual[name]} != {expected_hash}")
        pinned_tree = require_sha256(artifacts.robot["model"]["tree_sha256"], "model_tree_sha256")
        if actual["model_tree_sha256"] != pinned_tree:
            raise ValueError("Model tree hash does not match the immutable manifest")
        return actual
