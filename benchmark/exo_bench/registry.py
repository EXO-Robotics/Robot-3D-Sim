from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_tree
from .paths import ALIASES_PATH, CONTROLLERS_ROOT, COURSES_ROOT, ROBOTS_ROOT


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
    robot: dict[str, Any]
    controller: dict[str, Any]
    course: dict[str, Any]


class ArtifactRegistry:
    def __init__(self) -> None:
        self.aliases = _read_json(ALIASES_PATH)

    def resolve(self, robot: str, controller: str, course: str) -> ResolvedArtifacts:
        robot_id = self.aliases["robots"].get(robot, robot)
        controller_id = self.aliases["controllers"].get(controller, controller)
        course_ref = self.aliases["courses"].get(course, course)
        course_id, separator, version = course_ref.partition("@")
        robot_dir = ROBOTS_ROOT / robot_id
        controller_dir = CONTROLLERS_ROOT / controller_id
        course_path = COURSES_ROOT / f"{course_id}.json"
        resolved = ResolvedArtifacts(
            robot_id=robot_id,
            controller_id=controller_id,
            course_ref=course_ref,
            robot_dir=robot_dir,
            controller_dir=controller_dir,
            course_path=course_path,
            robot=_read_json(robot_dir / "manifest.json"),
            controller=_read_json(controller_dir / "manifest.json"),
            course=_read_json(course_path),
        )
        if resolved.robot["robot_id"] != robot_id:
            raise ValueError(f"Robot manifest ID mismatch: {robot_id}")
        if resolved.controller["controller_id"] != controller_id:
            raise ValueError(f"Controller manifest ID mismatch: {controller_id}")
        if resolved.controller["robot_id"] != robot_id:
            raise ValueError(f"Controller {controller_id} is not bound to {robot_id}")
        if separator and resolved.course["version"] != version:
            raise ValueError(f"Course version mismatch: requested {course_ref}")
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
        }
        expected = {
            "scene_sha256": artifacts.robot["model"]["entrypoint_sha256"],
            "model_xml_sha256": artifacts.robot["model"]["model_xml_sha256"],
            "policy_sha256": artifacts.controller["policy"]["sha256"],
            "controller_config_sha256": artifacts.controller["upstream_config"]["sha256"],
        }
        for name, expected_hash in expected.items():
            if actual[name] != expected_hash:
                raise ValueError(f"Artifact hash mismatch for {name}: {actual[name]} != {expected_hash}")
        pinned_tree = artifacts.robot["model"]["tree_sha256"]
        if pinned_tree != "pending-runtime-verification" and actual["model_tree_sha256"] != pinned_tree:
            raise ValueError("Model tree hash does not match the immutable manifest")
        return actual
