"""Shared utilities and data structures for skills providers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastmcp.utilities.skills import SkillEntry, SkillFrontmatter, SkillResourceEntry


@dataclass
class SkillFileInfo:
    """Information about a file within a skill."""

    path: str  # Relative path within skill directory
    size: int
    hash: str  # sha256 hash


@dataclass
class SkillInfo:
    """Parsed information about a skill."""

    name: str  # Directory name (canonical identifier)
    description: str  # From frontmatter or first line
    path: Path  # Absolute path to skill directory
    main_file: str  # Name of main file (e.g., "SKILL.md")
    files: list[SkillFileInfo] = field(default_factory=list)
    frontmatter: dict[str, Any] = field(default_factory=dict)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Markdown content potentially starting with ---

    Returns:
        Tuple of (frontmatter dict, remaining content)
    """
    content = content.removeprefix("\ufeff")

    if not content.startswith("---"):
        return {}, content

    # Find the closing ---
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content

    frontmatter_text = content[3 : 3 + end_match.start()]
    remaining = content[3 + end_match.end() :]

    # Parse YAML (simple key: value parsing, no complex types)
    frontmatter: dict[str, Any] = {}
    for line in frontmatter_text.strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # Handle quoted strings
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]

            # Handle lists [a, b, c]
            if value.startswith("[") and value.endswith("]"):
                items = value[1:-1].split(",")
                value = [item.strip().strip("\"'") for item in items if item.strip()]

            frontmatter[key] = value

    return frontmatter, remaining


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def build_skill_entry(skill: SkillInfo, main_file_name: str) -> SkillEntry:
    """Adapt a provider's loosely-parsed `SkillInfo` into a SEP-2640 `SkillEntry`.

    This is the PR 1 bridge from the existing, permissive filesystem parsing
    (`parse_frontmatter`, `scan_skill_files`) to the strict Skills extension
    wire model -- it does not change discovery or parsing behavior. A future
    PR replaces this with byte-accurate, strictly-validated snapshots; until
    then, the extension's real-provider catalog reflects exactly what the
    existing permissive resources already serve.

    `frontmatter.name` is forced to `skill.name` (the directory name) because
    the extension identifies a skill by the URI of its `SKILL.md`, whose final
    path segment SEP-2640 requires to equal `frontmatter.name`; this provider
    already requires the directory name as the skill's identity, so the two
    can never legitimately disagree.
    """
    frontmatter: dict[str, Any] = dict(skill.frontmatter)
    frontmatter["name"] = skill.name
    frontmatter.setdefault("description", skill.description)

    resources = [
        SkillResourceEntry(
            uri=f"skill://{skill.name}/{quote(f.path, safe='/')}",
            digest=f.hash,
            size=f.size,
        )
        for f in skill.files
    ]

    return SkillEntry(
        uri=f"skill://{skill.name}/{quote(main_file_name, safe='/')}",
        frontmatter=SkillFrontmatter.model_validate(frontmatter),
        resources=resources,
    )


def scan_skill_files(skill_dir: Path) -> list[SkillFileInfo]:
    """Scan a skill directory for all files."""
    files = []
    resolved_skill_dir = skill_dir.resolve()

    # Sort for deterministic ordering across platforms
    for file_path in sorted(skill_dir.rglob("*")):
        if file_path.is_file():
            resolved_file_path = file_path.resolve()
            if not resolved_file_path.is_relative_to(resolved_skill_dir):
                continue

            rel_path = file_path.relative_to(skill_dir)
            files.append(
                SkillFileInfo(
                    # Use POSIX paths for cross-platform URI consistency
                    path=rel_path.as_posix(),
                    size=resolved_file_path.stat().st_size,
                    hash=compute_file_hash(resolved_file_path),
                )
            )
    return files
