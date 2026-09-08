"""Client utilities for discovering and downloading skills from MCP servers."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mcp_types
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from fastmcp.client import Client

#: Reverse-DNS identifier of the SEP-2640 Skills extension
#: (`io.modelcontextprotocol/skills`). Shared by the server-side
#: `SkillsExtension` and the client-side `SkillsClientExtension` so neither
#: needs to import the other's package.
SKILLS_EXTENSION_ID = "io.modelcontextprotocol/skills"

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


# -----------------------------------------------------------------------------
# SEP-2640 wire protocol models
#
# These mirror the `Skill`/`SkillResource` shapes defined by the Skills
# extension specification (ext-skills `specification/stable/skills.mdx`).
# `SkillEntry` is the single entry shape returned, identically, by both
# `skills/list` and `skills/get` -- the spec calls it "a complete manifest of
# the skill rather than a summary", so there is no separate summary model.
# -----------------------------------------------------------------------------


class SkillFrontmatter(BaseModel):
    """A skill's `SKILL.md` YAML frontmatter, rendered verbatim as JSON.

    `name` and `description` are required by the Agent Skills specification and
    therefore always present. Every other author-written field passes through
    unchanged (`extra="allow"`), per SEP-2640's frontmatter contract.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    description: str


class SkillResourceEntry(BaseModel):
    """One file of a skill, with the digest and size of its raw bytes.

    Called `SkillResource` in the SEP-2640 spec; named `SkillResourceEntry`
    here to avoid colliding with `fastmcp.resources.base.Resource`.
    """

    model_config = ConfigDict(extra="forbid")

    uri: str
    digest: str
    size: int = Field(ge=0)

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, v: str) -> str:
        if not _DIGEST_PATTERN.match(v):
            raise ValueError(
                f"digest must be 'sha256:' followed by 64 lowercase hex "
                f"characters, got {v!r}"
            )
        return v


class SkillEntry(BaseModel):
    """The entry for a single skill, identical in shape from `skills/list` and
    `skills/get`.

    `resources` is either a complete enumeration of the skill's files or the
    literal string `"dynamic"` for skills whose content is generated such that
    stable digests cannot be published.
    """

    model_config = ConfigDict(extra="forbid")

    uri: str
    frontmatter: SkillFrontmatter
    resources: list[SkillResourceEntry] | Literal["dynamic"]


class GetSkillRequestParams(mcp_types.RequestParams):
    """Params for `skills/get`."""

    uri: str


class GetSkillRequest(mcp_types.Request[GetSkillRequestParams, Literal["skills/get"]]):
    """Client-side wire request for `skills/get`."""

    method: Literal["skills/get"] = "skills/get"
    params: GetSkillRequestParams


class ListSkillsRequest(mcp_types.PaginatedRequest[Literal["skills/list"]]):
    """Client-side wire request for `skills/list`."""

    method: Literal["skills/list"] = "skills/list"


class ListSkillsResult(mcp_types.PaginatedResult, mcp_types.CacheableResult):
    """Result of `skills/list`: a page of the server's skill catalog."""

    result_type: mcp_types.ResultType = "complete"
    skills: list[SkillEntry] = Field(default_factory=list)


class GetSkillResult(mcp_types.Result):
    """Result of `skills/get`: the entry for the requested skill."""

    result_type: mcp_types.ResultType = "complete"
    skill: SkillEntry


class SkillsExtensionSettings(BaseModel):
    """Per-extension settings advertised at `capabilities.extensions[identifier]`.

    `directory_read` is always `False` in this release (`resources/directory/read`
    is not yet implemented), so `model_dump` always produces `{}` -- an empty
    object per SEP-2640 meaning "support for the extension with no optional
    features".
    """

    model_config = ConfigDict(extra="forbid")

    directory_read: bool = Field(default=False, serialization_alias="directoryRead")

    def to_wire(self) -> dict[str, bool]:
        return self.model_dump(by_alias=True, exclude_defaults=True)


@dataclass
class SkillSummary:
    """Summary information about a skill available on a server."""

    name: str
    description: str
    uri: str


@dataclass
class SkillFile:
    """Information about a file within a skill."""

    path: str
    size: int
    hash: str


@dataclass
class SkillManifest:
    """Full manifest of a skill including all files."""

    name: str
    files: list[SkillFile]


async def list_skills(client: Client) -> list[SkillSummary]:
    """List all available skills from an MCP server.

    Discovers skills by finding resources with URIs matching the
    `skill://{name}/SKILL.md` pattern.

    Args:
        client: Connected FastMCP client

    Returns:
        List of SkillSummary objects with name, description, and URI

    Example:
        ```python
        from fastmcp import Client
        from fastmcp.utilities.skills import list_skills

        async with Client("http://skills-server/mcp") as client:
            skills = await list_skills(client)
            for skill in skills:
                print(f"{skill.name}: {skill.description}")
        ```
    """
    resources = await client.list_resources()
    skills = []

    for resource in resources:
        uri = str(resource.uri)
        # Match skill://{name}/SKILL.md pattern
        if uri.startswith("skill://") and uri.endswith("/SKILL.md"):
            # Extract skill name from URI
            path_part = uri[len("skill://") :]
            name = path_part.rsplit("/", 1)[0]
            skills.append(
                SkillSummary(
                    name=name,
                    description=resource.description or "",
                    uri=uri,
                )
            )

    return skills


async def get_skill_manifest(client: Client, skill_name: str) -> SkillManifest:
    """Get the manifest for a specific skill.

    Args:
        client: Connected FastMCP client
        skill_name: Name of the skill

    Returns:
        SkillManifest with file listing

    Raises:
        ValueError: If manifest cannot be read or parsed
    """
    manifest_uri = f"skill://{skill_name}/_manifest"
    result = await client.read_resource(manifest_uri)

    if not result:
        raise ValueError(f"Could not read manifest for skill: {skill_name}")

    content = result[0]
    if isinstance(content, mcp_types.TextResourceContents):
        try:
            manifest_data = json.loads(content.text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid manifest JSON for skill: {skill_name}") from e
    else:
        raise ValueError(f"Unexpected manifest format for skill: {skill_name}")

    try:
        return SkillManifest(
            name=manifest_data["skill"],
            files=[
                SkillFile(path=f["path"], size=f["size"], hash=f["hash"])
                for f in manifest_data["files"]
            ],
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid manifest format for skill: {skill_name}") from e


async def download_skill(
    client: Client,
    skill_name: str,
    target_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download a skill and all its files to a local directory.

    Creates a subdirectory named after the skill containing all files.

    Args:
        client: Connected FastMCP client
        skill_name: Name of the skill to download
        target_dir: Directory where skill folder will be created
        overwrite: If True, overwrite existing skill directory. If False
            (default), raise FileExistsError if directory exists.

    Returns:
        Path to the downloaded skill directory

    Raises:
        ValueError: If skill cannot be found or downloaded
        FileExistsError: If skill directory exists and overwrite=False

    Example:
        ```python
        from fastmcp import Client
        from fastmcp.utilities.skills import download_skill

        async with Client("http://skills-server/mcp") as client:
            skill_path = await download_skill(
                client,
                "pdf-processing",
                "~/.claude/skills"
            )
            print(f"Downloaded to: {skill_path}")
        ```
    """
    target_dir = Path(target_dir).expanduser().resolve()
    skill_dir = (target_dir / skill_name).resolve()

    # Security: ensure skill_dir stays within target_dir
    if not skill_dir.is_relative_to(target_dir):
        raise ValueError(f"Skill name {skill_name!r} would escape the target directory")

    # Check if directory exists
    if skill_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Skill directory already exists: {skill_dir}. "
            "Use overwrite=True to replace."
        )

    # Get manifest to know what files to download
    manifest = await get_skill_manifest(client, skill_name)

    # Create skill directory
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Download each file
    for file_info in manifest.files:
        # Security: reject absolute paths and paths that escape skill_dir
        if Path(file_info.path).is_absolute():
            continue
        file_path = (skill_dir / file_info.path).resolve()
        if not file_path.is_relative_to(skill_dir):
            continue

        file_uri = f"skill://{skill_name}/{file_info.path}"
        result = await client.read_resource(file_uri)

        if not result:
            continue

        content = result[0]

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        if isinstance(content, mcp_types.TextResourceContents):
            file_path.write_text(content.text, encoding="utf-8")
        elif isinstance(content, mcp_types.BlobResourceContents):
            file_path.write_bytes(base64.b64decode(content.blob))
        else:
            # Skip unknown content types
            continue

    return skill_dir


async def sync_skills(
    client: Client,
    target_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Download all available skills from a server.

    Args:
        client: Connected FastMCP client
        target_dir: Directory where skill folders will be created
        overwrite: If True, overwrite existing files

    Returns:
        List of paths to downloaded skill directories

    Example:
        ```python
        from fastmcp import Client
        from fastmcp.utilities.skills import sync_skills

        async with Client("http://skills-server/mcp") as client:
            paths = await sync_skills(client, "~/.claude/skills")
            print(f"Downloaded {len(paths)} skills")
        ```
    """
    skills = await list_skills(client)
    downloaded = []

    for skill in skills:
        try:
            path = await download_skill(
                client, skill.name, target_dir, overwrite=overwrite
            )
            downloaded.append(path)
        except FileExistsError:
            # Skip existing skills when not overwriting
            continue

    return downloaded
