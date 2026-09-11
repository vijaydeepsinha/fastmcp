"""Tests for the SEP-2640 Skills extension client support (PR 1).

Exercises the typed `Client` methods (`list_skills_mcp`, `list_skills`,
`get_skill`) against a real filesystem-backed server (`SkillsDirectoryProvider`
+ `SkillsExtension`), plus explicit opt-in and protocol-version behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastmcp import Client, FastMCP
from fastmcp.client.skills import SkillsClientExtension
from fastmcp.server.extensions.skills import SkillsExtension
from fastmcp.server.providers.skills import SkillsDirectoryProvider


def _make_skill(root: Path, name: str, description: str = "A test skill") -> None:
    skill_dir = root / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody.\n"
    )
    (skill_dir / "reference.md").write_text("Extra reference content.")


def _skills_server(tmp_path: Path, *, list_page_size: int | None = None) -> FastMCP:
    _make_skill(tmp_path, "alpha")
    _make_skill(tmp_path, "bravo")
    _make_skill(tmp_path, "charlie")

    mcp = FastMCP("skills-server", list_page_size=list_page_size)
    provider = SkillsDirectoryProvider(tmp_path)
    mcp.add_provider(provider)
    mcp.add_extension(SkillsExtension(providers=[provider]))
    return mcp


@pytest.fixture
def skills_server(tmp_path: Path) -> FastMCP:
    return _skills_server(tmp_path)


class TestListSkills:
    async def test_list_skills_mcp_one_page(self, skills_server: FastMCP):
        async with Client(
            skills_server, extensions=[SkillsClientExtension()]
        ) as client:
            result = await client.list_skills_mcp()
            assert result.next_cursor is None
            uris = sorted(s.uri for s in result.skills)
            assert uris == [
                "skill://alpha/SKILL.md",
                "skill://bravo/SKILL.md",
                "skill://charlie/SKILL.md",
            ]
            alpha = next(s for s in result.skills if s.frontmatter.name == "alpha")
            assert isinstance(alpha.resources, list)
            assert any(r.uri == alpha.uri for r in alpha.resources)

    async def test_list_skills_auto_paginates(self, tmp_path: Path):
        server = _skills_server(tmp_path, list_page_size=1)
        async with Client(server, extensions=[SkillsClientExtension()]) as client:
            skills = await client.list_skills()
            assert sorted(s.uri for s in skills) == [
                "skill://alpha/SKILL.md",
                "skill://bravo/SKILL.md",
                "skill://charlie/SKILL.md",
            ]

    async def test_list_skills_respects_max_pages(self, tmp_path: Path):
        server = _skills_server(tmp_path, list_page_size=1)
        async with Client(server, extensions=[SkillsClientExtension()]) as client:
            with pytest.raises(RuntimeError, match="auto-pagination limit"):
                await client.list_skills(max_pages=1)


class TestGetSkill:
    async def test_get_skill_by_uri(self, skills_server: FastMCP):
        async with Client(
            skills_server, extensions=[SkillsClientExtension()]
        ) as client:
            skill = await client.get_skill("skill://alpha/SKILL.md")
            assert skill.frontmatter.name == "alpha"
            assert isinstance(skill.resources, list)
            assert len(skill.resources) == 2  # SKILL.md + reference.md

    async def test_get_skill_unknown_uri_raises(self, skills_server: FastMCP):
        from mcp.shared.exceptions import MCPError

        async with Client(
            skills_server, extensions=[SkillsClientExtension()]
        ) as client:
            with pytest.raises(MCPError) as exc_info:
                await client.get_skill("skill://does-not-exist/SKILL.md")
            assert exc_info.value.code == -32602


class TestExtensionNegotiation:
    async def test_fails_clearly_without_client_opt_in(self, skills_server: FastMCP):
        async with Client(skills_server) as client:
            with pytest.raises(RuntimeError, match="SkillsClientExtension"):
                await client.list_skills()

            with pytest.raises(RuntimeError, match="SkillsClientExtension"):
                await client.get_skill("skill://alpha/SKILL.md")

    async def test_fails_clearly_when_server_lacks_extension(self):
        mcp = FastMCP("plain-server")

        async with Client(mcp, extensions=[SkillsClientExtension()]) as client:
            with pytest.raises(RuntimeError, match="did not declare"):
                await client.list_skills()


class TestProtocolVersionBehavior:
    async def test_legacy_connection_has_no_extension_capability(
        self, skills_server: FastMCP
    ):
        async with Client(
            skills_server, extensions=[SkillsClientExtension()], mode="legacy"
        ) as client:
            assert client.server_capabilities.extensions is None
            with pytest.raises(RuntimeError, match="did not declare"):
                await client.list_skills()
