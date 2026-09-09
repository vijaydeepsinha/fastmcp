"""Tests for the SEP-2640 `SkillsExtension` server adapter (PR 1).

`InMemorySkillsSource` below is the in-memory Skills source used to prove
`SkillsExtension`'s behavior -- pagination, direct lookup, invalid parameters,
negotiation, visibility, and authorization -- independently of any
filesystem-backed provider. It implements the same private `_SkillsSource`
contract `SkillProvider`/`SkillsDirectoryProvider` implement, and is not part
of the public FastMCP API.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from mcp.server.context import ServerRequestContext
from mcp.server.session import ServerSession
from mcp.shared.exceptions import MCPError
from mcp_types import MISSING_REQUIRED_CLIENT_CAPABILITY
from pydantic import AnyUrl

from fastmcp import Client, FastMCP
from fastmcp.client.skills import SkillsClientExtension
from fastmcp.resources.base import Resource
from fastmcp.resources.types import TextResource
from fastmcp.server.extensions.skills import SkillsExtension
from fastmcp.server.providers.base import Provider
from fastmcp.server.providers.skills import SkillProvider, SkillsDirectoryProvider
from fastmcp.utilities.skills import (
    SKILLS_EXTENSION_ID,
    GetSkillRequestParams,
    SkillEntry,
    SkillFrontmatter,
    SkillResourceEntry,
    missing_capability_error_data,
)


def _entry(name: str, *, description: str = "A test skill") -> SkillEntry:
    uri = f"skill://{name}/SKILL.md"
    return SkillEntry(
        uri=uri,
        frontmatter=SkillFrontmatter(name=name, description=description),
        resources=[
            SkillResourceEntry(uri=uri, digest="sha256:" + "0" * 64, size=1),
        ],
    )


class InMemorySkillsSource(Provider):
    """Minimal `_SkillsSource` implementation backed by a dict, for testing
    `SkillsExtension` without a filesystem provider.

    Entries named in `denied` back a resource whose `auth` check always
    returns False, so `SkillsExtension` must omit them exactly as it would an
    unauthorized filesystem-backed skill.
    """

    def __init__(
        self, entries: Sequence[SkillEntry], *, denied: frozenset[str] = frozenset()
    ) -> None:
        super().__init__()
        self._entries = {e.uri: e for e in entries}
        self._denied = denied

    @property
    def main_file_name(self) -> str:
        return "SKILL.md"

    async def list_skill_entries(self) -> Sequence[SkillEntry]:
        return list(self._entries.values())

    async def get_skill_entry(self, uri: str) -> SkillEntry | None:
        return self._entries.get(uri)

    def _resource_for(self, uri: str) -> Resource | None:
        entry = self._entries.get(uri)
        if entry is None:
            return None
        auth = (lambda ctx: False) if uri in self._denied else None
        return TextResource(
            uri=AnyUrl(uri), name=entry.frontmatter.name, text="", auth=auth
        )

    async def _list_resources(self) -> Sequence[Resource]:
        resources = [self._resource_for(uri) for uri in self._entries]
        return [r for r in resources if r is not None]

    async def _get_resource(self, uri: str, version=None) -> Resource | None:
        return self._resource_for(uri)


def _skills_client(mcp: FastMCP, **kwargs) -> Client:
    return Client(mcp, extensions=[SkillsClientExtension()], **kwargs)


class TestCapabilityAdvertisement:
    async def test_capability_advertised_when_registered(self):
        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("a")])
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            extensions = client.server_capabilities.extensions or {}
            assert extensions.get(SKILLS_EXTENSION_ID) == {}

    async def test_capability_absent_without_extension(self):
        mcp = FastMCP("t")

        async with _skills_client(mcp) as client:
            extensions = client.server_capabilities.extensions or {}
            assert SKILLS_EXTENSION_ID not in extensions

    async def test_capability_absent_on_legacy_protocol(self):
        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("a")])
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp, mode="legacy") as client:
            assert client.server_capabilities.extensions is None


class TestRegistration:
    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError, match="at least one provider"):
            SkillsExtension(providers=[])

    def test_rejects_unregistered_provider(self):
        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("a")])
        # Not registered via mcp.add_provider().
        with pytest.raises(ValueError, match="not directly registered"):
            mcp.add_extension(SkillsExtension(providers=[source]))

    def test_rejects_namespaced_provider(self, tmp_path: Path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: d\n---\nBody")
        provider = SkillProvider(skill_dir)

        mcp = FastMCP("t")
        mcp.add_provider(provider, namespace="ns")
        with pytest.raises(ValueError, match="not directly registered"):
            mcp.add_extension(SkillsExtension(providers=[provider]))

    def test_rejects_mounted_providers_source(self):
        """A provider registered on a *mounted* child is not in the parent's
        own `providers` list, so it fails the same "directly registered"
        check as an unregistered provider."""
        child_source = InMemorySkillsSource([_entry("a")])
        child = FastMCP("child")
        child.add_provider(child_source)

        mcp = FastMCP("t")
        mcp.mount(child, namespace="child")

        with pytest.raises(ValueError, match="not directly registered"):
            mcp.add_extension(SkillsExtension(providers=[child_source]))

    def test_rejects_non_source_provider(self):
        class PlainProvider(Provider):
            pass

        mcp = FastMCP("t")
        plain = PlainProvider()
        mcp.add_provider(plain)
        with pytest.raises(ValueError, match="Skills source contract"):
            mcp.add_extension(SkillsExtension(providers=[plain]))

    def test_rejects_alternate_main_file_name(self, tmp_path: Path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "MAIN.md").write_text("---\ndescription: d\n---\nBody")
        provider = SkillProvider(skill_dir, main_file_name="MAIN.md")

        mcp = FastMCP("t")
        mcp.add_provider(provider)
        with pytest.raises(ValueError, match="main_file_name"):
            mcp.add_extension(SkillsExtension(providers=[provider]))


class TestProtocolVersionGating:
    async def test_methods_unavailable_on_legacy_protocol(self):
        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("a")])
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp, mode="legacy") as client:
            with pytest.raises(RuntimeError, match="did not declare"):
                await client.list_skills()


def _bare_request_context(method: str, params: dict) -> ServerRequestContext:
    """A request context with no Skills capability in its `_meta`.

    Used to exercise `SkillsExtension`'s server-side opt-in check directly,
    bypassing the client-side `SkillsClientExtension` guard so the actual
    wire-level -32021 rejection is proven, not just the client's own
    precondition.
    """
    return ServerRequestContext(
        session=cast(ServerSession, SimpleNamespace()),
        lifespan_context={},
        protocol_version="2026-07-28",
        method=method,
        params=params,
    )


class TestExtensionNegotiation:
    async def test_rejects_request_without_opt_in(self):
        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("a")])
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with Client(mcp) as client:
            with pytest.raises(RuntimeError, match="SkillsClientExtension"):
                await client.list_skills()

    async def test_list_without_opt_in_raises_missing_capability_server_side(self):
        """`skills/list` called without the declared capability gets -32021,
        proven against the extension's handler directly (not the client-side
        guard, which never reaches the server for this case)."""
        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("a")])
        mcp.add_provider(source)
        extension = SkillsExtension(providers=[source])
        mcp.add_extension(extension)

        srctx = _bare_request_context("skills/list", {})
        with pytest.raises(MCPError) as exc_info:
            await extension._handle_list(srctx, None)
        error = exc_info.value.error
        assert error.code == MISSING_REQUIRED_CLIENT_CAPABILITY
        assert error.data == missing_capability_error_data()

    async def test_get_without_opt_in_raises_missing_capability_server_side(self):
        """`skills/get` called without the declared capability gets -32021,
        with the same `data.requiredCapabilities` payload the tasks
        extension returns for its own -32021 error."""
        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("a")])
        mcp.add_provider(source)
        extension = SkillsExtension(providers=[source])
        mcp.add_extension(extension)

        uri = "skill://a/SKILL.md"
        srctx = _bare_request_context("skills/get", {"uri": uri})
        params = GetSkillRequestParams.model_validate({"uri": uri})
        with pytest.raises(MCPError) as exc_info:
            await extension._handle_get(srctx, params)
        error = exc_info.value.error
        assert error.code == MISSING_REQUIRED_CLIENT_CAPABILITY
        assert error.data == missing_capability_error_data()


class TestListSkills:
    async def test_lists_all_entries(self):
        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("b"), _entry("a"), _entry("c")])
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            result = await client.list_skills_mcp()
            assert result.result_type == "complete"
            assert result.next_cursor is None

    async def test_deterministic_uri_ordering(self):
        mcp = FastMCP("t")
        source = InMemorySkillsSource(
            [_entry("charlie"), _entry("alpha"), _entry("bravo")]
        )
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            skills = await client.list_skills()
            assert [s.uri for s in skills] == [
                "skill://alpha/SKILL.md",
                "skill://bravo/SKILL.md",
                "skill://charlie/SKILL.md",
            ]

    async def test_pagination_does_not_split_entries(self):
        mcp = FastMCP("t", list_page_size=2)
        entries = [_entry(f"skill-{i}") for i in range(5)]
        source = InMemorySkillsSource(entries)
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            first = await client.list_skills_mcp()
            assert len(first.skills) == 2
            assert first.next_cursor is not None

            second = await client.list_skills_mcp(cursor=first.next_cursor)
            assert len(second.skills) == 2
            assert second.next_cursor is not None

            third = await client.list_skills_mcp(cursor=second.next_cursor)
            assert len(third.skills) == 1
            assert third.next_cursor is None

            all_uris = [s.uri for s in first.skills + second.skills + third.skills]
            assert all_uris == sorted(all_uris)
            assert len(set(all_uris)) == 5

    async def test_auto_pagination_returns_every_skill(self):
        mcp = FastMCP("t", list_page_size=2)
        entries = [_entry(f"skill-{i}") for i in range(5)]
        source = InMemorySkillsSource(entries)
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            skills = await client.list_skills()
            assert len(skills) == 5

    async def test_invalid_cursor_is_rejected(self):
        from mcp.shared.exceptions import MCPError

        mcp = FastMCP("t", list_page_size=1)
        source = InMemorySkillsSource([_entry("a")])
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            with pytest.raises(MCPError) as exc_info:
                await client.list_skills_mcp(cursor="not-a-valid-cursor")
            assert exc_info.value.code == -32602

    async def test_denied_skill_omitted_from_listing(self):
        mcp = FastMCP("t")
        source = InMemorySkillsSource(
            [_entry("visible"), _entry("secret")],
            denied=frozenset({"skill://secret/SKILL.md"}),
        )
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            skills = await client.list_skills()
            assert [s.uri for s in skills] == ["skill://visible/SKILL.md"]

    async def test_disabled_skill_omitted_from_listing(self, tmp_path: Path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: d\n---\nBody")
        provider = SkillsDirectoryProvider(tmp_path)

        mcp = FastMCP("t")
        mcp.add_provider(provider)
        mcp.add_extension(SkillsExtension(providers=[provider]))
        mcp.disable(keys={"resource:skill://my-skill/SKILL.md@"})

        async with _skills_client(mcp) as client:
            skills = await client.list_skills()
            assert skills == []


class TestGetSkill:
    async def test_direct_lookup(self):
        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("a")])
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            skill = await client.get_skill("skill://a/SKILL.md")
            assert skill.uri == "skill://a/SKILL.md"
            assert skill.frontmatter.name == "a"

    async def test_authoritative_even_when_absent_from_listing(self):
        """`skills/get` answers for a skill omitted from `skills/list`."""

        class ListlessSource(InMemorySkillsSource):
            async def list_skill_entries(self):
                return []

        mcp = FastMCP("t")
        source = ListlessSource([_entry("a")])
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            assert await client.list_skills() == []
            skill = await client.get_skill("skill://a/SKILL.md")
            assert skill.uri == "skill://a/SKILL.md"

    async def test_unknown_uri_is_invalid_params(self):
        from mcp.shared.exceptions import MCPError

        mcp = FastMCP("t")
        source = InMemorySkillsSource([_entry("a")])
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            with pytest.raises(MCPError) as exc_info:
                await client.get_skill("skill://does-not-exist/SKILL.md")
            assert exc_info.value.code == -32602

    async def test_unauthorized_uri_reports_same_error_as_unknown(self):
        from mcp.shared.exceptions import MCPError

        mcp = FastMCP("t")
        source = InMemorySkillsSource(
            [_entry("secret")], denied=frozenset({"skill://secret/SKILL.md"})
        )
        mcp.add_provider(source)
        mcp.add_extension(SkillsExtension(providers=[source]))

        async with _skills_client(mcp) as client:
            with pytest.raises(MCPError) as exc_info:
                await client.get_skill("skill://secret/SKILL.md")
            assert exc_info.value.code == -32602
