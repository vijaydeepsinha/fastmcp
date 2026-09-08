"""`SkillsExtension`: the server-side adapter for SEP-2640 (`io.modelcontextprotocol/skills`).

Registering this extension does not create a new component type or a second
registry of skills. It reads entries from the `SkillProvider` /
`SkillsDirectoryProvider` instances it is given (through the private
`_SkillsSource` contract in `_source.py`) and answers `skills/list` and
`skills/get` from them, applying the same session visibility and component
authorization decisions FastMCP already applies to `resources/list` and
`resources/read` -- by calling `FastMCP.get_resource()` per candidate skill,
which is inherently middleware-free at the single-resource granularity, rather
than by re-dispatching the middleware chain.

```python
from fastmcp import FastMCP
from fastmcp.server.extensions.skills import SkillsExtension
from fastmcp.server.providers.skills import SkillsDirectoryProvider

mcp = FastMCP("Skills")
skills = SkillsDirectoryProvider("./skills")

mcp.add_provider(skills)
mcp.add_extension(SkillsExtension(providers=[skills]))
```

Only directly registered `SkillProvider`/`SkillsDirectoryProvider` instances
are supported: a provider that is mounted, transformed, namespaced, or
proxied does not implement the private source contract by the time it reaches
`add_provider` (wrapping produces a different, unrelated object), and a
provider that is not `is`-present on `server.providers` was never actually
registered on this server. Both cases fail extension registration rather than
advertise an incomplete catalog. The `resources` capability is unaffected --
this extension only adds discovery metadata and methods on top of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from mcp.server.context import ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS, MISSING_REQUIRED_CLIENT_CAPABILITY
from mcp_types import PaginatedRequestParams as ListSkillsRequestParams
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from fastmcp.server.extensions import (
    MethodBinding,
    ServerExtension,
    read_client_extension_settings,
)
from fastmcp.server.mixins.mcp_operations import _apply_pagination
from fastmcp.server.providers.base import Provider
from fastmcp.utilities.logging import get_logger
from fastmcp.utilities.skills import (
    SKILLS_EXTENSION_ID,
    GetSkillRequestParams,
    GetSkillResult,
    ListSkillsResult,
    SkillEntry,
    SkillsExtensionSettings,
)

from ._source import _SkillsSource

if TYPE_CHECKING:
    from fastmcp.server.server import FastMCP

logger = get_logger(__name__)

#: `skills/list` and `skills/get` exist only at the era the extensions
#: mechanism itself requires (SEP-2133's per-request envelope).
_SKILLS_METHOD_VERSIONS = frozenset(MODERN_PROTOCOL_VERSIONS)


def _missing_capability_error(method: str) -> MCPError:
    return MCPError(
        code=MISSING_REQUIRED_CLIENT_CAPABILITY,
        message=(
            f"{method} targets the Skills extension ({SKILLS_EXTENSION_ID}); "
            "the client did not declare it for this request."
        ),
    )


class SkillsExtension(ServerExtension):
    """FastMCP server extension implementing the SEP-2640 Skills extension.

    Construct with the skill-serving providers to expose. Every provider must
    already be registered on the server (via `add_provider`) before this
    extension is registered, must be a directly registered `SkillProvider` or
    `SkillsDirectoryProvider` (not mounted, transformed, namespaced, or
    proxied), and must use the default `SKILL.md` main-file name.
    """

    identifier = SKILLS_EXTENSION_ID

    def __init__(self, *, providers: Sequence[Provider]) -> None:
        if not providers:
            raise ValueError("SkillsExtension requires at least one provider.")
        self._providers: list[Provider] = list(providers)
        self._sources: list[_SkillsSource] = []

    def _bind(self, server: FastMCP) -> None:
        super()._bind(server)
        sources: list[_SkillsSource] = []
        for provider in self._providers:
            if not isinstance(provider, _SkillsSource):
                raise ValueError(
                    f"SkillsExtension does not support {provider!r}: only "
                    "directly registered SkillProvider or SkillsDirectoryProvider "
                    "instances implement the Skills source contract. Mounted, "
                    "transformed, namespaced, and proxied providers are not "
                    "supported in this release."
                )
            if provider not in server.providers:
                raise ValueError(
                    f"SkillsExtension does not support {provider!r}: it is not "
                    "directly registered on this server. Register it with "
                    "server.add_provider(...) before registering SkillsExtension."
                )
            if provider.main_file_name != "SKILL.md":
                raise ValueError(
                    f"SkillsExtension does not support {provider!r}: it is "
                    f"configured with main_file_name={provider.main_file_name!r}. "
                    "The Skills extension identifies every skill by the URI of "
                    "its SKILL.md, so alternate main-file names are not "
                    "extension-compatible."
                )
            sources.append(provider)
        self._sources = sources

    def settings(self) -> dict[str, Any]:
        """Advertise the extension with no optional features (PR 1 does not
        implement `resources/directory/read`)."""
        return SkillsExtensionSettings().to_wire()

    def methods(self) -> Sequence[MethodBinding]:
        return [
            MethodBinding(
                method="skills/list",
                params_type=ListSkillsRequestParams,
                handler=self._handle_list,
                protocol_versions=_SKILLS_METHOD_VERSIONS,
            ),
            MethodBinding(
                method="skills/get",
                params_type=GetSkillRequestParams,
                handler=self._handle_get,
                protocol_versions=_SKILLS_METHOD_VERSIONS,
            ),
        ]

    def _require_opt_in(self, ctx: ServerRequestContext[Any, Any], method: str) -> None:
        if read_client_extension_settings(ctx, SKILLS_EXTENSION_ID) is None:
            raise _missing_capability_error(method)

    async def _is_entry_visible(self, entry: SkillEntry) -> bool:
        """Whether the caller may see `entry`, using the same session
        visibility and component authorization decisions Resources uses.

        Checking the entry's main `SKILL.md` resource is sufficient: providers
        in this release back every skill through a concrete `Resource` for the
        main file (see `SkillProvider._list_resources`), and `get_resource`
        already applies session-transform visibility and auth checks without
        dispatching the middleware chain (unlike `list_resources(run_middleware=True)`).
        """
        return await self.server.get_resource(entry.uri) is not None

    async def _all_entries(self) -> list[SkillEntry]:
        entries: list[SkillEntry] = []
        for source in self._sources:
            entries.extend(await source.list_skill_entries())
        return entries

    async def _handle_list(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: ListSkillsRequestParams | None,
    ) -> ListSkillsResult:
        self._require_opt_in(ctx, "skills/list")

        entries = sorted(await self._all_entries(), key=lambda e: e.uri)
        visible = [e for e in entries if await self._is_entry_visible(e)]

        cursor = params.cursor if params else None
        page, next_cursor = _apply_pagination(
            visible, cursor, self.server._list_page_size
        )
        return ListSkillsResult(skills=page, next_cursor=next_cursor)

    async def _handle_get(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: GetSkillRequestParams,
    ) -> GetSkillResult:
        self._require_opt_in(ctx, "skills/get")

        entry: SkillEntry | None = None
        for source in self._sources:
            entry = await source.get_skill_entry(params.uri)
            if entry is not None:
                break

        # Unknown, invalid, and unauthorized URIs are all reported the same
        # way (SEP-2640 Error Handling): authorization failures must not be
        # distinguishable from a plain unknown skill.
        if entry is None or not await self._is_entry_visible(entry):
            raise MCPError(
                code=INVALID_PARAMS,
                message=f"No skill is served at {params.uri!r}",
            )
        return GetSkillResult(skill=entry)
