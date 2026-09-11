"""Skills-related methods for FastMCP Client (SEP-2640)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp.client.skills import send_get_skill, send_list_skills
from fastmcp.utilities.logging import get_logger
from fastmcp.utilities.skills import (
    SKILLS_EXTENSION_ID,
    GetSkillResult,
    ListSkillsResult,
    SkillEntry,
)

if TYPE_CHECKING:
    from fastmcp.client.client import Client

from fastmcp.client.telemetry import client_span

logger = get_logger(__name__)

AUTO_PAGINATION_MAX_PAGES = 250


class ClientSkillsMixin:
    """Mixin providing Skills-extension methods for Client (SEP-2640).

    Requires `Client(..., extensions=[SkillsClientExtension()])` -- these
    methods are never enabled implicitly, and raise a clear error if the
    negotiated server did not declare the extension.
    """

    def _require_skills_extension(self: Client, method: str) -> None:
        declared = any(
            getattr(ext, "identifier", None) == SKILLS_EXTENSION_ID
            for ext in (self._extensions_arg or ())
        )
        if not declared:
            raise RuntimeError(
                f"Cannot call {method}: this client was not constructed with "
                "SkillsClientExtension. Pass "
                "extensions=[SkillsClientExtension()] to Client() to opt in "
                "to the Skills extension (io.modelcontextprotocol/skills)."
            )

        capabilities = self.server_capabilities
        extensions = capabilities.extensions if capabilities else None
        if not extensions or SKILLS_EXTENSION_ID not in extensions:
            raise RuntimeError(
                f"Cannot call {method}: the connected server did not declare "
                f"the Skills extension ({SKILLS_EXTENSION_ID}). The server "
                "must register fastmcp.server.extensions.skills.SkillsExtension."
            )

    async def list_skills_mcp(
        self: Client,
        *,
        cursor: str | None = None,
    ) -> ListSkillsResult:
        """Send a `skills/list` request and return the complete protocol result.

        Args:
            cursor: Optional pagination cursor from a previous request's
                `next_cursor`.

        Returns:
            The complete `ListSkillsResult`, including `skills` and, when more
            pages remain, `next_cursor`.

        Raises:
            RuntimeError: If the connected server did not declare the Skills
                extension.
            MCPError: If the request results in a protocol error.
        """
        self._require_skills_extension("list_skills_mcp")
        with client_span(
            "skills/list",
            "skills/list",
            "",
            session_id=self.transport.get_session_id(),
        ):
            return await self._await_with_session_monitoring(
                send_list_skills(self.session, cursor)
            )

    async def list_skills(
        self: Client,
        max_pages: int = AUTO_PAGINATION_MAX_PAGES,
    ) -> list[SkillEntry]:
        """Retrieve every skill entry the server serves, following pagination.

        For manual pagination control, use `list_skills_mcp()` with `cursor`.

        Args:
            max_pages: Maximum number of pages to fetch before raising.

        Raises:
            RuntimeError: If the connected server did not declare the Skills
                extension, or if `max_pages` is exhausted before pagination
                completes.
            MCPError: If a request results in a protocol error.
        """
        all_skills: list[SkillEntry] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()

        for _ in range(max_pages):
            result = await self.list_skills_mcp(cursor=cursor)
            all_skills.extend(result.skills)
            if not result.next_cursor:
                break
            if result.next_cursor in seen_cursors:
                logger.warning(
                    f"[{self.name}] Server returned duplicate pagination cursor"
                    f" {result.next_cursor!r} for list_skills; stopping pagination"
                )
                break
            seen_cursors.add(result.next_cursor)
            cursor = result.next_cursor
        else:
            raise RuntimeError(
                f"[{self.name}] Reached auto-pagination limit"
                f" ({max_pages} pages) for list_skills."
                " Use list_skills_mcp() with cursor for manual pagination,"
                " or increase max_pages."
            )

        return all_skills

    async def get_skill(self: Client, uri: str) -> SkillEntry:
        """Get the entry for a single skill by the URI of its `SKILL.md`.

        `skills/get` is authoritative: it answers for any skill the server
        serves whether or not that skill appeared in `list_skills()`.

        Raises:
            RuntimeError: If the connected server did not declare the Skills
                extension.
            MCPError: If `uri` does not identify a skill the server serves
                (unknown, invalid, and unauthorized URIs all report the same
                `-32602` error).
        """
        self._require_skills_extension("get_skill")
        with client_span(
            "skills/get",
            "skills/get",
            uri,
            session_id=self.transport.get_session_id(),
        ):
            result: GetSkillResult = await self._await_with_session_monitoring(
                send_get_skill(self.session, uri)
            )
            return result.skill
