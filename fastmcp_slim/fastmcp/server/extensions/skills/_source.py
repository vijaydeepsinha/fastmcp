"""The private Skills source contract (SEP-2640, PR 1).

`SkillsExtension` does not read the filesystem, parse frontmatter, or maintain
its own registry of skills. It asks whichever `Provider` instances it was
given for skill entries through this contract, and applies the extension's own
visibility/authorization and pagination rules on top.

The contract is intentionally unexported: it exists so `SkillsExtension` and
its skill-serving providers agree on a shape, not as a public extension point.
`fastmcp.server.providers.skills.SkillProvider` and `SkillsDirectoryProvider`
implement it structurally (no inheritance required -- `_SkillsSource` is a
`Protocol`); FastMCP's own test suite implements it with an in-memory fake to
exercise `SkillsExtension` without a filesystem. No other provider is
supported in this release: mounted, transformed, namespaced, and proxied
providers do not implement this contract, so `SkillsExtension` rejects them at
registration (see `extension.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from fastmcp.utilities.skills import SkillEntry


@runtime_checkable
class _SkillsSource(Protocol):
    """Structural contract a provider must satisfy to back `SkillsExtension`.

    Entries are returned unfiltered for visibility/authorization -- the
    extension itself decides, per request, which entries the caller may see
    (see `SkillsExtension._visible_entries`), so a source need not know
    anything about sessions, auth, or middleware.
    """

    @property
    def main_file_name(self) -> str:
        """The configured main-file name for this source's skills.

        The extension only supports the default `"SKILL.md"`; a source
        configured with an alternate name fails registration rather than
        advertise a catalog whose entries don't match the extension's
        URI-identifies-a-`SKILL.md` contract.
        """

    async def list_skill_entries(self) -> Sequence[SkillEntry]:
        """Every skill entry this source can currently produce."""
        ...

    async def get_skill_entry(self, uri: str) -> SkillEntry | None:
        """The entry for `uri`, or `None` if this source doesn't serve it."""
        ...
