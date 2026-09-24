"""Client-side support for the SEP-2640 Skills extension (`io.modelcontextprotocol/skills`).

Unlike the tasks extension (folded into every `Client` automatically),
`SkillsClientExtension` is never registered implicitly: pass it explicitly to
opt in.

```python
from fastmcp import Client
from fastmcp.client.skills import SkillsClientExtension

async with Client(mcp, extensions=[SkillsClientExtension()]) as client:
    skills = await client.list_skills()
```
"""

from __future__ import annotations

import mcp_types
from mcp.client.extension import ClientExtension
from mcp.client.session import ClientSession

from fastmcp.utilities.skills import (
    SKILLS_EXTENSION_ID,
    GetSkillRequest,
    GetSkillRequestParams,
    GetSkillResult,
    ListSkillsRequest,
    ListSkillsResult,
)


class SkillsClientExtension(ClientExtension):
    """The client half of the SEP-2640 Skills extension.

    Advertises the extension identifier with no settings, which is all a
    client needs to tell the server it may serve `skills/list` and
    `skills/get` requests for this connection -- neither method claims a
    `tools/call` result or observes a server notification, so the base
    class's defaults for `claims()`/`notifications()` apply unchanged.
    """

    identifier = SKILLS_EXTENSION_ID


async def send_list_skills(
    session: ClientSession,
    cursor: str | None,
    read_timeout_seconds: float | None = None,
) -> ListSkillsResult:
    """Send `skills/list` and return the complete protocol result."""
    request = ListSkillsRequest(params=mcp_types.PaginatedRequestParams(cursor=cursor))
    return await session.send_request(
        request, ListSkillsResult, request_read_timeout_seconds=read_timeout_seconds
    )


async def send_get_skill(
    session: ClientSession,
    uri: str,
    read_timeout_seconds: float | None = None,
) -> GetSkillResult:
    """Send `skills/get` for `uri` and return the complete protocol result."""
    request = GetSkillRequest(params=GetSkillRequestParams(uri=uri))
    return await session.send_request(
        request, GetSkillResult, request_read_timeout_seconds=read_timeout_seconds
    )
