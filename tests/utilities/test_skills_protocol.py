"""Tests for the SEP-2640 Skills wire protocol models in `fastmcp.utilities.skills`."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from fastmcp.utilities.skills import (
    GetSkillRequest,
    GetSkillRequestParams,
    GetSkillResult,
    ListSkillsRequest,
    ListSkillsResult,
    SkillEntry,
    SkillFrontmatter,
    SkillResourceEntry,
    SkillsExtensionSettings,
)

_VALID_DIGEST = "sha256:" + "a" * 64


def _entry(
    *, resources: list[SkillResourceEntry] | Literal["dynamic"] | None = None
) -> SkillEntry:
    return SkillEntry(
        uri="skill://demo/SKILL.md",
        frontmatter=SkillFrontmatter(name="demo", description="A demo skill"),
        resources=(
            resources
            if resources is not None
            else [
                SkillResourceEntry(
                    uri="skill://demo/SKILL.md", digest=_VALID_DIGEST, size=10
                )
            ]
        ),
    )


class TestSkillFrontmatter:
    def test_requires_name_and_description(self):
        with pytest.raises(ValidationError):
            SkillFrontmatter.model_validate({"name": "demo"})

    def test_preserves_unknown_fields_verbatim(self):
        fm = SkillFrontmatter.model_validate(
            {
                "name": "demo",
                "description": "d",
                "license": "Apache-2.0",
                "metadata": {"version": "2.1.0"},
            }
        )
        assert fm.model_dump()["license"] == "Apache-2.0"
        assert fm.model_dump()["metadata"] == {"version": "2.1.0"}


class TestSkillResourceEntry:
    def test_valid_digest(self):
        entry = SkillResourceEntry(
            uri="skill://x/SKILL.md", digest=_VALID_DIGEST, size=1
        )
        assert entry.digest == _VALID_DIGEST

    @pytest.mark.parametrize(
        "bad_digest",
        [
            "sha256:short",
            "md5:" + "a" * 32,
            "sha256:" + "A" * 64,  # uppercase hex is invalid
            "not-a-digest",
        ],
    )
    def test_invalid_digest_rejected(self, bad_digest: str):
        with pytest.raises(ValidationError):
            SkillResourceEntry(uri="skill://x/SKILL.md", digest=bad_digest, size=1)

    def test_negative_size_rejected(self):
        with pytest.raises(ValidationError):
            SkillResourceEntry(uri="skill://x/SKILL.md", digest=_VALID_DIGEST, size=-1)


class TestSkillEntry:
    def test_resources_may_be_dynamic(self):
        entry = _entry(resources="dynamic")
        assert entry.resources == "dynamic"

    def test_resources_array_or_dynamic_only(self):
        with pytest.raises(ValidationError):
            SkillEntry.model_validate(
                {
                    "uri": "skill://demo/SKILL.md",
                    "frontmatter": {"name": "demo", "description": "d"},
                    "resources": "not-dynamic",
                }
            )

    def test_round_trips_through_json(self):
        entry = _entry()
        restored = SkillEntry.model_validate_json(entry.model_dump_json())
        assert restored == entry


class TestRequestParams:
    def test_get_skill_request_params_requires_uri(self):
        with pytest.raises(ValidationError):
            GetSkillRequestParams.model_validate({})

    def test_get_skill_request_wraps_params(self):
        request = GetSkillRequest(
            params=GetSkillRequestParams(uri="skill://x/SKILL.md")
        )
        assert request.method == "skills/get"
        assert request.params.uri == "skill://x/SKILL.md"

    def test_list_skills_request_default_method(self):
        request = ListSkillsRequest()
        assert request.method == "skills/list"


class TestResults:
    def test_list_skills_result_defaults_to_complete(self):
        result = ListSkillsResult(skills=[_entry()])
        assert result.result_type == "complete"
        assert result.next_cursor is None

    def test_get_skill_result_defaults_to_complete(self):
        result = GetSkillResult(skill=_entry())
        assert result.result_type == "complete"
        assert result.skill.uri == "skill://demo/SKILL.md"

    def test_list_skills_result_wire_shape(self):
        dumped = ListSkillsResult(skills=[_entry()]).model_dump(
            by_alias=True, exclude_none=True
        )
        assert dumped["resultType"] == "complete"
        assert "nextCursor" not in dumped
        assert dumped["skills"][0]["uri"] == "skill://demo/SKILL.md"


class TestSkillsExtensionSettings:
    def test_default_advertises_empty_object(self):
        assert SkillsExtensionSettings().to_wire() == {}

    def test_directory_read_uses_camel_case_alias(self):
        settings = SkillsExtensionSettings(directory_read=True)
        assert settings.to_wire() == {"directoryRead": True}
