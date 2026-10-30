"""Regression tests for mental model ids and knowledge page max_tokens the API accepted but could not use.

CreateMentalModelRequest documented its id as "alphanumeric lowercase with hyphens" without enforcing
it. An id with "/" was stored but could not be read or deleted through the HTTP API afterwards (the
slash splits the URL path), and any id outside that form made GET /banks/{bank_id}/export answer 500
for the whole bank,
because BankTemplateMentalModel does enforce the rule. Knowledge pages are mental models, but
CreatePageRequest and UpdateNodeRequest left max_tokens unbounded, so a large value overflowed the INT
column with a 500.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from hindsight_api import RequestContext
from hindsight_api.api.http import CreateMentalModelRequest, CreatePageRequest, UpdateNodeRequest
from hindsight_api.engine.bank_templates import BankTemplateMentalModel
from hindsight_api.engine.memory_engine import MemoryEngine

BAD_IDS = ["team/standup", "has spaces", "Customer-Profile", "-leading-hyphen", "ünïcode", "trailing-newline\n"]
GOOD_IDS = ["team-communication", "mm-0b6f1a524f0b4b539a596a0a8c4c2f10", "7-day-summary"]


@pytest.mark.parametrize("bad", BAD_IDS)
def test_create_request_rejects_ids_the_templates_reject(bad):
    with pytest.raises(ValidationError, match="must be alphanumeric lowercase with hyphens"):
        CreateMentalModelRequest(id=bad, name="n", source_query="q")
    # The two models now agree, so a model created through the API can always be exported.
    with pytest.raises(ValidationError):
        BankTemplateMentalModel(id=bad, name="n", source_query="q")


@pytest.mark.parametrize("good", GOOD_IDS)
def test_create_request_accepts_ids_the_templates_accept(good):
    assert CreateMentalModelRequest(id=good, name="n", source_query="q").id == good
    assert BankTemplateMentalModel(id=good, name="n", source_query="q").id == good


def test_the_id_stays_optional():
    assert CreateMentalModelRequest(name="n", source_query="q").id is None
    # An empty id has always meant "generate one"; it must not turn into a 422.
    assert CreateMentalModelRequest(id="", name="n", source_query="q").id is None


def _stub_engine() -> MemoryEngine:
    engine = object.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None
    engine._get_backend = AsyncMock(return_value=MagicMock())
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        lambda e, ctx: e.create_mental_model(
            bank_id="b", name="n", source_query="q", content="", mental_model_id="team/standup", request_context=ctx
        ),
        lambda e, ctx: e.create_knowledge_page(
            bank_id="b", name="n", source_query="q", content="", mental_model_id="team/standup", request_context=ctx
        ),
    ],
    ids=["mental-model", "knowledge-page"],
)
async def test_the_engine_checks_custom_ids_for_callers_without_a_request_model(call):
    # The MCP tools pass a caller-chosen id straight to the engine.
    engine = _stub_engine()

    with pytest.raises(ValueError, match="must be alphanumeric lowercase with hyphens"):
        await call(engine, RequestContext(api_key="anything"))

    engine._authenticate_tenant.assert_not_awaited()


@pytest.mark.parametrize("model", [CreatePageRequest, UpdateNodeRequest])
@pytest.mark.parametrize("max_tokens", [0, 255, 8193, 2**40])
def test_page_max_tokens_outside_the_mental_model_range_is_rejected(model, max_tokens):
    with pytest.raises(ValidationError, match="max_tokens must be between 256 and 8192"):
        model(name="n", source_query="q", max_tokens=max_tokens)


@pytest.mark.parametrize("model", [CreatePageRequest, UpdateNodeRequest])
@pytest.mark.parametrize("max_tokens", [256, 4096, 8192, None])
def test_page_max_tokens_inside_the_range_is_accepted(model, max_tokens):
    assert model(name="n", source_query="q", max_tokens=max_tokens).max_tokens == max_tokens


@pytest.mark.asyncio
async def test_the_engine_checks_page_max_tokens_too():
    engine = _stub_engine()

    with pytest.raises(ValueError, match="max_tokens must be between"):
        await engine.create_knowledge_page(
            bank_id="b",
            name="n",
            source_query="q",
            content="",
            max_tokens=2**40,
            request_context=RequestContext(api_key="x"),
        )
