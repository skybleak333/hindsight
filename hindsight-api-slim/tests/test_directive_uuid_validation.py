"""Regression tests for malformed ids that reached the database and answered 500.

Same class of bug as test_get_entity_uuid_validation.py: get_memory_unit, get_entity and the
operation endpoints validate their id against uuid.UUID and the route answers 400, but the
directive endpoints and clear_observations_for_memory passed the raw string on. Postgres then
rejected it against the uuid column (or uuid.UUID raised a bare ValueError mid-transaction) and
the route mapped that to 500.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from hindsight_api import RequestContext
from hindsight_api.api.http import CreateDirectiveRequest, UpdateDirectiveRequest
from hindsight_api.engine.bank_templates import BankTemplateDirective
from hindsight_api.engine.memory_engine import MemoryEngine

BAD_UUID = "not-a-uuid"


def _stub_engine() -> MemoryEngine:
    engine = object.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None
    # Never reached: the id is rejected before any lookup.
    engine._get_backend = AsyncMock(return_value=MagicMock())
    engine._directive_writable = AsyncMock(return_value=True)
    engine._require_writable = AsyncMock()
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        lambda e, ctx: e.get_directive(bank_id="b", directive_id=BAD_UUID, request_context=ctx),
        lambda e, ctx: e.update_directive(bank_id="b", directive_id=BAD_UUID, name="x", request_context=ctx),
        lambda e, ctx: e.delete_directive(bank_id="b", directive_id=BAD_UUID, request_context=ctx),
    ],
    ids=["get", "update", "delete"],
)
async def test_directive_methods_reject_a_malformed_id(call):
    engine = _stub_engine()

    with pytest.raises(ValueError, match="Invalid directive_id: 'not-a-uuid' is not a valid UUID"):
        await call(engine, RequestContext(api_key="anything"))

    engine._get_backend.assert_not_awaited()
    engine._directive_writable.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_observations_for_memory_rejects_a_malformed_id():
    engine = _stub_engine()

    with pytest.raises(ValueError, match="Invalid memory_id: 'not-a-uuid' is not a valid UUID"):
        await engine.clear_observations_for_memory(
            bank_id="b", memory_id=BAD_UUID, request_context=RequestContext(api_key="anything")
        )

    engine._require_writable.assert_not_awaited()
    engine._get_backend.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_well_formed_directive_id_still_reaches_the_lookup():
    engine = _stub_engine()
    engine._get_backend = AsyncMock(side_effect=RuntimeError("reached the database"))

    with pytest.raises(RuntimeError, match="reached the database"):
        await engine.get_directive(
            bank_id="b",
            directive_id="0b6f1a52-4f0b-4b53-9a59-6a0a8c4c2f10",
            request_context=RequestContext(api_key="anything"),
        )


CANONICAL = "0b6f1a52-4f0b-4b53-9a59-6a0a8c4c2f10"


@pytest.mark.parametrize(
    "raw", ["{0b6f1a52-4f0b-4b53-9a59-6a0a8c4c2f10}", "urn:uuid:0b6f1a52-4f0b-4b53-9a59-6a0a8c4c2f10"]
)
def test_other_uuid_spellings_reach_the_query_in_canonical_form(raw):
    # uuid.UUID accepts these, the uuid column does not: passing the raw string on still 500ed.
    assert MemoryEngine._require_directive_uuid(raw) == CANONICAL


@pytest.mark.asyncio
async def test_update_and_delete_look_up_the_canonical_id():
    engine = _stub_engine()
    engine._directive_writable = AsyncMock(return_value=False)
    ctx = RequestContext(api_key="anything")

    await engine.update_directive(bank_id="b", directive_id="{" + CANONICAL + "}", name="x", request_context=ctx)
    await engine.delete_directive(bank_id="b", directive_id="urn:uuid:" + CANONICAL, request_context=ctx)

    assert [c.args[1] for c in engine._directive_writable.await_args_list] == [CANONICAL, CANONICAL]


@pytest.mark.asyncio
async def test_clear_observations_for_memory_uses_the_canonical_id():
    engine = _stub_engine()
    engine._require_writable = AsyncMock(side_effect=RuntimeError("stop here"))

    with pytest.raises(RuntimeError, match="stop here"):
        await engine.clear_observations_for_memory(
            bank_id="b", memory_id="{" + CANONICAL + "}", request_context=RequestContext(api_key="anything")
        )

    assert engine._require_writable.await_args.kwargs["memory_id"] == CANONICAL


@pytest.mark.parametrize("model", [CreateDirectiveRequest, UpdateDirectiveRequest, BankTemplateDirective])
@pytest.mark.parametrize("priority", [2**31, -(2**31) - 1])
def test_priority_outside_the_int_column_is_a_validation_error(model, priority):
    # directives.priority is INT; a larger value used to fail the insert with a 500.
    with pytest.raises(ValidationError, match="priority must be between"):
        model(name="n", content="c", priority=priority)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        lambda e, ctx: e.create_directive(bank_id="b", name="n", content="c", priority=2**31, request_context=ctx),
        lambda e, ctx: e.update_directive(bank_id="b", directive_id=CANONICAL, priority=2**31, request_context=ctx),
    ],
    ids=["create", "update"],
)
async def test_the_engine_checks_priority_for_callers_without_a_request_model(call):
    # The MCP tools call the engine directly.
    engine = _stub_engine()

    with pytest.raises(ValueError, match="priority must be between"):
        await call(engine, RequestContext(api_key="anything"))

    engine._get_backend.assert_not_awaited()


@pytest.mark.parametrize("model", [CreateDirectiveRequest, UpdateDirectiveRequest, BankTemplateDirective])
@pytest.mark.parametrize("priority", [2**31 - 1, -(2**31), 0])
def test_priority_inside_the_int_column_is_accepted(model, priority):
    assert model(name="n", content="c", priority=priority).priority == priority
