"""Two restores into the same fresh bank id must not merge into one bank."""

import asyncio
import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.transfer import TransferScope, export_bank
from hindsight_api.engine.transfer.importer import import_bank


async def _fact_texts(memory, bank_id: str, request_context) -> list[str]:
    units = await memory.list_memory_units(bank_id, request_context=request_context)
    return sorted(item["text"] for item in units["items"] if item["fact_type"] != "observation")


async def _wait_until_blocked_by(memory, backend_pid: int) -> None:
    """Wait until another backend is waiting on a lock held by ``backend_pid``."""
    async with acquire_with_retry(await memory._get_backend()) as conn:
        for _ in range(200):
            if await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE $1 = ANY(pg_blocking_pids(pid))", backend_pid
            ):
                return
            await asyncio.sleep(0.05)
    raise AssertionError("the second restore never blocked on the first one's bank row")


@pytest.mark.asyncio
@pytest.mark.parametrize("bank_config", [True, False], ids=["with-config", "without-config"])
async def test_a_second_restore_racing_into_the_same_fresh_bank_is_refused(memory, bank_config):
    """Both restores pass the "target exists" check while neither has committed. The second
    one's bank insert then waits for the first; it must refuse, not pour a second copy of
    every fact into the bank the first one created."""
    ctx = RequestContext()
    source = f"race-src-{uuid.uuid4().hex[:8]}"
    target = f"race-dst-{uuid.uuid4().hex[:8]}"
    await memory.retain_async(
        bank_id=source,
        content="Alice joined Acme in 2024.",
        context="Test context",
        document_id="doc",
        request_context=ctx,
    )
    facts = await _fact_texts(memory, source, ctx)
    assert facts
    async with acquire_with_retry(await memory._get_backend()) as conn:
        archive = await export_bank(conn, source)
    scope = TransferScope(bank_config=bank_config)

    second: asyncio.Task | None = None

    async def start_the_second_restore(conn) -> None:
        # Still inside the first restore's transaction: its bank row is written but not
        # committed, so the second restore finds no bank and goes on to insert its own.
        nonlocal second
        second = asyncio.create_task(memory.import_bank_async(archive, ctx, target_bank_id=target, scope=scope))
        await _wait_until_blocked_by(memory, await conn.fetchval("SELECT pg_backend_pid()"))

    await import_bank(
        backend=await memory._get_backend(),
        embeddings_model=memory.embeddings,
        entity_resolver=memory.entity_resolver,
        resolve_config=lambda: memory._config_resolver.resolve_full_config(target, ctx),
        format_date_fn=memory._format_readable_date,
        archive=archive,
        target_bank_id=target,
        scope=scope,
        file_storage=memory._file_storage,
        on_bank_created=start_the_second_restore,
    )

    assert second is not None
    with pytest.raises(ValueError, match="created by another restore"):
        await second
    assert await _fact_texts(memory, target, ctx) == facts
