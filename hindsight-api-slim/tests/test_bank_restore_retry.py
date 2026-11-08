"""A retried whole-bank restore: what its crash-retry marker may delete, and when it stops retrying.

The two follow-ups #6408 left open:

- The crash-retry marker stored only the bank id. A restore that crashed, whose leftovers
  were then deleted and the id taken by another bank before the retry ran, deleted that
  other bank as its own half-restored one. The marker now also carries the bank row's
  ``internal_id``, and the retry deletes only a bank that still has it.
- A restore's refusals (the target exists, another restore created it first, the archive
  has no bank row) are the same on every run, but they were plain ``ValueError``s, which
  the worker retried before failing the operation. They now fail it at once.
"""

import json
import uuid
from typing import Any

import pytest

from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memory_engine import _is_non_retryable_task_error
from hindsight_api.engine.schema import fq_table
from hindsight_api.engine.transfer import TransferScope, export_bank, importer
from hindsight_api.worker.exceptions import RetryTaskAt


def _bank(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def _documents(memory, bank_id: str, request_context) -> list[str]:
    docs = await memory.list_documents(bank_id, request_context=request_context)
    return sorted(d["id"] for d in docs["items"])


async def _crash_a_restore(
    memory, monkeypatch, request_context, src: str, dst: str, *, bank_config: bool = True
) -> tuple[bytes, dict[str, Any]]:
    """Run a restore of ``src`` into ``dst`` through the task layer and crash it after the
    first document batch committed. Returns the archive and the task the worker would retry."""
    await memory.retain_async(
        bank_id=src, content="Alice works at Google.", document_id="doc-1", request_context=request_context
    )
    await memory.retain_async(
        bank_id=src, content="Carol lives in Paris.", document_id="doc-2", request_context=request_context
    )
    async with acquire_with_retry(await memory._get_backend()) as conn:
        archive = await export_bank(conn, src)

    submitted: list[dict[str, Any]] = []
    real_submit = memory._task_backend.submit_task

    async def record_task(task_dict: dict[str, Any]) -> None:
        if task_dict.get("type") == "import_bank":
            submitted.append(dict(task_dict))
        await real_submit(task_dict)

    monkeypatch.setattr(memory._task_backend, "submit_task", record_task)
    monkeypatch.setattr(importer, "_DOCUMENT_BATCH_SIZE", 1)
    real_batch = importer._import_document_batch
    calls = 0

    async def crash_on_second_batch(**kwargs: Any) -> list[importer._ImportedFactBatch]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("worker died")
        return await real_batch(**kwargs)

    monkeypatch.setattr(importer, "_import_document_batch", crash_on_second_batch)
    with pytest.raises(RetryTaskAt, match="worker died"):
        await memory.submit_bank_import_async(
            src, archive, request_context, target_bank_id=dst, scope=TransferScope(bank_config=bank_config)
        )
    monkeypatch.setattr(importer, "_import_document_batch", real_batch)
    [task] = submitted
    # The crash left a partial bank behind, marked as this operation's own.
    assert await _documents(memory, dst, request_context) == ["doc-1"]
    return archive, task


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("bank_config", [True, False], ids=["with-config", "without-config"])
async def test_a_retry_leaves_alone_a_bank_that_took_the_id_after_the_crash(
    memory, request_context, monkeypatch, bank_config
):
    src = _bank("retry-src")
    dst = _bank("retry-dst")
    try:
        _archive, task = await _crash_a_restore(memory, monkeypatch, request_context, src, dst, bank_config=bank_config)

        # Before the retry runs, someone deletes the leftovers and the id is taken again.
        await memory.delete_bank(dst, request_context=request_context)
        await memory.retain_async(
            bank_id=dst, content="Dave lives in Rome.", document_id="dave", request_context=request_context
        )

        # The retry, as the worker runs it. It must not delete the bank that is there now,
        # and since that refusal repeats on every run, it fails the operation instead of
        # scheduling another retry.
        await memory.execute_task({**task, "_retry_count": 1})

        assert await _documents(memory, dst, request_context) == ["dave"]
        status = await memory.get_operation_status(src, task["operation_id"], request_context=request_context)
        assert status["status"] == "failed"
        assert "already exists" in (status.get("error_message") or "")
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_a_marker_without_the_internal_id_still_starts_over(memory, request_context, monkeypatch):
    """An operation that crashed before the upgrade carries only the bank id. Its retry keeps
    the old behaviour rather than refusing its own leftovers."""
    src = _bank("retry-src")
    dst = _bank("retry-dst")
    try:
        archive, task = await _crash_a_restore(memory, monkeypatch, request_context, src, dst)
        async with acquire_with_retry(await memory._get_backend()) as conn:
            await conn.execute(
                f"UPDATE {fq_table('async_operations')} SET result_metadata = $1::jsonb WHERE operation_id = $2",
                json.dumps({"restored_bank_id": dst}),
                uuid.UUID(task["operation_id"]),
            )

        result = await memory.import_bank_async(
            archive, request_context, target_bank_id=dst, operation_id=task["operation_id"]
        )

        assert result.documents_imported == 2
        assert await _documents(memory, dst, request_context) == ["doc-1", "doc-2"]
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_restore_refusals_are_not_retried(memory, request_context):
    """A refusal reflects the target's state, which a rerun finds unchanged. The other two
    refusals are checked where their own tests raise them (test_bank_restore_race.py)."""
    src = _bank("refuse-src")
    try:
        await memory.retain_async(
            bank_id=src, content="Alice works at Google.", document_id="doc", request_context=request_context
        )
        async with acquire_with_retry(await memory._get_backend()) as conn:
            archive = await export_bank(conn, src)

        # The target exists (here: the source itself, restored under its own id).
        with pytest.raises(ValueError, match="already exists") as exists:
            await memory.import_bank_async(archive, request_context)
        assert _is_non_retryable_task_error(exists.value)

        # A transient failure is still retried.
        assert not _is_non_retryable_task_error(RuntimeError("worker died"))
    finally:
        await memory.delete_bank(src, request_context=request_context)
