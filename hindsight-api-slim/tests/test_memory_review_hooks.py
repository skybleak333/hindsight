"""The two memory review hooks: an extension can drop or retag what retain extracted and what
consolidation decided, before either is written."""

import uuid
from unittest.mock import patch

import pytest

from hindsight_api.engine.consolidation import consolidator
from hindsight_api.engine.consolidation.consolidator import _BatchLLMResult, _CreateAction, run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain.orchestrator import RetainFactFilter
from hindsight_api.engine.retain.types import CausalRelation, ExtractedFact
from hindsight_api.extensions import (
    ConsolidationActionsContext,
    MemoryReview,
    OperationValidatorExtension,
    RetainFactsContext,
    ValidationResult,
)
from tests.test_consolidation_temporal_merge import (  # noqa: F401 — fixture
    _retain_fact,
    observations_enabled,
)


class _ReviewingValidator(OperationValidatorExtension):
    """Keeps only memories mentioning ``keep``, and tags what it keeps ``reviewed``."""

    def __init__(self, keep: str) -> None:
        super().__init__({})
        self.keep = keep
        self.retain_calls: list[RetainFactsContext] = []
        self.consolidation_calls: list[ConsolidationActionsContext] = []

    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    def _review(self, text: str, tags: list[str]) -> MemoryReview:
        if self.keep not in text:
            return MemoryReview(keep=False)
        return MemoryReview(tags=[*tags, "reviewed"])

    async def filter_retain_facts(self, ctx: RetainFactsContext) -> list[MemoryReview] | None:
        self.retain_calls.append(ctx)
        return [self._review(f.text, f.tags) for f in ctx.facts]

    async def filter_consolidation_actions(self, ctx: ConsolidationActionsContext) -> list[MemoryReview] | None:
        self.consolidation_calls.append(ctx)
        return [self._review(a.text, a.tags) for a in ctx.actions]


def _fact(text: str, relations: list[CausalRelation] | None = None) -> ExtractedFact:
    return ExtractedFact(fact_text=text, fact_type="world", tags=["user:a"], causal_relations=relations or [])


@pytest.mark.asyncio
async def test_dropping_a_fact_renumbers_the_causal_links_of_the_rest(request_context):
    facts = [
        _fact("Bob was late."),
        _fact("Alice missed the train.", [CausalRelation(relation_type="caused_by", target_fact_index=2)]),
        _fact("Alice overslept."),
        _fact("Alice was fired.", [CausalRelation(relation_type="caused_by", target_fact_index=0)]),
    ]
    review = RetainFactFilter(validator=_ReviewingValidator("Alice"), bank_id="b", request_context=request_context)

    kept = await review.review(facts, "doc-1")

    assert [f.fact_text for f in kept] == ["Alice missed the train.", "Alice overslept.", "Alice was fired."]
    assert all(f.tags == ["user:a", "reviewed"] for f in kept)
    # "missed the train" pointed at fact 2, now fact 1; "fired" pointed at the dropped fact.
    assert [r.target_fact_index for r in kept[0].causal_relations] == [1]
    assert kept[2].causal_relations == []


@pytest.mark.asyncio
async def test_a_wrong_number_of_reviews_fails_loudly(request_context):
    class _Short(_ReviewingValidator):
        async def filter_retain_facts(self, ctx):
            return []

    review = RetainFactFilter(validator=_Short("x"), bank_id="b", request_context=request_context)
    with pytest.raises(ValueError, match="0 reviews for 1 items"):
        await review.review([_fact("Alice")], None)


@pytest.mark.asyncio
async def test_retain_stores_only_what_the_extension_kept(memory: MemoryEngine, request_context):
    validator = _ReviewingValidator("Alice")
    memory._operation_validator = validator
    bank_id = f"test-review-retain-{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_batch_async(
            bank_id,
            [{"content": "Alice lives in Paris. Bob enjoys hiking.", "document_id": "doc-review", "tags": ["user:a"]}],
            request_context=request_context,
        )
        units = (await memory.list_memory_units(bank_id, request_context=request_context))["items"]
        stored = [u for u in units if u["fact_type"] != "observation"]

        assert validator.retain_calls and all(c.document_id == "doc-review" for c in validator.retain_calls)
        assert stored, "the kept Alice fact must be stored"
        assert all("Alice" in u["text"] for u in stored)
        assert all(sorted(u["tags"]) == ["reviewed", "user:a"] for u in stored)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_consolidation_writes_only_what_the_extension_kept(
    memory: MemoryEngine, request_context, observations_enabled
):
    bank_id = f"test-review-consolidation-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    alice = await _retain_fact(memory, request_context, observations_enabled, bank_id, "Alice likes tea.", "Alice")
    bob = await _retain_fact(memory, request_context, observations_enabled, bank_id, "Bob likes coffee.", "Bob")
    # The helper stamps every fact but its own consolidated, Alice's included: put it back.
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_units SET consolidated_at = NULL WHERE bank_id = $1 AND id = $2", bank_id, alice
        )
    validator = _ReviewingValidator("Alice")
    memory._operation_validator = validator

    async def scripted_llm(**_kwargs) -> _BatchLLMResult:
        return _BatchLLMResult(
            creates=[
                _CreateAction(text="Alice prefers tea.", source_fact_ids=[str(alice)]),
                _CreateAction(text="Bob prefers coffee.", source_fact_ids=[str(bob)]),
            ]
        )

    try:
        with patch.object(consolidator, "_consolidate_batch_with_llm", scripted_llm):
            await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        observations = (
            await memory.list_memory_units(bank_id, fact_type="observation", request_context=request_context)
        )["items"]
        # How the job splits the facts into batches is not this test's business: assert on what
        # the hook was shown, not on how many calls it took.
        seen = {(a.kind, a.text) for c in validator.consolidation_calls for a in c.actions}
        assert seen == {("create", "Alice prefers tea."), ("create", "Bob prefers coffee.")}
        assert [o["text"] for o in observations] == ["Alice prefers tea."]
        assert "reviewed" in observations[0]["tags"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


class _Restricting(OperationValidatorExtension):
    """Moves a memory about pay out of the document's tags into ``hr`` alone."""

    def __init__(self) -> None:
        super().__init__({})

    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def filter_retain_facts(self, ctx: RetainFactsContext) -> list[MemoryReview] | None:
        return [MemoryReview(tags=["hr"]) if "salary" in f.text else MemoryReview() for f in ctx.facts]


# Two chunks under a 60-char chunk size, so a re-retain that edits only the second keeps the first
# one's memories as survivors.
_PAY = "Alice's salary is ninety thousand euros, paid monthly."
_HIKING = "Bob enjoys hiking in the hills on weekends with friends."
_CYCLING = "Bob enjoys cycling by the river on weekends with friends."


async def _units(memory: MemoryEngine, bank_id: str, request_context) -> list[dict]:
    units = (await memory.list_memory_units(bank_id, document_id="doc-pay", request_context=request_context))["items"]
    return [u for u in units if u["fact_type"] != "observation"]


async def _retain_pay(memory: MemoryEngine, bank_id: str, request_context, content: str, **item) -> None:
    await memory.retain_batch_async(
        bank_id,
        [{"content": content, "document_id": "doc-pay", "tags": ["public"], **item}],
        request_context=request_context,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("again", "item"),
    [
        (_PAY + "\n\n" + _HIKING, {}),
        (_PAY + "\n\n" + _CYCLING, {}),
        (_CYCLING, {"update_mode": "append"}),
    ],
    ids=["same-content", "one-chunk-edited", "append"],
)
async def test_retaining_the_document_again_keeps_the_tags_the_extension_gave(
    memory: MemoryEngine, request_context, again, item
):
    """The memories of a re-retain's unchanged chunks are not shown to the hook again, so the
    re-retain must not undo what the hook decided for them the first time."""
    memory._operation_validator = _Restricting()
    bank_id = f"test-review-reretain-{uuid.uuid4().hex[:8]}"
    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        await memory._config_resolver.update_bank_config(bank_id, {"retain_chunk_size": 60})
        first = _PAY if item else _PAY + "\n\n" + _HIKING
        await _retain_pay(memory, bank_id, request_context, first)
        units = await _units(memory, bank_id, request_context)
        pay = {u["id"]: sorted(u["tags"]) for u in units if "salary" in u["text"]}
        assert pay and all(tags == ["hr"] for tags in pay.values()), pay

        await _retain_pay(memory, bank_id, request_context, again, **item)

        after = {u["id"]: (u["text"], sorted(u["tags"])) for u in await _units(memory, bank_id, request_context)}
        # The same units (their chunk was not extracted again), with the tags the hook gave them,
        # and every other memory on the document's tags.
        assert {uid: after[uid][1] for uid in pay if uid in after} == pay, after
        assert all(tags == ["public"] for uid, (text, tags) in after.items() if uid not in pay), after
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
