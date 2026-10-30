"""`_internal_error` is the single place a handler's unhandled exception is mapped.

79 route handlers used to spell this out inline: 72 byte-identical copies of

    import traceback
    error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
    logger.error(f"Error in <route>: {error_detail}")
    raise HTTPException(status_code=500, detail=str(e))

plus 7 near-copies that logged the traceback without the message. Re-typing the
policy per route is how it drifted; these tests pin the behaviour the copies
shared so the shared version cannot quietly change it.
"""

import logging

import asyncpg
import pytest
from fastapi import HTTPException

from hindsight_api.api.http import _internal_error


def _raise_through(exc: Exception, where: str):
    """Call the helper from inside a real `except` block, as every call site does."""
    try:
        raise exc
    except Exception as e:
        return _internal_error(e, where)


def test_maps_to_500_with_the_message_as_detail():
    """The client gets the message, never the traceback."""
    result = _raise_through(RuntimeError("boom"), "GET /v1/x")
    assert isinstance(result, HTTPException)
    assert result.status_code == 500
    assert result.detail == "boom"
    assert "Traceback" not in str(result.detail)


def test_logs_the_route_the_message_and_the_traceback(caplog):
    with caplog.at_level(logging.ERROR, logger="hindsight_api.api.http"):
        _raise_through(ValueError("bad input"), "GET /v1/default/banks/b1/graph")
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    # Exactly the format the 72 inline copies produced.
    assert msg.startswith("Error in GET /v1/default/banks/b1/graph: bad input\n\nTraceback:\n")
    assert "ValueError: bad input" in msg, "the traceback itself must be in the log"


def test_returns_rather_than_raises():
    """Call sites read `raise _internal_error(...)`; the helper must not raise itself."""
    result = _raise_through(RuntimeError("x"), "GET /v1/x")
    assert isinstance(result, HTTPException)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError('relation "memory_units" does not exist'),
        ValueError(""),
        KeyError("missing"),
        Exception("multi\nline\nmessage"),
    ],
)
def test_detail_is_always_the_stringified_exception(exc):
    assert _raise_through(exc, "GET /v1/x").detail == str(exc)


@pytest.mark.parametrize(
    "driver_error",
    [
        asyncpg.exceptions.CharacterNotInRepertoireError('invalid byte sequence for encoding "UTF8": 0x00'),
        asyncpg.exceptions.UntranslatableCharacterError("unsupported Unicode escape sequence"),
    ],
)
def test_text_the_database_cannot_store_is_a_400(driver_error, caplog):
    """A NUL byte in a filter, an id or a name is the client's input, not a server fault."""
    with caplog.at_level(logging.WARNING, logger="hindsight_api.api.http"):
        result = _raise_through(driver_error, "GET /v1/default/banks/b1/memories/list")

    assert result.status_code == 400
    assert "NUL byte" in result.detail
    assert [r.levelno for r in caplog.records] == [logging.WARNING], "no ERROR traceback for client input"


def test_text_the_database_cannot_store_is_found_behind_an_engine_wrapper():
    """Recall wraps the driver error (`raise RuntimeError(...) from e`); the 400 must survive that."""
    try:
        try:
            raise asyncpg.exceptions.CharacterNotInRepertoireError('invalid byte sequence for encoding "UTF8": 0x00')
        except Exception as inner:
            raise RuntimeError("Failed to search memories (CharacterNotInRepertoireError)") from inner
    except Exception as e:
        result = _internal_error(e, "POST /v1/default/banks/b1/memories/recall")

    assert result.status_code == 400


def test_an_unpaired_surrogate_is_a_400():
    """asyncpg fails to encode "\\ud800" before the query is sent; the cause is a UnicodeEncodeError."""
    try:
        try:
            "a\ud800".encode("utf-8")
        except UnicodeEncodeError as encode_error:
            raise asyncpg.exceptions.DataError("invalid input for query argument $2") from encode_error
    except Exception as e:
        result = _internal_error(e, "POST /v1/default/banks/b1/directives")

    assert result.status_code == 400
    assert "unpaired surrogate" in result.detail


def test_other_encoding_errors_stay_500():
    """Only the surrogate case is client text; any other UnicodeEncodeError is a server bug."""
    try:
        "ü".encode("ascii")
    except UnicodeEncodeError as e:
        assert _internal_error(e, "GET /v1/x").status_code == 500


def test_other_database_errors_stay_500():
    result = _raise_through(asyncpg.exceptions.UndefinedTableError('relation "x" does not exist'), "GET /v1/x")
    assert result.status_code == 500


def test_handlers_with_their_own_catch_all_also_answer_400_for_unstorable_text():
    """Recall, retain and reflect keep their own catch-all and call _store_unavailable by hand.

    Each of them has to consult _unstorable_text too, or a NUL byte in, say, a recall tag
    filter still answers 500 while every route using _internal_error answers 400.
    """
    import ast
    import pathlib

    source = pathlib.Path(__file__).resolve().parent.parent / "hindsight_api" / "api" / "http.py"
    tree = ast.parse(source.read_text())

    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = "\n".join(ast.unparse(s) for s in node.body)
        if "_store_unavailable(e)" in body and "_unstorable_text(e)" not in body:
            missing.append(node.lineno)
    assert not missing, f"catch-alls at lines {missing} map store errors but not unstorable text"


def test_no_handler_still_builds_the_500_inline():
    """Structural guard over the whole module, not any one route.

    A reintroduced inline copy is invisible per-route — it behaves identically
    until someone changes the policy in the helper and one route ignores it.
    """
    import ast
    import pathlib

    source = pathlib.Path(__file__).resolve().parent.parent / "hindsight_api" / "api" / "http.py"
    tree = ast.parse(source.read_text())

    # Two handlers legitimately keep their own catch-all because they add
    # diagnostics the shared helper cannot know about. Named, not silently
    # skipped, so adding a third is a deliberate act:
    #   recall  — logs handler_duration in its own [RECALL ERROR] format
    #   retain  — maps MemoryDefenseAllBlockedError to 422 and logs an input summary
    EXEMPT_MARKERS = ("[RECALL ERROR]", "MemoryDefenseAllBlockedError")

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if handler.type is None or ast.unparse(handler.type) != "Exception":
                continue
            body = "\n".join(ast.unparse(s) for s in handler.body)
            if "HTTPException(status_code=500" not in body or "_internal_error" in body:
                continue
            if any(marker in body for marker in EXEMPT_MARKERS):
                continue
            offenders.append(handler.lineno)
    assert not offenders, f"http.py builds a 500 inline instead of via _internal_error at lines {offenders}"
