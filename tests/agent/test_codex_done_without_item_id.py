"""Regression: reconciling a settled Codex ``function_call`` when ``id`` is absent.

Observed live on Copilot-routed ``gpt-5.6-sol`` (2026-08-24, hermes-dev
session ``20260824_173438_aa3352``).

``pending_function_calls`` is keyed by the ``response.output_item.added`` item
id. The reconciliation at ``response.output_item.done`` resolved that key from
the ``.done`` item's own ``id``. A backend is free to omit ``id`` on the
``.done`` frame — when it does, nothing is popped and the pending settle block
re-emits the SAME call a second time.

That phantom twin is a distinct ``(name, arguments)`` pair, so it survives
``_deduplicate_tool_calls``, executes, and fails. Three good tool calls become
three failures, which trips the tool-loop guardrail and kills the turn with
``repeated_exact_failure_block`` — the operator-visible symptom, while the real
commands ran fine.

Two properties must hold together, and testing only the first is how the
ordering half of this defect was originally missed:

* the call is emitted exactly ONCE (no duplicate), and
* it keeps its ANNOUNCED stream position (no reordering against siblings that
  are still pending).
"""

from types import SimpleNamespace

from agent.codex_runtime import _consume_codex_event_stream


def _added(item_id, call_id, name, output_index, arguments=""):
    item = SimpleNamespace(type="function_call", name=name, arguments=arguments)
    if item_id is not None:
        item.id = item_id
    if call_id is not None:
        item.call_id = call_id
    return SimpleNamespace(
        type="response.output_item.added",
        output_index=output_index,
        item=item,
    )


def _done(name, arguments, item_id=None, call_id=None, output_index=None):
    """A ``.done`` frame, with ``id`` / ``call_id`` / ``output_index`` optional."""
    item = SimpleNamespace(
        type="function_call", name=name, arguments=arguments, status="completed"
    )
    if item_id is not None:
        item.id = item_id
    if call_id is not None:
        item.call_id = call_id
    event = SimpleNamespace(type="response.output_item.done", item=item)
    if output_index is not None:
        event.output_index = output_index
    return event


def _completed():
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            id="resp_1",
            status="completed",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
            output=None,
        ),
    )


def _function_calls(final):
    return [
        item
        for item in (final.output or [])
        if getattr(item, "type", None) == "function_call"
    ]


def _run(events):
    return _function_calls(
        _consume_codex_event_stream(events, model="gpt-test")
    )


REAL_ARGS = '{"command": "hostname"}'


# ── single call: every shape of a .done frame missing its id ───────────


def test_done_without_item_id_emits_the_call_exactly_once():
    """The production shape: ``.done`` carries ``call_id`` but no ``id``."""
    calls = _run([
        _added("fc_1", "call_1", "terminal", 0),
        _done("terminal", REAL_ARGS, call_id="call_1", output_index=0),
        _completed(),
    ])

    assert len(calls) == 1, (
        "the call was emitted twice — a `.done` frame lacking `id` failed to "
        "clear its pending entry, so the settle block re-emitted it: "
        f"{[getattr(c, 'arguments', None) for c in calls]}"
    )
    assert getattr(calls[0], "arguments", None) == REAL_ARGS
    assert getattr(calls[0], "name", None) == "terminal"


def test_done_without_id_or_call_id_emits_the_call_exactly_once():
    """Neither identifier present — ``output_index`` is the remaining handle."""
    calls = _run([
        _added("fc_1", "call_1", "terminal", 0),
        _done("terminal", REAL_ARGS, output_index=0),
        _completed(),
    ])

    assert len(calls) == 1, (
        "a `.done` frame with neither `id` nor `call_id` duplicated its call: "
        f"{[getattr(c, 'arguments', None) for c in calls]}"
    )
    assert getattr(calls[0], "arguments", None) == REAL_ARGS


def test_announced_item_without_call_id_emits_the_call_exactly_once():
    """The ANNOUNCED item lacks ``call_id``, so no pending entry can match on it."""
    calls = _run([
        _added("fc_1", None, "terminal", 0),
        _done("terminal", REAL_ARGS, call_id="call_1", output_index=0),
        _completed(),
    ])

    assert len(calls) == 1, (
        "an announced item without `call_id` duplicated its call: "
        f"{[getattr(c, 'arguments', None) for c in calls]}"
    )
    assert getattr(calls[0], "arguments", None) == REAL_ARGS


# ── ordering: the half a single-call test cannot see ───────────────────


def test_settled_call_keeps_its_announced_position_against_a_pending_sibling():
    """A settled call must not jump behind a sibling that is still pending.

    ``AAA`` is announced first and confirmed by a ``.done`` frame carrying no
    ``id`` and no ``output_index``; ``BBB`` is announced second and only ever
    settles from pending state. Resolving the pending key and the stream
    position from different sources hands ``AAA`` a fresh tail sequence, so it
    is emitted AFTER ``BBB`` — inverting the side effects of dependent calls
    (``cd`` then ``run``, write then read).
    """
    calls = _run([
        _added("fc_1", "call_1", "AAA", None),
        _added("fc_2", "call_2", "BBB", None),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_2",
            delta='{"b": 2}',
        ),
        _done("AAA", '{"a": 1}', call_id="call_1"),
        _completed(),
    ])

    assert len(calls) == 2, (
        f"expected exactly two calls, got {[getattr(c, 'name', None) for c in calls]}"
    )
    assert [getattr(c, "name", None) for c in calls] == ["AAA", "BBB"], (
        "the settled call lost its announced position and was reordered "
        "behind a still-pending sibling"
    )


# ── the settle path this must not weaken ──────────────────────────────


def test_a_call_with_no_done_frame_at_all_is_still_settled():
    """Backends that omit ``.done`` entirely must keep being rescued.

    This is the behavior the pending settle block exists for
    (anomalyco/opencode#37159); reconciliation must not re-drop it.
    """
    calls = _run([
        _added("fc_1", "call_1", "terminal", 0),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            delta=REAL_ARGS,
        ),
        _completed(),
    ])

    assert len(calls) == 1, "a call with no `.done` frame was dropped"
    assert getattr(calls[0], "arguments", None) == REAL_ARGS


def test_a_normal_done_frame_with_an_id_is_unaffected():
    """The ordinary path — ``id`` present — keeps working unchanged."""
    calls = _run([
        _added("fc_1", "call_1", "terminal", 0),
        _done("terminal", REAL_ARGS, item_id="fc_1", call_id="call_1", output_index=0),
        _completed(),
    ])

    assert len(calls) == 1
    assert getattr(calls[0], "arguments", None) == REAL_ARGS
