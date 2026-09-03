"""No handler in ``src/`` may catch everything and say nothing.

``except Exception: pass`` is the shape that turns a bug into a wrong number.
The exception carried the only evidence that the computation did not happen;
discarding it leaves a caller holding a partial result that is indistinguishable
from a complete one. Every diagnostic this package reports — a failed frontier
anchor, a dropped resampling draw, an unanswerable feasibility probe — exists
because someone downstream needed to know that something did not work.

Two scans run here, in increasing order of subtlety.

**A. The classic shape.** A broad handler (``except Exception``, ``except
BaseException``, or a bare ``except``) whose body is nothing but ``pass`` or
``continue``. The plan's audit counted five of these; §1.4 and §1.5 cleared
``frontier.py`` (two) and ``resampling.py`` (one), and Task 3.3 cleared
``feasibility.py`` (one). One survives, and it is on the allowlist with its
reason.

**B. The same defect with a return statement.** A handler whose body is a lone
``return`` is just as silent as one whose body is ``pass`` when the value it
returns says nothing about the failure — the caller gets a fallback it cannot
tell apart from a real answer. The discriminator used here is whether the
handler *surfaces the exception it caught*: a handler that binds ``as exc`` and
puts ``exc`` in the value it returns is reporting, not swallowing. That rule
separates the two cleanly on this tree, with no judgement calls — see
``_returns_the_exception``.

Both allowlists are keyed by file, not by line, so that ordinary edits above a
handler do not break this test while a genuinely new handler still does. Every
entry carries a reason, because an exemption without one is a TODO wearing a
decision's clothes. The bookkeeping is exact in both directions: an unlisted
site fails, and so does a listed site that no longer exists, so a fix cannot
leave a stale exemption behind for the next silent handler to hide under.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class Allowed(NamedTuple):
    """One permitted silent handler: where it is, and why it is allowed.

    Attributes:
        path: Path to the file, relative to ``src/``. One entry per handler,
            so a file with two permitted handlers needs two entries.
        reason: Why swallowing is the right behaviour *here*. Required, and
            asserted non-empty: the point of an allowlist is that every line
            of it is an argument someone made and can be argued with.
    """

    path: str
    reason: str


#: Handlers whose body is only ``pass`` or ``continue``.
#:
#: One entry. The audit found five; four were fixed rather than exempted.
PASS_ALLOWLIST: tuple[Allowed, ...] = (
    Allowed(
        "optimization_engine/optimizers/hrp.py",
        # The cluster-membership diagnostic is decoration on a solve that has
        # already succeeded: ``fcluster`` can refuse a degenerate linkage, and
        # failing the whole allocation because the pretty-printed tree could
        # not be built would be a worse answer than omitting the tree. The
        # weights are unaffected, and ``hrp_order``/``hrp_linkage`` are still
        # written below the handler, so the result does not silently lose its
        # entire clustering record. Carries ``# pragma: no cover``.
        "scipy's fcluster can refuse a degenerate linkage; the diagnostic is "
        "cosmetic and the solve it describes has already succeeded",
    ),
)

#: Handlers whose body is a lone ``return`` that never mentions the exception.
#:
#: A shorter list than A's, and a more argued one — returning a fallback is
#: sometimes exactly right. What is never right is returning it in a way the
#: caller cannot distinguish from success.
RETURN_ALLOWLIST: tuple[Allowed, ...] = (
    Allowed(
        "optimization_engine/optimizers/feasibility.py",
        # ``_is_lp_feasible`` is tri-state on purpose: ``True``/``False``
        # are verdicts and ``None`` means "could not be determined", which is a
        # distinct value the caller already branches on — it declines to name
        # the component as the culprit rather than pretending it cleared it.
        # The fallback is therefore *not* indistinguishable from an answer,
        # which is the property that makes a silent return a defect.
        "the probe is tri-state and returns None for 'unanswerable', which the "
        "caller reads as 'not proven guilty' rather than as a clean bill",
    ),
    Allowed(
        "optimization_engine/optimizers/factory.py",
        # KNOWN DEFECT, recorded rather than fixed: this agent owns no source
        # file. ``_expected_returns_for`` falls back to the prior mean when the
        # Black-Litterman posterior raises, so Task 1.6's new out-of-universe
        # and degenerate-view validation is swallowed on the preview path while
        # the real solve raises on the same config. The two disagree, silently,
        # which is exactly the failure mode this module exists to prevent.
        # Delete this entry when the handler learns to report; the stale-entry
        # check below will then insist on it.
        "KNOWN DEFECT (execution log item 8): swallows Black-Litterman's "
        "validation so the preview falls back where the real solve raises",
    ),
)


class Site(NamedTuple):
    """A silent handler the scan found."""

    path: str
    lineno: int
    body: str


def _is_broad(handler: ast.ExceptHandler) -> bool:
    """Whether ``handler`` catches everything.

    A bare ``except:`` catches ``BaseException``; ``except Exception`` and
    ``except BaseException`` are named equivalents. A tuple counts if any of
    its members is one of those, since the broad member subsumes the rest.
    """
    if handler.type is None:
        return True
    caught = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(ast.unparse(node) in {"Exception", "BaseException"} for node in caught)


def _returns_the_exception(handler: ast.ExceptHandler) -> bool:
    """Whether a single-``return`` handler puts the caught exception in its answer.

    This is the whole discriminator for scan B, so it is deliberately narrow:
    the handler must bind the exception with ``as`` *and* name that binding
    somewhere inside the returned expression. ``return None, f"{type(exc)}: {exc}"``
    passes; ``return expected_returns`` does not. Nothing here is a judgement
    about how good the report is — only that a report exists at all.
    """
    if handler.name is None:
        return False
    returned = handler.body[0]
    assert isinstance(returned, ast.Return)
    if returned.value is None:
        return False
    return any(
        isinstance(node, ast.Name) and node.id == handler.name
        for node in ast.walk(returned.value)
    )


def _scan() -> tuple[list[Site], list[Site]]:
    """Walk ``src/`` and return the (pass-bodied, silently-return-bodied) sites."""
    pass_sites: list[Site] = []
    return_sites: list[Site] = []
    for path in sorted(SRC.rglob("*.py")):
        relative = str(path.relative_to(SRC))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not _is_broad(node):
                continue
            body = node.body
            if all(isinstance(stmt, (ast.Pass, ast.Continue)) for stmt in body):
                pass_sites.append(Site(relative, node.lineno, ast.unparse(body[0])))
            elif len(body) == 1 and isinstance(body[0], ast.Return):
                if not _returns_the_exception(node):
                    return_sites.append(
                        Site(relative, node.lineno, ast.unparse(body[0]))
                    )
    return pass_sites, return_sites


def _report_new(sites: list[Site], allowlist: tuple[Allowed, ...], shape: str) -> str:
    """Explain what to do about handlers the allowlist does not cover."""
    listed = "\n".join(f"    {entry.path} — {entry.reason}" for entry in allowlist)
    found = "\n".join(f"    {s.path}:{s.lineno}  {s.body}" for s in sites)
    return (
        f"{len(sites)} handler(s) catch everything and {shape}, beyond what the "
        f"allowlist permits:\n{found}\n\n"
        "A caught exception is evidence. Do one of:\n"
        "  * re-raise it, if the caller cannot carry on without the result;\n"
        "  * record it — a diagnostics key, a `failures` entry, a returned "
        "error string — and carry on, which is what frontier.py, "
        "resampling.py and feasibility.py were changed to do; or\n"
        "  * narrow the `except` to the specific exception you expected, so "
        "the ones you did not expect still travel.\n"
        "If it genuinely belongs here, add it to the allowlist in this file "
        f"with a reason. Currently allowed:\n{listed or '    (nothing)'}"
    )


def _report_stale(stale: list[Allowed], shape: str) -> str:
    """Explain that an exemption has outlived the handler it excused."""
    entries = "\n".join(f"    {entry.path} — {entry.reason}" for entry in stale)
    return (
        f"{len(stale)} allowlist entry/entries no longer match any handler that "
        f"catches everything and {shape}:\n{entries}\n\n"
        "The handler was fixed or removed, so the exemption is now cover for "
        "the next one to appear in that file unnoticed. Delete the entry."
    )


def _check(sites: list[Site], allowlist: tuple[Allowed, ...], shape: str) -> None:
    """Match sites against the allowlist one-for-one, in both directions."""
    remaining = list(allowlist)
    unexcused: list[Site] = []
    for site in sites:
        match = next((e for e in remaining if e.path == site.path), None)
        if match is None:
            unexcused.append(site)
        else:
            remaining.remove(match)
    assert not unexcused, _report_new(unexcused, allowlist, shape)
    assert not remaining, _report_stale(remaining, shape)


def test_every_allowlist_entry_states_a_reason() -> None:
    """An exemption with no argument behind it is a TODO, not a decision."""
    for entry in PASS_ALLOWLIST + RETURN_ALLOWLIST:
        assert entry.reason.strip(), f"{entry.path} is exempted without a reason"
        assert (SRC / entry.path).is_file(), (
            f"{entry.path} is on the allowlist but does not exist under src/; "
            "the file was moved or renamed and the entry was not"
        )


def test_no_silently_swallowed_exceptions() -> None:
    """Scan A: nothing in ``src/`` catches everything and does nothing.

    The audit counted five of these. Four were fixed — the two frontier
    anchors now record ``anchor_failures``, the resampling draw counts itself
    as failed, and the feasibility probe reports a solver crash as a solver
    crash instead of as an infeasible mandate. The fifth is on the allowlist.
    """
    pass_sites, _ = _scan()
    _check(pass_sites, PASS_ALLOWLIST, "then do nothing")


def test_no_silently_returned_fallbacks() -> None:
    """Scan B: a lone ``return`` in a broad handler must report the failure.

    The extension the ``pass``-only scan cannot make. A handler that returns a
    fallback *and* names the exception in what it returns is doing the right
    thing — ``frontier.py``'s ``return target, None, str(exc)`` is the model,
    and it is why this scan does not simply flag every single-return handler.
    A handler that returns a fallback and says nothing has the same defect as
    ``pass``: the caller gets a value it cannot tell apart from an answer.
    """
    _, return_sites = _scan()
    _check(return_sites, RETURN_ALLOWLIST, "return a fallback without reporting it")


def test_the_discriminator_separates_reporting_from_swallowing() -> None:
    """Pin scan B's rule, so a future edit cannot quietly widen the exemption.

    ``_returns_the_exception`` is the only thing standing between "this scan
    finds a real defect" and "this scan flags every fallback in the tree".
    If it ever returns ``True`` for a handler that discards the exception,
    scan B goes blind without failing.
    """
    module = ast.parse(
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as exc:\n"
        "        return None, str(exc)\n"
        "    except Exception as exc:\n"
        "        return fallback\n"
        "    except Exception:\n"
        "        return fallback\n"
    )
    handlers = [n for n in ast.walk(module) if isinstance(n, ast.ExceptHandler)]
    handlers.sort(key=lambda node: node.lineno)
    reporting, bound_but_unused, unbound = handlers
    assert _returns_the_exception(reporting)
    assert not _returns_the_exception(bound_but_unused)
    assert not _returns_the_exception(unbound)
    assert _is_broad(unbound)


def test_the_scan_sees_the_handler_it_allows() -> None:
    """A scan that found nothing would pass this file without meaning anything.

    Both allowlists are short, and a refactor that broke ``_scan`` — a changed
    AST node name, a directory move — would empty them silently and turn every
    assertion above into a tautology. Pinning that the known handler is still
    *found* is what keeps the green meaningful.
    """
    pass_sites, return_sites = _scan()
    found = {site.path for site in pass_sites}
    assert "optimization_engine/optimizers/hrp.py" in found, (
        "the scan no longer finds HRP's known silent handler; either it was "
        "fixed (delete its allowlist entry) or the scan is broken"
    )
    assert {site.path for site in return_sites} == {
        entry.path for entry in RETURN_ALLOWLIST
    }
