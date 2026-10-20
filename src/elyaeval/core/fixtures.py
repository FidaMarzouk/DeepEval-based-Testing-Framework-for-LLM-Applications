"""
Most of the time you should skip this module entirely: point your test's
GOLDENS at a goldens file written against what your app's corpus already
contains, call your app normally, and let it retrieve for real.

This fixture exists for one specific situation: you want to run the
*shared* standard dataset (elyaeval/standard_dataset/goldens.jsonl) a
generic, cross-app benchmark   against YOUR app. In that situation, and only then, retrieval-dependent
metrics (ContextualPrecision/Recall/Relevancy, Faithfulness) need
something to find, so this fixture temporarily inserts each golden's
ground-truth `context` passages before the suite runs and removes them
after. Generate that variant explicitly with
`elyaeval init --task-type <type> --corpus-mode seeded`.

This module owns everything that is the SAME for every consuming app:
looping goldens, deduping context passages, and the pytest fixture
lifecycle (ingest before the session, teardown after). It knows nothing
about HOW to ingest or tear down, or what your storage looks like
every app's backend is different (pgvector, Chroma, FAISS, an HTTP
endpoint, ...)   so that part is supplied by the app as two callables.

Contract: ingest_fn(passage) inserts one passage and returns whatever
handle your storage needs to delete it again later   a SQL row id, a
Chroma document id, a vector index position, a dict key, anything your
backend can address. teardown_fn(handles) receives the exact list of
handles that were returned and removes exactly those, nothing else.

"""
from collections.abc import Callable
from typing import Any, Literal, Optional

import pytest
from deepeval.dataset import Golden


#factory function: you feed it your goldens and your app's specific insert/delete logic,
#  and it hands you back a ready-made pytest fixture shaped for your app
def make_golden_context_fixture(
    goldens: list[Golden],
    ingest_fn: Callable[[str], Any],
    teardown_fn: Optional[Callable[[list[Any]], None]] = None,
    #how long the fixture lives for (default: once per test session)
    scope: Literal["session", "package", "module", "class", "function"] = "session",
):
    """
    Returns a pytest fixture (autouse, given `scope`) that:
      1. Collects every unique `context` passage across `goldens`.
      2. Calls `ingest_fn(passage)` once per unique passage before the suite
         runs, collecting each returned handle.
      3. Calls `teardown_fn(handles)` once after the suite finishes, passing
         the full list of handles collected in step 2   if provided.

    ingest_fn / teardown_fn are the only app-specific pieces you write.

    Use this only when `goldens` is (or includes) the shared standard
    dataset and your app's corpus doesn't already contain matching
    content   see this module's docstring. If you've written your own
    goldens against your app's real, already-ingested corpus, don't wire
    this fixture in at all; just call your app and let it retrieve
    normally.
    """
    #set automatically drops duplicates
    seen: set[str] = set()
    for golden in goldens:
        for passage in golden.context or []:
            seen.add(passage)

    @pytest.fixture(scope=scope, autouse=True)
    def _golden_context_fixture():
        handles = [ingest_fn(passage) for passage in seen]
        yield
        if teardown_fn is not None:
            teardown_fn(handles)

    return _golden_context_fixture
