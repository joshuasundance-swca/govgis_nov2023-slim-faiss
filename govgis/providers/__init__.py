"""Provider-neutral answer-synthesis clients.

Per `docs/modernization-plan.md`'s "Target architecture": each sibling module
in this package (`anthropic.py`, `openai.py`, `huggingface.py`) implements
the `Provider` protocol defined in `base.py` and raises only `base.py`'s
typed exception hierarchy -- never a raw SDK exception, which can carry the
request, including an API key. This package intentionally has no other
public surface; import from the specific submodule you need.
"""

from __future__ import annotations
