"""OpenAI `Provider` implementation using the official SDK's Responses API.

Per `docs/modernization-plan.md`'s "Optional answer synthesis" and this
stage's shared contract: uses the official `openai` Python SDK (never a raw
HTTP call), the Responses API (`client.responses.create`) with `store=False`
(the plan's explicit instruction -- this app never needs OpenAI to retain
conversation state server-side), and current model IDs supplied by the
caller (`model` is a `Provider.generate` parameter, not hardcoded here --
model selection/curation is the separate "provider/model comparison-table
action" named in the shared contract, out of this module's scope).

The API key is BYOK and session-scoped (see `docs/modernization-plan.md`'s
"Secrets, cost, and public abuse"): `OpenAIProvider` only ever holds it in
memory for its own lifetime and never logs, persists, or re-embeds it in an
exception message -- every error raised by `generate` carries a fixed,
literal string, never SDK-supplied text, because OpenAI's own error bodies
have been observed to echo back a masked fragment of an invalid key (e.g.
`AuthenticationError`'s "Incorrect API key provided: sk-...") and `APIError`
subclasses also carry the originating `httpx.Request` (with the
`Authorization` header) as a public attribute -- neither is safe to touch
here, let alone log.

Every re-raise below deliberately uses `raise ... from None`, not `from exc`:
`from exc` sets `__cause__`, and Python's *default* traceback formatting
(`traceback.format_exc()`, `logging.exception()`, an uncaught-exception
handler) walks and prints the full `__cause__` chain -- including the raw
SDK exception's own message, which is exactly what this module's fixed,
literal messages exist to avoid. Verified concretely, not assumed: with
`from exc`, `traceback.format_exc()` on a wrapped `AuthenticationError`
printed the original "Incorrect API key provided: sk-..." text in full.
`from None` breaks the chain so the raw exception is only ever reachable
in-process at the `except ... as exc:` binding itself (never persisted onto
the raised exception object).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Final

import openai
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from govgis.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransientError,
)

# Directionally correct as of 2026-07-20 per
# `docs/stage4/provider_comparison.md`'s "Recommended defaults" (the
# script-generated comparison table Stage 4's Gate requires) -- both IDs were
# confirmed live and current there, not re-guessed here. Re-verify against
# OpenAI's own model/pricing docs at implementation time before relying on
# these, mirroring `govgis/providers/anthropic.py`'s
# `DEFAULT_FAST_MODEL`/`DEFAULT_QUALITY_MODEL` constants.
DEFAULT_FAST_MODEL: Final[str] = "gpt-5.6-luna"
DEFAULT_QUALITY_MODEL: Final[str] = "gpt-5.6-terra"

# Per the shared contract: "retries limited to 1-2 attempts" -- 2 total
# attempts means one retry. Only `ProviderTransientError`/
# `ProviderRateLimitError` are retried (see `_RETRYABLE_ERRORS` below);
# `timeout` is treated as a hard per-attempt bound (see `base.py`'s
# `Provider` docstring), not a total-across-retries budget, so retrying
# does not silently double the caller-supplied timeout into a single
# apparent call -- each attempt independently respects it.
_MAX_ATTEMPTS: Final[int] = 2
_RETRY_WAIT_MIN_SECONDS: Final[float] = 0.5
_RETRY_WAIT_MAX_SECONDS: Final[float] = 2.0

_RETRYABLE_ERRORS: Final[tuple[type[ProviderError], ...]] = (
    ProviderTransientError,
    ProviderRateLimitError,
)

_AUTH_MESSAGE: Final[str] = "OpenAI rejected the request as unauthenticated or unauthorized."
_RATE_LIMIT_MESSAGE: Final[str] = "OpenAI reported rate limiting."
_TRANSIENT_MESSAGE: Final[str] = "OpenAI reported a transient/server-side failure."
_INVALID_REQUEST_MESSAGE: Final[str] = "OpenAI rejected the request as invalid."
_TIMEOUT_MESSAGE: Final[str] = "The OpenAI request did not complete within the given timeout."
_GENERATION_FAILED_MESSAGE: Final[str] = "OpenAI reported the generation did not complete."
_UNEXPECTED_MESSAGE: Final[str] = "OpenAI returned an unexpected error."

# Response.status values that mean "the SDK call itself succeeded (HTTP 200)
# but the generation did not produce a usable, final answer" -- see
# `openai.types.responses.Response`. `background`/`stream` are never passed
# by this module, so `queued`/`in_progress` are not expected in practice;
# they are still handled explicitly (as failures) rather than silently
# returning `output_text` for a response that has not actually finished.
_FAILED_STATUSES: Final[frozenset[str]] = frozenset(
    {"failed", "cancelled", "queued", "in_progress"}
)


@dataclass(slots=True)
class OpenAIProvider:
    """A `base.Provider` implementation backed by the official OpenAI SDK.

    `api_key` is stored only for this instance's lifetime (`repr=False` so
    it can never leak through an accidental `repr()`/log of the provider
    object itself) -- see the module docstring for the full BYOK rationale.

    One instance is expected per user session, mirroring
    `govgis.providers.anthropic.AnthropicProvider`. `generate` rejects a
    second concurrent call outright (rather than queuing it) as a defensive,
    fail-fast backstop for the shared contract's per-session concurrency
    limit of 1 in-flight generation call -- the primary enforcement is
    expected at the UI call site once this provider is wired into `app.py`,
    not here. This mirrors `AnthropicProvider`'s own module-level lock;
    unlike that class, this one predates it being wired into `app.py`, so
    closing this gap now (rather than after wiring) avoids a window where the
    concurrency guarantee documented in `docs/modernization-plan.md`'s
    shared contract would not actually hold for this provider.
    """

    api_key: str = field(repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
        init=False,
    )

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int,
        timeout: float,
    ) -> str:
        if not self._lock.acquire(blocking=False):
            raise ProviderTransientError(
                "OpenAI provider is already handling a generation call for this session.",
            )
        try:
            for attempt in Retrying(
                reraise=True,
                stop=stop_after_attempt(_MAX_ATTEMPTS),
                wait=wait_exponential(min=_RETRY_WAIT_MIN_SECONDS, max=_RETRY_WAIT_MAX_SECONDS),
                retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            ):
                with attempt:
                    return self._generate_once(
                        prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
            # Unreachable: `Retrying.__iter__` always either returns a value
            # from the `with attempt:` block above or raises (via
            # `reraise=True`).
            raise AssertionError("unreachable")  # pragma: no cover
        finally:
            self._lock.release()

    def _generate_once(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float | None,
        max_tokens: int,
        timeout: float,
    ) -> str:
        # `max_retries=0`: this module owns the retry policy above via
        # tenacity so the caller-supplied "1-2 attempts, transient/5xx only,
        # never auth" contract is respected exactly once, not doubled by the
        # SDK's own default (`max_retries=2`, which does not distinguish
        # auth failures from transient ones the way `base.py` requires).
        client = openai.OpenAI(api_key=self.api_key, max_retries=0)
        try:
            if temperature is None:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    store=False,
                    timeout=timeout,
                )
            else:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    store=False,
                    timeout=timeout,
                    temperature=temperature,
                )
        except openai.APITimeoutError:
            raise ProviderTimeoutError(_TIMEOUT_MESSAGE) from None
        except openai.AuthenticationError, openai.PermissionDeniedError:
            raise ProviderAuthError(_AUTH_MESSAGE) from None
        except openai.RateLimitError:
            raise ProviderRateLimitError(_RATE_LIMIT_MESSAGE) from None
        except (
            openai.BadRequestError,
            openai.NotFoundError,
            openai.ConflictError,
            openai.UnprocessableEntityError,
        ):
            raise ProviderInvalidRequestError(_INVALID_REQUEST_MESSAGE) from None
        except openai.InternalServerError:
            raise ProviderTransientError(_TRANSIENT_MESSAGE) from None
        except openai.APIConnectionError:
            # Non-timeout connection failure (DNS, TLS, reset, ...) --
            # `APITimeoutError` is a subclass and is already handled above.
            raise ProviderTransientError(_TRANSIENT_MESSAGE) from None
        except openai.APIStatusError as exc:
            # Any status-coded failure not already matched by a specific
            # except clause above (an SDK version skew or a status code not
            # yet mapped to a named exception type). Classify by the raw
            # status code rather than guessing from the SDK's message text.
            if exc.status_code in (401, 403):
                raise ProviderAuthError(_AUTH_MESSAGE) from None
            if exc.status_code == 429:
                raise ProviderRateLimitError(_RATE_LIMIT_MESSAGE) from None
            if exc.status_code >= 500:
                raise ProviderTransientError(_TRANSIENT_MESSAGE) from None
            raise ProviderInvalidRequestError(_INVALID_REQUEST_MESSAGE) from None
        except openai.OpenAIError:
            # Catch-all for any other SDK-raised error shape (e.g. response
            # schema validation failures) -- never let a raw SDK exception
            # (which can carry the request, including the key) propagate.
            raise ProviderError(_UNEXPECTED_MESSAGE) from None

        if response.status in _FAILED_STATUSES:
            raise ProviderTransientError(_GENERATION_FAILED_MESSAGE)
        # `status == "incomplete"` (e.g. hit `max_output_tokens`) still
        # carries whatever partial text was generated; returning it rather
        # than raising matches the plan's "show results even if answer
        # synthesis [is imperfect]" spirit better than discarding usable
        # partial output. `status in {"completed", None}` is the normal path.
        return response.output_text
