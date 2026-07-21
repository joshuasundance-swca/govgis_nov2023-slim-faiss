"""Anthropic Messages API provider.

Implements `govgis.providers.base.Provider` via the official Anthropic Python
SDK. Per the Stage 4 shared contract: the caller supplies a session-scoped,
BYOK API key (never the Space owner's); every call carries a bounded
timeout, retries are limited to 1-2 attempts and only for transient/5xx
failures (never for auth errors), concurrent calls against one instance are
rejected rather than queued, and no raw SDK exception -- which can carry the
request, including the API key -- is ever allowed to propagate: every
`anthropic.AnthropicError` is caught here and re-raised as one of
`govgis.providers.base`'s sanitized, message-only exception types.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Final

import anthropic

from govgis.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransientError,
)

if TYPE_CHECKING:
    from anthropic.types import Message

# Directionally correct as of 2026-07-20 per the modernization plan's Stage 4
# shared contract -- re-verify against Anthropic's model-deprecations /
# choosing-a-model pages at implementation time before relying on these.
DEFAULT_FAST_MODEL: Final[str] = "claude-haiku-4-5-20251001"
DEFAULT_QUALITY_MODEL: Final[str] = "claude-sonnet-5"

# "1-2 attempts" per the shared contract, counted as retries after the
# initial call (so at most 3 total calls to the SDK per `generate`).
_MAX_RETRY_ATTEMPTS: Final[int] = 2


class AnthropicProvider:
    """Bound to one caller-supplied, session-scoped Anthropic API key.

    One instance is expected per user session. `generate` rejects a second
    concurrent call outright (rather than queuing it) as a defensive,
    fail-fast backstop for the shared contract's per-session concurrency
    limit of 1 in-flight generation call -- the primary enforcement is
    expected at the UI call site (Stage 3/4 `app.py`), not here.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise ProviderAuthError("Anthropic API key is empty.")
        # max_retries=0: this class owns retry policy (bounded, transient-only,
        # never-retry-auth per the shared contract). Leaving the SDK's own
        # retrier enabled underneath would double-retry and could retry
        # requests this policy forbids retrying.
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        self._lock = threading.Lock()

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
                "Anthropic provider is already handling a generation call for this session.",
            )
        try:
            return self._generate_with_retry(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        finally:
            self._lock.release()

    def _generate_with_retry(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float | None,
        max_tokens: int,
        timeout: float,
    ) -> str:
        last_error: ProviderError | None = None
        for attempt in range(_MAX_RETRY_ATTEMPTS + 1):
            try:
                return self._call(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            except (ProviderTransientError, ProviderTimeoutError, ProviderRateLimitError) as exc:
                last_error = exc
                if attempt >= _MAX_RETRY_ATTEMPTS:
                    raise
        # Unreachable: the loop above either returns or raises on its final
        # iteration. Satisfies mypy's control-flow analysis without `assert`.
        raise (
            last_error
            if last_error is not None
            else ProviderTransientError(
                "Anthropic call failed with no recorded error.",
            )
        )

    def _call(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float | None,
        max_tokens: int,
        timeout: float,
    ) -> str:
        try:
            if temperature is None:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                )
            else:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=timeout,
                )
        # Ordered most-specific-first: AuthenticationError/PermissionDeniedError/
        # RateLimitError/APITimeoutError/NotFoundError are APIStatusError or
        # APIConnectionError subclasses, so they must be caught before those
        # parents. `from None` is deliberate on every branch -- an SDK
        # exception's message/args/request can carry the API key, and this
        # module must never let that reach a caller, log, or traceback.
        except anthropic.AuthenticationError:
            raise ProviderAuthError("Anthropic rejected the provided API key.") from None
        except anthropic.PermissionDeniedError:
            raise ProviderAuthError("Anthropic denied access for the provided API key.") from None
        except anthropic.RateLimitError:
            raise ProviderRateLimitError("Anthropic rate-limited this request.") from None
        except anthropic.APITimeoutError:
            raise ProviderTimeoutError(
                f"Anthropic call did not complete within {timeout}s.",
            ) from None
        except anthropic.APIConnectionError:
            raise ProviderTransientError("Could not reach Anthropic.") from None
        except anthropic.NotFoundError:
            raise ProviderInvalidRequestError(f"Anthropic model not found: {model}.") from None
        except anthropic.APIStatusError as exc:
            # Covers BadRequestError/UnprocessableEntityError/ConflictError/
            # RequestTooLargeError/InternalServerError/OverloadedError: every
            # other status-carrying error not special-cased above.
            if exc.status_code >= 500:
                raise ProviderTransientError(
                    f"Anthropic returned a server error ({exc.status_code}).",
                ) from None
            raise ProviderInvalidRequestError(
                f"Anthropic rejected the request ({exc.status_code}).",
            ) from None
        except anthropic.AnthropicError:
            raise ProviderTransientError("Anthropic request failed.") from None

        return _extract_text(response, model=model)


def _extract_text(response: Message, *, model: str) -> str:
    if response.stop_reason == "refusal":
        raise ProviderInvalidRequestError(
            f"Anthropic ({model}) declined to generate a response for this request.",
        )
    return "".join(block.text for block in response.content if block.type == "text")
