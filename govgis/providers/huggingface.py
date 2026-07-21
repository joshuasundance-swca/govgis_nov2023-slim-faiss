"""Curated Hugging Face Inference Providers client, authorized as the signed-in user.

Per ``docs/modernization-plan.md``'s "Target architecture" and Confirmed
decision 4: every call this module makes is authorized with the SIGNED-IN
Hugging Face user's own OAuth access token (``inference-api`` scope) -- never
the Space owner's token. Callers construct one ``HuggingFaceProvider`` per
request, passing that user's token explicitly; this module holds no
fallback/default token of its own and never reads ``HF_TOKEN`` or any other
ambient credential (an ambient-credential fallback would silently reintroduce
owner-funded usage, which the plan's "Secrets, cost, and public abuse"
section forbids without a separate, explicit decision).

``CURATED_MODELS`` is a small, hand-picked set of open chat models, per the
plan's "curate a small set of available open chat models" and "Do not expose
arbitrary provider-specific parameters in the first release" -- this module
never accepts an arbitrary model string or a provider-specific override.
Each ID was confirmed live against
<https://huggingface.co/docs/inference-providers/en/index> on 2026-07-20 (the
doc's own quick-start and provider-selection code samples), not guessed;
re-verify at a later implementation/release milestone per the plan's
"Evidence sources" note that these links are time-sensitive.

``HuggingFaceProvider.generate`` implements this module's share of the
Stage 4 shared contract's bounded-retry requirement (1-2 attempts, transient/
rate-limit failures only, never on auth or invalid-request errors) internally
via ``tenacity`` -- callers must not additionally retry a ``ProviderError``
this raises, since that would compound retries beyond the contract's bound
(see ``govgis/providers/base.py``'s ``Provider`` docstring). The per-session
concurrency limit of 1 in-flight generation call is a caller-side concern
(a "session" is a Gradio browser session, which this stateless class has no
visibility into) -- see ``app.py``'s HF answer-synthesis wiring.

Every re-raise below deliberately uses ``raise ... from None``, not
``from exc``: ``from exc`` sets ``__cause__``, and Python's *default*
traceback formatting (``traceback.format_exc()``, ``logging.exception()``, an
uncaught-exception handler) walks and prints the full ``__cause__`` chain --
including the raw ``HfHubHTTPError``/``httpx`` exception's own message and
response body, which can carry this call's OAuth access token (e.g. in a
"token ..." rejection message or an echoed ``Authorization`` header). A
caller that logs a caught ``ProviderError`` with ``logging.exception(...)``,
or lets one propagate uncaught, would otherwise print the token in full even
though this module's own message stays sanitized -- see
``govgis/providers/openai.py``'s module docstring for the same property,
verified concretely there.
"""

from __future__ import annotations

from typing import Final, Literal

import httpx
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from govgis.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTransientError,
)

# `provider="auto"` (this module's only mode) lets Hugging Face route each
# request to whichever backing Inference Provider currently serves the model
# fastest -- see the "Provider Selection" section of the docs cited above.
# Not user-configurable here: exposing a specific downstream provider name
# would be exactly the "provider-specific parameter" the plan says to keep
# out of the first release.
_ROUTING_POLICY: Final[Literal["auto"]] = "auto"

CURATED_MODELS: Final[tuple[str, ...]] = (
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
)
DEFAULT_MODEL: Final[str] = CURATED_MODELS[0]

# Shared contract: "retries limited to 1-2 attempts on transient/5xx failures
# only (never retry auth errors)".
_MAX_ATTEMPTS: Final[int] = 2
_RETRY_MIN_WAIT_SECONDS: Final[float] = 0.5
_RETRY_MAX_WAIT_SECONDS: Final[float] = 2.0

_AUTH_STATUS_CODES: Final[frozenset[int]] = frozenset({401, 403})
_RATE_LIMIT_STATUS_CODE: Final[int] = 429
_INVALID_REQUEST_STATUS_CODES: Final[frozenset[int]] = frozenset({400, 404, 422})


class HuggingFaceProvider:
    """``Provider`` implementation backed by ``huggingface_hub.InferenceClient``.

    Structurally satisfies ``govgis.providers.base.Provider`` (see that
    module's ``Provider`` ``Protocol``) without inheriting from it.
    """

    def __init__(self, *, token: str) -> None:
        if not token:
            # Deliberately does not echo `token` -- even an empty/falsy
            # value -- into this message; see module docstring and the
            # plan's "never let a key/credential enter ... exception
            # messages" non-negotiable.
            raise ProviderAuthError("a Hugging Face OAuth access token is required")
        self._token = token

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Return one chat-completion response for ``prompt`` from ``model``.

        Raises only ``govgis.providers.base.ProviderError`` subclasses --
        never a raw ``huggingface_hub``/``httpx`` exception, which can carry
        this call's Authorization header (see module docstring).
        """
        if model not in CURATED_MODELS:
            raise ProviderInvalidRequestError(
                f"model {model!r} is not in the curated Hugging Face model set",
            )
        return self._generate_with_retry(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    @retry(
        retry=retry_if_exception_type((ProviderTransientError, ProviderRateLimitError)),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(min=_RETRY_MIN_WAIT_SECONDS, max=_RETRY_MAX_WAIT_SECONDS),
        reraise=True,
    )
    def _generate_with_retry(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float | None,
        max_tokens: int,
        timeout: float,
    ) -> str:
        client = InferenceClient(
            model=model,
            token=self._token,
            timeout=timeout,
            provider=_ROUTING_POLICY,
        )
        try:
            completion = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        # `from None` on every branch below is deliberate -- see module
        # docstring. `from exc` would preserve the raw SDK/httpx exception as
        # `__cause__`, and default traceback formatting prints that full
        # chain, which can carry the OAuth access token.
        except InferenceTimeoutError:
            raise ProviderTimeoutError(
                f"Hugging Face inference call for {model!r} timed out after {timeout}s",
            ) from None
        except HfHubHTTPError as exc:
            raise _translate_http_error(exc, model=model) from None
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                f"Hugging Face inference call for {model!r} timed out after {timeout}s",
            ) from None
        except httpx.HTTPError:
            # Broad transport-layer failures (connection reset, DNS, TLS,
            # ...) that never reached a response with a status code -- this
            # is the "transient/5xx failures" bucket by nature (nothing about
            # the request itself is at fault), so it is safe to retry.
            raise ProviderTransientError(
                "Hugging Face inference call failed due to a network error",
            ) from None

        choices = completion.choices
        if not choices or choices[0].message.content is None:
            raise ProviderError(f"Hugging Face returned no completion content for {model!r}")
        return choices[0].message.content


def _translate_http_error(exc: HfHubHTTPError, *, model: str) -> ProviderError:
    # `HfHubHTTPError.response` is a real `httpx.Response` (see
    # `huggingface_hub.errors`), so `.status_code` is always present here --
    # never fall back to parsing `str(exc)`, which could echo response body
    # text this module has not vetted for secret-shaped content.
    status = exc.response.status_code
    if status in _AUTH_STATUS_CODES:
        return ProviderAuthError(
            "Hugging Face rejected the request as unauthenticated or unauthorized",
        )
    if status == _RATE_LIMIT_STATUS_CODE:
        return ProviderRateLimitError("Hugging Face rate-limited this request")
    if status in _INVALID_REQUEST_STATUS_CODES:
        return ProviderInvalidRequestError(
            f"Hugging Face rejected the request for {model!r} (HTTP {status})",
        )
    if status >= 500:
        return ProviderTransientError(f"Hugging Face returned a transient error (HTTP {status})")
    return ProviderError(f"Hugging Face inference call failed (HTTP {status})")
