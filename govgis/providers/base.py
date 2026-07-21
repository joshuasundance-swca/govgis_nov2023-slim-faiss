"""Narrow provider protocol and common typed exception hierarchy.

Per `docs/modernization-plan.md`'s "Target architecture" and this stage's
shared contract: every provider module (`govgis/providers/{anthropic,openai,
huggingface}.py`) implements `Provider` below and raises only exceptions
defined in this module -- never a raw SDK exception. A raw provider
exception can carry the request, including an API key (see "Secrets, cost,
and public abuse"), so it must never reach a log, a caller, or telemetry;
provider modules are responsible for catching their SDK's own exception
types and re-raising as one of these, with a sanitized message.

Shared controls are limited to the portable concepts the plan names --
provider, model, temperature (where supported), and maximum output tokens.
No provider-specific parameter (OpenAI's `store`, Anthropic's `top_k`, an
HF Inference Provider route, ...) may appear on this interface; each
provider module owns translating these portable arguments into its own SDK
call.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ProviderError(Exception):
    """Base class for every typed error a `Provider.generate` call may raise."""


class ProviderTimeoutError(ProviderError):
    """The call did not complete within the caller-supplied `timeout`."""


class ProviderAuthError(ProviderError):
    """The provider rejected the request as unauthenticated or unauthorized.

    Never retried by callers (see the Stage 4 shared contract: "never retry
    auth errors") -- an invalid/expired key will not become valid on retry,
    and retrying only spends more of the caller's bounded attempt budget
    against a request that cannot succeed.
    """


class ProviderRateLimitError(ProviderError):
    """The provider reported rate limiting (HTTP 429 or an SDK equivalent).

    Distinct from `ProviderTransientError` because a caller's retry policy
    may want to treat rate limiting differently (e.g. respect a
    provider-supplied backoff hint) even though, like a transient error, a
    bounded retry is appropriate here and an immediate one is not.
    """


class ProviderTransientError(ProviderError):
    """A transient/5xx-class failure. Safe to retry a bounded number of times.

    Per the Stage 4 shared contract: retries limited to 1-2 attempts, and
    only for failures of this shape -- never for `ProviderAuthError` or
    `ProviderInvalidRequestError`, where the request itself cannot succeed
    regardless of how many times it is repeated.
    """


class ProviderInvalidRequestError(ProviderError):
    """The provider rejected the request itself (bad model name, bad parameters, ...).

    Never retried: repeating an unchanged malformed request fails the same
    way every time.
    """


@runtime_checkable
class Provider(Protocol):
    """The interface every `govgis/providers/*.py` client module implements.

    Deliberately narrow, per the plan's "Shared controls should be limited
    to portable concepts" -- see the module docstring. Implementations must
    treat `timeout` as a hard bound on the call and must not perform more
    internal retries than the caller's own retry policy accounts for (see
    the Stage 4 shared contract: bounded timeout in the 15-30s range,
    retries limited to 1-2 attempts on transient/5xx failures only).
    """

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Return the provider's completion text for `prompt`.

        Raises only `ProviderError` subclasses defined in this module --
        never a raw SDK exception (see module docstring).
        """
        ...
