# Stage 4 — providers and authentication

Supporting artifacts for `docs/modernization-plan.md`'s Stage 4. Nothing
here is authoritative on its own — the plan is.

- `provider_comparison.md` — the required "provider/model comparison table"
  (Stage 4's action/Gate) across Anthropic, OpenAI, and Hugging Face
  candidate models: retrieval-answer quality score (UNVERIFIED as of every
  run so far — no funded API key/OAuth token has been available in any
  environment this table was built in, blocking reason recorded explicitly,
  not fabricated), real generation p50/p95 latency (also UNVERIFIED, same
  blocker), real invalid-key auth-rejection latency (measured live, clearly
  distinguished from generation latency), and cost-per-1k-tokens (sourced
  from each provider's own published pricing page, informational only per
  the shared contract).
- `build_provider_comparison.py` — the script that produced
  `provider_comparison.md`'s non-prose numbers. Re-run:
  `python docs/stage4/build_provider_comparison.py`. Always makes real
  network calls to Anthropic/OpenAI/Hugging Face with a deliberately
  invalid test credential (never a real key) to measure auth-rejection
  latency and confirm no key-substring leak. **Also supports a real
  generation benchmark** against every `docs/stage0/query_set.json` entry
  with an `expected_urls` list (17 of 23), using the real Stage 2 retrieval
  index — but only for whichever of `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/
  `HF_TOKEN` is actually present and funded in the environment the script is
  run in; a provider with no funded credential present is skipped, not
  estimated. No environment this script has been run in (through this
  document's most recent revision) has had a funded credential available,
  so every quality-score cell remains UNVERIFIED — see the script's own
  module docstring for the exact mechanics and why it deliberately does not
  compute an automated quality score even when it does have a funded key
  (that scoring methodology is still an explicitly open decision — see
  `provider_comparison.md`'s "What would close the quality-score gap").
- `evidence/` — raw JSON output from each script run, retained per Stage 0's
  established gate pattern ("raw outputs are retained for review"); never
  overwritten, one dated file per run. Each file's `live_generation_results`
  is `[]` (and `live_generation_skip_reasons` explains why) for every run so
  far.

Provider contract tests, the forced-failure BYOK secret-leak test, the
simulated slow/failing provider (timeout/retry/concurrency) test, and the
retrieval-survives-every-provider-failure (kill-switch) test are NOT in this
directory — they are pytest suites under `tests/` (`test_providers_anthropic.py`,
`test_providers_openai.py`, `test_providers_huggingface.py`,
`test_provider_secret_leak.py`, `test_provider_timeout_retry_concurrency.py`,
`test_app_provider_kill_switch.py`), run as part of this repo's normal
`pytest` gate, not as a separate manual step like the comparison-table
script above.
