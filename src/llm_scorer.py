"""
llm_scorer.py
-------------
Scores essays using an LLM (GPT-4o-mini by default) and appends AI scores
to the corpus DataFrame.

Operating Modes (set via .env → SCORING_MODE):
  "live"  – Calls the OpenAI API.  Requires OPENAI_API_KEY.
  "mock"  – Deterministic, cost-free simulation; injects a realistic DIF signal.
             ⚠ MOCK MODE PRODUCES ARTIFICIAL RESULTS — for pipeline testing only.

The scoring rubric is explicitly anchored to the ASAP/LEAF 1–6 scale and
asks the model to evaluate four sub-constructs mirroring ETS Communication
skill progressions:
  (C1) Clarity of Argumentation
  (C2) Coherence & Organisation
  (C3) Lexical Range & Precision
  (C4) Grammatical Accuracy

Author: Sai Chaitanya Pachipulusu (Independent Psychometric Audit, 2026)
"""

from __future__ import annotations

import os
import time
import json
import random
from collections import deque
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
        return value if value > 0 else default
    except ValueError:
        return default

# ──────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────
SCORING_MODE   = os.getenv("SCORING_MODE", "mock").lower()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

OPENAI_MAX_REQUESTS_PER_MINUTE = _env_int("OPENAI_MAX_REQUESTS_PER_MINUTE", 90)
OPENAI_MIN_INTERVAL_SECONDS    = _env_float("OPENAI_MIN_INTERVAL_SECONDS", 0.10)
OPENAI_MAX_RETRIES             = _env_int("OPENAI_MAX_RETRIES", 5)
OPENAI_RETRY_BASE_SECONDS      = _env_float("OPENAI_RETRY_BASE_SECONDS", 2.0)

SCORING_PROGRESS_EVERY   = _env_int("SCORING_PROGRESS_EVERY", 25)
SCORING_CHECKPOINT_EVERY = _env_int("SCORING_CHECKPOINT_EVERY", 25)
ESTIMATED_COST_PER_ESSAY = _env_float("ESTIMATED_COST_PER_ESSAY_USD", 0.004)

RUBRIC_PROMPT = """\
You are an expert educational assessment specialist trained in the ASAP (Automated
Student Assessment Prize) scoring framework and ETS rubric standards.

Score the following student essay on a 1–6 holistic scale where:
  1 = Inadequate  2 = Limited  3 = Developing  4 = Adequate  5 = Proficient  6 = Exemplary

Consider the following four sub-dimensions and produce a single holistic score:
  C1 – Clarity of Argumentation   (is the thesis clear and well-supported?)
  C2 – Coherence & Organisation   (logical flow, transitions, paragraph structure)
  C3 – Lexical Range & Precision  (vocabulary breadth and accuracy)
  C4 – Grammatical Accuracy       (syntax, punctuation, morphology)

Return ONLY a JSON object in this exact format (no extra text):
{{"holistic_score": <int 1-6>, "c1": <int 1-6>, "c2": <int 1-6>, "c3": <int 1-6>, "c4": <int 1-6>}}

ESSAY:
\"\"\"{essay}\"\"\"
"""

# ──────────────────────────────────────────────────
# Mock Scorer  (no API cost — FOR TESTING ONLY)
# ──────────────────────────────────────────────────
_RNG = np.random.default_rng(seed=99)

def _mock_score(row: pd.Series) -> dict:
    """
    Simulate LLM scoring with a calibrated DIF signal.
    ⚠ WARNING: This produces ARTIFICIAL results.
    Use SCORING_MODE=live for real analysis.
    """
    human = row["human_score"]
    group = row["group"]

    noise = _RNG.normal(0, 0.6)
    base  = human + noise

    # DIF injection: AI systematically under-scores non-native on surface form
    dif_penalty = -0.55 if group == "Non-Native" else 0.0

    holistic = int(np.clip(round(base + dif_penalty), 1, 6))

    # Sub-dimension scores: perturb holistic slightly
    c1 = int(np.clip(holistic + _RNG.integers(-1, 2), 1, 6))
    c2 = int(np.clip(holistic + _RNG.integers(-1, 2), 1, 6))
    # C3 & C4 show the DIF most clearly (surface features)
    c3 = int(np.clip(holistic + round(dif_penalty * 0.8) + _RNG.integers(-1, 2), 1, 6))
    c4 = int(np.clip(holistic + round(dif_penalty * 0.9) + _RNG.integers(-1, 2), 1, 6))

    return {"holistic_score": holistic, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


# ──────────────────────────────────────────────────
# Live LLM Scorer
# ──────────────────────────────────────────────────
class _RateLimiter:
    """Simple rolling-window RPM limiter with optional minimum inter-call spacing."""

    def __init__(self, max_requests_per_minute: int, min_interval_seconds: float):
        self.max_requests_per_minute = max(1, int(max_requests_per_minute))
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.call_timestamps = deque()
        self.last_call_ts = 0.0

    def wait_for_slot(self):
        now = time.monotonic()
        if self.min_interval_seconds > 0 and self.last_call_ts > 0:
            wait_min_gap = self.min_interval_seconds - (now - self.last_call_ts)
            if wait_min_gap > 0:
                time.sleep(wait_min_gap)
                now = time.monotonic()

        window_start = now - 60.0
        while self.call_timestamps and self.call_timestamps[0] < window_start:
            self.call_timestamps.popleft()

        if len(self.call_timestamps) >= self.max_requests_per_minute:
            wait_window = 60.0 - (now - self.call_timestamps[0]) + 0.01
            if wait_window > 0:
                time.sleep(wait_window)
            now = time.monotonic()
            window_start = now - 60.0
            while self.call_timestamps and self.call_timestamps[0] < window_start:
                self.call_timestamps.popleft()

        stamp = time.monotonic()
        self.call_timestamps.append(stamp)
        self.last_call_ts = stamp


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _retry_wait_seconds(attempt_idx: int, is_rate_limited: bool) -> float:
    base = OPENAI_RETRY_BASE_SECONDS * (2 ** attempt_idx)
    if is_rate_limited:
        base *= 1.75
    jitter = random.uniform(0.0, max(0.5, base * 0.20))
    return min(base + jitter, 45.0)


def _coerce_score(value, default: int = 3) -> int:
    try:
        return int(np.clip(int(value), 1, 6))
    except Exception:
        return default


def _build_result_row(essay_id: str, scores: dict) -> dict:
    return {
        "essay_id": essay_id,
        "ai_score": _coerce_score(
            scores.get("holistic_score", scores.get("holistic", scores.get("ai_score", 3)))
        ),
        "ai_c1": _coerce_score(scores.get("c1", scores.get("ai_c1", 3))),
        "ai_c2": _coerce_score(scores.get("c2", scores.get("ai_c2", 3))),
        "ai_c3": _coerce_score(scores.get("c3", scores.get("ai_c3", 3))),
        "ai_c4": _coerce_score(scores.get("c4", scores.get("ai_c4", 3))),
    }


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or np.isnan(seconds):
        return "0s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _print_progress(
    done: int,
    total: int,
    skipped: int,
    api_calls: int,
    failed_calls: int,
    retry_events: int,
    start_time: float,
):
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    pct = (100.0 * done / total) if total else 100.0
    bar_width = 24
    filled = int(round((pct / 100.0) * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)

    print(
        f"  [{bar}] {done}/{total} ({pct:5.1f}%) | "
        f"cache={skipped} api={api_calls} failed={failed_calls} retries={retry_events} | "
        f"rate={rate:.2f}/s ETA={_format_eta(eta)}"
    )


def _live_score(
    row: pd.Series,
    client,
    rate_limiter: _RateLimiter,
    max_retries: int,
) -> tuple[dict, bool, int]:
    """Call OpenAI API and parse structured JSON response."""
    # Truncate to 1500 chars to manage token cost (avg essay ~300-600 words)
    essay_text = str(row.get("essay_text", ""))[:1500]
    prompt = RUBRIC_PROMPT.format(essay=essay_text)
    raw = ""

    for attempt in range(max_retries):
        try:
            rate_limiter.wait_for_slot()
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            raw = (response.choices[0].message.content or "").strip()

            # Parse JSON — strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            scores = json.loads(raw)

            # Validate range
            for key in ["holistic_score", "c1", "c2", "c3", "c4"]:
                if key in scores:
                    scores[key] = int(np.clip(int(scores[key]), 1, 6))

            return scores, True, attempt

        except json.JSONDecodeError as exc:
            print(f"  [LLMScorer] JSON parse error (attempt {attempt + 1}/{max_retries}): {exc}")
            print(f"  [LLMScorer] Raw response: {raw[:200]}")
            if attempt < max_retries - 1:
                time.sleep(_retry_wait_seconds(attempt, is_rate_limited=False))
        except Exception as exc:
            is_rate_limited = _is_rate_limit_error(exc)
            wait = _retry_wait_seconds(attempt, is_rate_limited=is_rate_limited)
            print(
                f"  [LLMScorer] API error (attempt {attempt + 1}/{max_retries}): "
                f"{exc} — retrying in {wait:.1f}s"
            )
            if attempt < max_retries - 1:
                time.sleep(wait)

    # Fallback on persistent failure — return neutral score, NOT mock
    print(f"  [LLMScorer] ⚠ All attempts failed for essay {row.get('essay_id', '?')} — returning neutral 3")
    return {"holistic_score": 3, "c1": 3, "c2": 3, "c3": 3, "c4": 3}, False, max_retries - 1


# ──────────────────────────────────────────────────
# Checkpoint / Resume Support
# ──────────────────────────────────────────────────
def _load_checkpoint(checkpoint_path: Path) -> dict:
    """Load previously scored results from checkpoint."""
    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text())
            print(f"  [Checkpoint] Loaded {len(data)} previously scored essays")
            return {r["essay_id"]: r for r in data if "essay_id" in r}
        except json.JSONDecodeError:
            print("  [Checkpoint] Existing checkpoint is malformed; ignoring and starting fresh.")
    return {}


def _save_checkpoint(results: list[dict], checkpoint_path: Path):
    """Save scored results to checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(results, indent=2))


# ──────────────────────────────────────────────────
# Public Entry Point
# ──────────────────────────────────────────────────
def score_corpus(
    df: pd.DataFrame,
    save_path: str = "data/scored_corpus.csv",
    checkpoint_path: str = "data/_scoring_checkpoint.json",
) -> pd.DataFrame:
    """
    Add AI scores to the corpus DataFrame.

    Parameters
    ----------
    df             : Output of leaf_loader.load_corpus()
    save_path      : Where to persist the scored corpus
    checkpoint_path: Where to save/resume scoring progress
                     (ensures you don't lose work if API errors occur)

    Returns
    -------
    df with additional columns:
        ai_score, ai_c1, ai_c2, ai_c3, ai_c4
    """
    mode = SCORING_MODE  # local copy — avoids UnboundLocalError on fallback
    rate_limiter = None
    progress_every = max(1, SCORING_PROGRESS_EVERY)
    checkpoint_every = max(1, SCORING_CHECKPOINT_EVERY)

    if mode == "live":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            rate_limiter = _RateLimiter(
                max_requests_per_minute=OPENAI_MAX_REQUESTS_PER_MINUTE,
                min_interval_seconds=OPENAI_MIN_INTERVAL_SECONDS,
            )
            est_minutes = max(
                len(df) / max(OPENAI_MAX_REQUESTS_PER_MINUTE, 1),
                len(df) * max(OPENAI_MIN_INTERVAL_SECONDS, 0.6) / 60.0,
            )
            print(f"[LLMScorer] Mode=LIVE  Model={OPENAI_MODEL}  n={len(df)} essays")
            print(f"[LLMScorer] Estimated cost: ~${len(df) * ESTIMATED_COST_PER_ESSAY:.2f}")
            print(f"[LLMScorer] Estimated time: ~{est_minutes:.1f} minutes")
            print(
                "[LLMScorer] Throttle config: "
                f"{OPENAI_MAX_REQUESTS_PER_MINUTE} RPM, "
                f"min {OPENAI_MIN_INTERVAL_SECONDS:.2f}s between calls, "
                f"max_retries={OPENAI_MAX_RETRIES}"
            )
            print(
                "[LLMScorer] Progress/checkpoint cadence: "
                f"every {progress_every} / {checkpoint_every} essays"
            )
        except ImportError:
            print("[LLMScorer] openai package not found – falling back to mock mode")
            mode   = "mock"
            client = None
    else:
        client = None
        print(f"[LLMScorer] Mode=MOCK  (deterministic DIF simulation)  n={len(df)} essays")
        print(f"[LLMScorer] ⚠ WARNING: Mock mode produces ARTIFICIAL results!")
        print(f"[LLMScorer]   Set SCORING_MODE=live in .env for real analysis.")

    # ── Load checkpoint for resume support ────────────────
    ckpt_path = Path(checkpoint_path)
    scored_cache = _load_checkpoint(ckpt_path) if mode == "live" else {}

    df = df.reset_index(drop=True)  # ensure integer RangeIndex
    results = []
    skipped = 0
    api_calls = 0
    failed_calls = 0
    retry_events = 0
    start_time = time.time()

    for pos in range(len(df)):
        row = df.iloc[pos]
        essay_id = row["essay_id"]

        # Resume from checkpoint if available
        if essay_id in scored_cache:
            result_row = _build_result_row(essay_id, scored_cache[essay_id])
            skipped += 1
        else:
            if mode == "live" and client and rate_limiter:
                scores, success, retries_used = _live_score(
                    row,
                    client,
                    rate_limiter=rate_limiter,
                    max_retries=max(1, OPENAI_MAX_RETRIES),
                )
                api_calls += 1
                retry_events += retries_used
                if not success:
                    failed_calls += 1
            else:
                scores = _mock_score(row)
            result_row = _build_result_row(essay_id, scores)

        results.append(result_row)

        done = pos + 1

        # Save checkpoint periodically (live mode only)
        if mode == "live" and done % checkpoint_every == 0:
            _save_checkpoint(results, ckpt_path)

        # Progress reporting
        if done % progress_every == 0 or done == len(df):
            _print_progress(
                done=done,
                total=len(df),
                skipped=skipped,
                api_calls=api_calls,
                failed_calls=failed_calls,
                retry_events=retry_events,
                start_time=start_time,
            )

    # Final checkpoint save
    if mode == "live" and results:
        _save_checkpoint(results, ckpt_path)

    if skipped > 0:
        print(
            f"  [LLMScorer] Resumed {skipped} from checkpoint, "
            f"made {api_calls} new API calls"
        )

    scores_df = pd.DataFrame(results).set_index("essay_id")
    df = df.set_index("essay_id").join(scores_df).reset_index()

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    elapsed_total = time.time() - start_time
    print(f"[LLMScorer] Scored corpus saved → {out.resolve()}")
    print(
        f"[LLMScorer] Total time: {elapsed_total:.0f}s | "
        f"API calls: {api_calls} | failed calls: {failed_calls} | retries: {retry_events}"
    )
    return df


if __name__ == "__main__":
    from leaf_loader import load_corpus
    corpus  = load_corpus()
    scored  = score_corpus(corpus)
    print(scored[["essay_id", "group", "human_score", "ai_score"]].head(10))
