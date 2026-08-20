"""ASR benchmark for Q3 -> evaluation/asr_benchmark.md.

Q3 asks for a real ASR report: provider and model, languages tested,
code-switching behaviour, approximate quality, observed errors, and
regional-accent performance for Indonesia.

Method, and why it is honest:

The phrases are synthesised by TTS, so the **exact ground-truth text is known**.
That is the only way to compute a real word error rate here - hand-transcribing
audio I generated would be circular, and eyeballing "looks about right" is not a
measurement.

The obvious objection is that TTS audio is cleaner than a human on a phone line,
so these WERs are a **best case**. That is stated in the report rather than
glossed over, and it is why the error analysis matters more than the headline
number: the errors that appear even on clean audio are the ones that will
dominate on a real line.

Providers compared:
  * Groq whisper-large-v3-turbo  (what the agent uses)
  * Groq whisper-large-v3        (larger, slower - is the accuracy worth it?)
  * Gemini audio understanding   (a different architecture entirely)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from ..common.asr import get_asr
from ..common.config import EVAL_DIR, settings
from ..common.logging import log
from ..common.tts import get_tts

# Phrases chosen to test one thing each, not to flatter the model.
PHRASES: list[dict] = [
    # --- India: English with Indian finance vocabulary ---------------------
    {"id": "IN-1", "locale": "en-IN", "voice": "caller_en_in_m",
     "text": "Our turnover was seventy two lakh rupees last year and I need fifteen lakh for working capital.",
     "tests": "Indian numerals (lakh) and finance vocabulary"},
    {"id": "IN-2", "locale": "en-IN", "voice": "caller_en_in_f",
     "text": "What is the processing fee and the foreclosure charge on this loan?",
     "tests": "domain terms that drive retrieval"},

    # --- Philippines: Taglish code-switching -------------------------------
    {"id": "PH-1", "locale": "fil-PH", "voice": "caller_fil_f",
     "text": "Opo, nabayaran ko po yung premium last month through auto-debit.",
     "tests": "Taglish: Tagalog frame with English financial nouns"},
    {"id": "PH-2", "locale": "fil-PH", "voice": "caller_fil_m",
     "text": "Sino po ito? Scam ba ito? Hindi po ako nagbibigay ng OTP sa telepono.",
     "tests": "scam-suspicion objection, the most common opening in this market"},
    {"id": "PH-3", "locale": "fil-PH", "voice": "caller_fil_f",
     "text": "Wala po akong pambayad ngayon, next week na lang po ba ang premium?",
     "tests": "payment difficulty in natural Tagalog"},
    {"id": "PH-4", "locale": "en-PH", "voice": "caller_en_ph_m",
     "text": "I want to talk to a real person, hindi ako komportable sa robot.",
     "tests": "mid-sentence English to Tagalog switch"},

    # --- Indonesia: standard Jakarta register ------------------------------
    {"id": "ID-1", "locale": "id-ID", "voice": "caller_id_m",
     "text": "Cicilan saya jatuh tempo tanggal lima, tapi duitnya belum ada bulan ini.",
     "tests": "core multifinance vocabulary"},
    {"id": "ID-2", "locale": "id-ID", "voice": "caller_id_f",
     "text": "Berapa dendanya kalau telat bayar angsuran? Tenornya tiga puluh enam bulan.",
     "tests": "denda, angsuran, tenor"},
    {"id": "ID-3", "locale": "id-ID", "voice": "caller_id_m",
     "text": "Nanti saya kabari deh kalau sudah ada, sekarang belum bisa transfer.",
     "tests": "indirect refusal - must be transcribed exactly to be detected"},

    # --- Indonesia: REGIONAL ACCENTS (the assessment's explicit ask) -------
    {"id": "JV-1", "locale": "jv-ID", "voice": "caller_jv_m",
     "text": "Nggih Bu, sampun saya bayar minggu kemarin lewat transfer, monggo dicek malih.",
     "tests": "Javanese-accented Indonesian with Javanese interjections"},
    {"id": "JV-2", "locale": "jv-ID", "voice": "caller_jv_f",
     "text": "Matur nuwun Bu, angsuran kula sampun lunas nggih.",
     "tests": "heavier Javanese lexical content"},
    {"id": "SU-1", "locale": "su-ID", "voice": "caller_su_m",
     "text": "Punten Bu, hatur nuhun, cicilan abdi tos dibayar minggu kamari.",
     "tests": "Sundanese-accented Indonesian"},
]

_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: str) -> list[str]:
    return _WORD.sub(" ", (text or "").lower()).split()


def wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Levenshtein word error rate. Returns (wer, edits, ref_len)."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return (0.0 if not hyp else 1.0), len(hyp), 0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[len(hyp)] / len(ref), prev[len(hyp)], len(ref)


def diff_words(reference: str, hypothesis: str) -> list[str]:
    """Words in the reference that never appear in the hypothesis - a rough but
    readable view of what the model actually lost."""
    ref, hyp = normalise(reference), set(normalise(hypothesis))
    return [w for w in ref if w not in hyp]


async def _transcribe(asr, pcm: bytes, name: str,
                      attempts: int = 4) -> tuple[str, str, float]:
    """Transcribe once, retrying only rate limits. Returns (text, error, ms).

    This is an offline benchmark, so waiting out a free-tier 429 is strictly
    better than recording a hole in the comparison. Anything that is not a rate
    limit fails immediately - a retry loop must not hide a real fault.

    The reported latency is the *final* attempt only. Backoff sleeps are the
    free tier's, not the provider's response time, and folding them into the
    number would make a rate-limited run look slow rather than throttled.
    """
    err = ""
    for attempt in range(attempts):
        t0 = time.perf_counter()
        try:
            text = (await asr.transcribe_pcm(pcm, provider=name)).text
            return text, "", (time.perf_counter() - t0) * 1000
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - t0) * 1000
            err = str(exc)
            if "429" not in err[:40] or attempt == attempts - 1:
                return "", err[:400], elapsed
            delay = _retry_delay(err) or min(60.0, 8.0 * 2 ** attempt)
            log("asr_bench.retry", provider=name, attempt=attempt + 1, wait_s=round(delay))
            await asyncio.sleep(delay)
    return "", err[:400], 0.0


def _retry_delay(error_text: str) -> float:
    """Honour the provider's own retryDelay when it sends one."""
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', error_text)
    return float(m.group(1)) + 1.0 if m else 0.0


async def run(providers: list[str] | None = None) -> dict:
    providers = providers or ["groq_turbo", "groq_large", "gemini"]
    tts, asr = get_tts(), get_asr()
    results: list[dict] = []

    for phrase in PHRASES:
        pcm = await tts.synthesize(phrase["text"], phrase["voice"])
        row = {**phrase, "audio_seconds": round(len(pcm) / 32000, 2), "providers": {}}

        for provider in providers:
            model_backup = settings.groq_asr_model
            if provider == "groq_large":
                settings.groq_asr_model = "whisper-large-v3"
            name = "gemini" if provider == "gemini" else "groq"
            try:
                text, err, elapsed = await _transcribe(asr, pcm, name)
            finally:
                settings.groq_asr_model = model_backup

            if err:
                # A provider that never answered has no word error rate. Scoring
                # the empty string would record 100% and average that into the
                # summary as if it were a measurement - it is not, it is a
                # missing one, and the report says so instead.
                entry = {"text": "", "wer": None, "edits": None, "ref_words": None,
                         "latency_ms": round(elapsed), "missed_words": [],
                         "error": err, "ok": False}
            else:
                score, edits, ref_len = wer(phrase["text"], text)
                entry = {"text": text, "wer": round(score, 3), "edits": edits,
                         "ref_words": ref_len, "latency_ms": round(elapsed),
                         "missed_words": diff_words(phrase["text"], text)[:8],
                         "error": "", "ok": True}
            row["providers"][provider] = entry
            log("asr_bench", phrase=phrase["id"], provider=provider,
                wer=entry["wer"], ms=round(elapsed))
            await asyncio.sleep(1.5)  # free-tier rate limits
        results.append(row)

    # Re-running a single provider (a rate-limited one, typically) keeps the
    # other providers' measurements instead of throwing them away.
    providers = _merge_previous(results, providers)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "providers": providers,
        "results": results,
        "summary": _summarise(results, providers),
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "asr_benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload)
    return payload


def _merge_previous(results: list[dict], providers: list[str]) -> list[str]:
    """Fold provider results from the previous run into this one, in place.

    Mutates `results`; returns the provider order for the merged payload, with
    this run's providers first so a re-measured provider stays visible.
    """
    path = EVAL_DIR / "asr_benchmark.json"
    if not path.exists():
        return providers
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return providers
    by_id = {r["id"]: r for r in prev.get("results", [])}
    carried: list[str] = []
    for row in results:
        for name, entry in by_id.get(row["id"], {}).get("providers", {}).items():
            current = row["providers"].get(name)
            # Carry a provider this run did not cover, and keep an earlier
            # successful measurement rather than replacing it with a rate limit.
            if current is None or (not current.get("ok") and entry.get("ok")):
                row["providers"][name] = entry
                if name not in carried:
                    carried.append(name)
    if carried:
        log("asr_bench.merged", carried=",".join(carried))
    # Keep the earlier column order so re-measuring one provider does not
    # reshuffle the report; genuinely new providers go on the end.
    order = [p for p in prev.get("providers", []) if p in providers or p in carried]
    return order + [p for p in providers if p not in order]


def _summarise(results: list[dict], providers: list[str]) -> dict:
    groups = {"India (en-IN)": ["IN"], "Philippines (Taglish)": ["PH"],
              "Indonesia (standard)": ["ID"], "Indonesia (Javanese)": ["JV"],
              "Indonesia (Sundanese)": ["SU"]}
    out: dict = {}
    for label, prefixes in groups.items():
        rows = [r for r in results if any(r["id"].startswith(p) for p in prefixes)]
        if not rows:
            continue
        out[label] = {"phrases": len(rows)}
        for p in providers:
            got = [r["providers"][p] for r in rows if p in r["providers"]]
            ok = [d for d in got if d.get("ok", not d.get("error"))]
            entry = {"n": len(ok), "failed": len(got) - len(ok)}
            if ok:
                entry["mean_wer"] = round(sum(d["wer"] for d in ok) / len(ok), 3)
                entry["mean_latency_ms"] = round(
                    sum(d["latency_ms"] for d in ok) / len(ok))
            out[label][p] = entry
    return out


def _failures(payload: dict) -> dict[str, str]:
    """First error message seen per provider - what to tell the reader instead
    of a number we do not have."""
    seen: dict[str, str] = {}
    for r in payload["results"]:
        for p, d in r["providers"].items():
            if d.get("error") and p not in seen:
                seen[p] = " ".join(d["error"].split())[:120]
    return seen


PROVIDER_LABELS = {
    "groq_turbo": "Groq whisper-large-v3-turbo",
    "groq_large": "Groq whisper-large-v3",
    "gemini": "Gemini audio",
}


def write_report(payload: dict) -> Path:
    lines: list[str] = []
    add = lines.append
    providers = payload["providers"]

    add("# Q3 — ASR benchmark")
    add("")
    add("Generated by `python -m darwix.evaluation.asr_benchmark` on "
        + payload["generated_at"][:19].replace("T", " ") + " UTC.")
    add("")
    add("## Method")
    add("")
    add("Every phrase is synthesised by TTS in a **native voice for its locale**, so the "
        "exact reference text is known and word error rate is a real measurement rather "
        "than an impression. Each phrase targets one specific difficulty — code-switching, "
        "a regional accent, a piece of market vocabulary — rather than being chosen to "
        "flatter the model.")
    add("")
    add("**The honest caveat:** TTS audio is cleaner than a human on a mobile line in "
        "traffic. These numbers are a **best case**, and real-world WER will be higher. "
        "That is why the error analysis below matters more than the headline figure — the "
        "mistakes that survive on clean audio are the ones that will dominate on a real call.")
    add("")

    failures = _failures(payload)
    if failures:
        add("## Provider availability")
        add("")
        add("A provider that returned an error was **not measured**. Those runs are "
            "excluded from every average below rather than scored as 100% word error "
            "rate — a failed API call is a missing measurement, not a bad transcript.")
        add("")
        for p, msg in failures.items():
            failed = sum(s.get(p, {}).get("failed", 0) for s in payload["summary"].values())
            total = failed + sum(s.get(p, {}).get("n", 0) for s in payload["summary"].values())
            add("- **" + PROVIDER_LABELS.get(p, p) + "** — " + str(failed) + "/"
                + str(total) + " phrases failed: `" + msg + "`")
        add("")

    add("## Summary by market")
    add("")
    add("| Market | Phrases | " + " | ".join(
        PROVIDER_LABELS.get(p, p) + " WER" for p in providers) + " |")
    add("|---|---:|" + "---:|" * len(providers))
    for label, per_provider in payload["summary"].items():
        cells = []
        for p in providers:
            stats = per_provider.get(p) or {}
            if stats.get("n"):
                cell = "%.1f%%" % (stats["mean_wer"] * 100)
                if stats.get("failed"):
                    cell += " (n=%d)" % stats["n"]
            else:
                cell = "not measured"
            cells.append(cell)
        add("| " + label + " | " + str(per_provider.get("phrases", 0)) + " | "
            + " | ".join(cells) + " |")
    add("")

    add("## Latency")
    add("")
    add("| Provider | Mean latency | Phrases measured |")
    add("|---|---:|---:|")
    for p in providers:
        stats = [s[p] for s in payload["summary"].values() if s.get(p, {}).get("n")]
        if not stats:
            add("| " + PROVIDER_LABELS.get(p, p) + " | not measured | 0 |")
            continue
        n = sum(d["n"] for d in stats)
        mean = round(sum(d["mean_latency_ms"] * d["n"] for d in stats) / n)
        add("| " + PROVIDER_LABELS.get(p, p) + " | " + str(mean) + " ms | " + str(n) + " |")
    add("")

    add("## Phrase-by-phrase")
    add("")
    for r in payload["results"]:
        add("### " + r["id"] + " — " + r["tests"])
        add("")
        add("Voice: `" + r["voice"] + "` (" + r["locale"] + "), " + str(r["audio_seconds"]) + "s")
        add("")
        add("> **Reference:** " + r["text"])
        add("")
        add("| Provider | WER | Transcript |")
        add("|---|---:|---|")
        for p in providers:
            d = r["providers"].get(p)
            if not d:
                continue
            if d["wer"] is None:
                score, text = "not measured", "_" + (
                    " ".join(d["error"].split())[:80] or "no response") + "_"
            else:
                score = "%.1f%%" % (d["wer"] * 100)
                text = d["text"] or "_(empty)_"
            add("| " + PROVIDER_LABELS.get(p, p) + " | " + score
                + " | " + text.replace("|", "/") + " |")
        add("")
        missed = r["providers"].get(providers[0], {}).get("missed_words") or []
        if missed:
            add("Words lost by the production model: `" + "`, `".join(missed) + "`")
            add("")

    path = EVAL_DIR / "asr_benchmark.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log("asr_bench.report", path=str(path))
    return path


def rerender() -> dict:
    """Rebuild the markdown from the stored run - no API calls. Used when the
    report wording changes, so a presentation fix never costs a re-measurement."""
    payload = json.loads((EVAL_DIR / "asr_benchmark.json").read_text(encoding="utf-8"))
    payload["summary"] = _summarise(payload["results"], payload["providers"])
    (EVAL_DIR / "asr_benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload)
    return payload


def _print_summary(payload: dict) -> None:
    print()
    for label, per in payload["summary"].items():
        cells = []
        for name in payload["providers"]:
            stats = per.get(name) or {}
            cells.append(name + "=" + ("%.1f%%" % (stats["mean_wer"] * 100)
                                       if stats.get("n") else "n/a"))
        print(f"{label:26s} " + ", ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark ASR across locales and providers")
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--rerender", action="store_true",
                    help="rebuild the markdown report from the stored JSON, no API calls")
    args = ap.parse_args()
    _print_summary(rerender() if args.rerender else asyncio.run(run(args.providers)))
