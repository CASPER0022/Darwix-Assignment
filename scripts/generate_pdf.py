"""Generate a polished, executive-ready PDF report for the Darwix AI Assessment submission."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Darwix AI Assessment — Executive Submission Overview</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  @page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-right {
      content: counter(page) " / " counter(pages);
      font-family: 'Inter', sans-serif;
      font-size: 8pt;
      color: #64748b;
    }
  }

  * { box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1e293b;
    line-height: 1.5;
    font-size: 9.5pt;
    margin: 0;
    padding: 0;
    background: #ffffff;
  }

  /* Header / Banner */
  .header-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
    color: #ffffff;
    padding: 22px 24px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
  }
  .header-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(165, 180, 252, 0.35);
    color: #c7d2fe;
    font-size: 7.5pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 9px;
    border-radius: 12px;
    margin-bottom: 8px;
  }
  .header-title {
    font-size: 18pt;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
    color: #ffffff;
  }
  .header-subtitle {
    font-size: 10pt;
    color: #94a3b8;
    margin: 0 0 14px 0;
    font-weight: 400;
  }
  .header-meta-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.12);
    font-size: 8pt;
  }
  .meta-label { color: #94a3b8; font-size: 7.5pt; text-transform: uppercase; letter-spacing: 0.04em; }
  .meta-val { color: #f8fafc; font-weight: 600; margin-top: 2px; }
  .meta-val a { color: #818cf8; text-decoration: none; }

  /* Sections */
  h2 {
    font-size: 12pt;
    font-weight: 700;
    color: #0f172a;
    margin: 20px 0 10px 0;
    padding-bottom: 5px;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 6px;
    page-break-after: avoid;
  }
  h2::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 14px;
    background: #4f46e5;
    border-radius: 2px;
  }
  h3 {
    font-size: 10pt;
    font-weight: 600;
    color: #1e293b;
    margin: 14px 0 6px 0;
    page-break-after: avoid;
  }

  p { margin: 0 0 8px 0; color: #334155; }
  strong { color: #0f172a; font-weight: 600; }

  /* Callout box */
  .callout {
    background: #f8fafc;
    border-left: 3.5px solid #4f46e5;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 10px 0;
    font-size: 9pt;
  }
  .callout-quote {
    font-weight: 600;
    color: #1e1b4b;
    font-style: italic;
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 16px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }
  th {
    background: #f1f5f9;
    color: #334155;
    font-weight: 600;
    text-align: left;
    padding: 7px 9px;
    border: 1px solid #cbd5e1;
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  td {
    padding: 6px 9px;
    border: 1px solid #e2e8f0;
    color: #334155;
    vertical-align: top;
  }
  tr:nth-child(even) td {
    background: #f8fafc;
  }

  .badge-pass {
    display: inline-block;
    background: #dcfce7;
    color: #15803d;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7.5pt;
  }
  .badge-metric {
    display: inline-block;
    background: #e0e7ff;
    color: #3730a3;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7.5pt;
  }

  /* Grid cards */
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 10px 0;
    page-break-inside: avoid;
  }
  .card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 11px 13px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
  }
  .card-title {
    font-size: 9.5pt;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .card-p {
    font-size: 8.5pt;
    color: #475569;
    margin: 0;
    line-height: 1.45;
  }

  /* Code block */
  code {
    font-family: 'JetBrains Mono', monospace;
    background: #f1f5f9;
    color: #0f172a;
    padding: 1.5px 4px;
    border-radius: 3px;
    font-size: 8pt;
  }
  pre {
    background: #0f172a;
    color: #f8fafc;
    padding: 10px 12px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.8pt;
    line-height: 1.45;
    margin: 8px 0;
    overflow-x: hidden;
  }
  pre code {
    background: transparent;
    color: inherit;
    padding: 0;
  }

  .page-break { page-break-before: always; }
  ul { margin: 4px 0 8px 18px; padding: 0; }
  li { margin-bottom: 3px; font-size: 8.5pt; color: #334155; }
</style>
</head>
<body>

<!-- HEADER BANNER -->
<div class="header-card">
  <div class="header-badge">Technical Assessment Submission</div>
  <h1 class="header-title">Darwix AI — AI Engineer Assessment</h1>
  <div class="header-subtitle">Production-Ready Knowledge Base, Grounded Voice Agents & Real-Time Call Intelligence</div>
  <div class="header-meta-grid">
    <div>
      <div class="meta-label">Candidate</div>
      <div class="meta-val">Albin John</div>
    </div>
    <div>
      <div class="meta-label">Email</div>
      <div class="meta-val">heyitsmealbinjohn@gmail.com</div>
    </div>
    <div>
      <div class="meta-label">GitHub Repository</div>
      <div class="meta-val"><a href="https://github.com/CASPER0022/Darwix-Assignment">CASPER0022/Darwix-Assignment</a></div>
    </div>
    <div>
      <div class="meta-label">Test Suite</div>
      <div class="meta-val"><span class="badge-pass">235/235 PASSING</span> (100% Offline)</div>
    </div>
  </div>
</div>

<!-- EXECUTIVE SUMMARY & SCORECARD -->
<h2>1. Executive Summary & Verifiable Results</h2>
<p>
  This submission implements a <strong>unified voice AI and real-time intelligence platform</strong> solving all four deliverables across Indian lending, Southeast Asian financial markets, and live contact-center assistance. Every benchmark number below is produced by automated test harnesses from committed real audio and transcript data.
</p>

<div class="callout">
  <span class="callout-quote">Core Architecture Principle:</span> <strong>"The model decides what the customer meant; deterministic code decides everything else."</strong> Legal disclosures, loan qualification thresholds, numeric grounding, and escalation policies are strictly owned by code with unit tests.
</div>

<table>
  <thead>
    <tr>
      <th>Evaluation Metric</th>
      <th>Measured Result</th>
      <th>Benchmark Source & Evidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Q2 Knowledge Base Retrieval</strong></td>
      <td><span class="badge-pass">14 Correct, 1 Partial, 0 Incorrect</span> (15 queries)</td>
      <td>Pre-declared test set in <code>config/retrieval_tests.yaml</code></td>
    </tr>
    <tr>
      <td><strong>Q4 Live Nudge End-to-End Latency</strong></td>
      <td><span class="badge-metric">854 ms p50</span> / <span class="badge-metric">1,228 ms p95</span></td>
      <td>1.0x wall-clock replay in <code>evaluation/latency_report.md</code></td>
    </tr>
    <tr>
      <td><strong>Q3 Multilingual ASR WER</strong></td>
      <td><strong>8.8%</strong> (en-IN) · <strong>9.5%</strong> (Taglish) · <strong>11.9%</strong> (id-ID) · <strong>25.0%</strong> (Javanese)</td>
      <td>Dialect benchmark in <code>evaluation/asr_benchmark.md</code></td>
    </tr>
    <tr>
      <td><strong>Indexed Knowledge Records</strong></td>
      <td><strong>352 searchable</strong> (39 PII-redacted & quarantined)</td>
      <td>SQLite database + 768d NumPy embeddings</td>
    </tr>
    <tr>
      <td><strong>Q1 Turn Latency (to decision)</strong></td>
      <td><strong>967 ms p50</strong> (ASR 335 ms + Understand/Slots 632 ms)</td>
      <td>Per-turn audit in <code>data/transcripts/sim_q1_*.json</code></td>
    </tr>
    <tr>
      <td><strong>Automated Unit Test Suite</strong></td>
      <td><span class="badge-pass">235 Passing</span> (Zero API keys required)</td>
      <td>Offline pytest suite in <code>tests/</code></td>
    </tr>
  </tbody>
</table>

<!-- DELIVERABLE MATRIX -->
<h2>2. Deliverables & Question-by-Question Overview</h2>

<div class="grid-2">
  <div class="card">
    <div class="card-title">
      <span>Q1: Grounded Voice Agent</span>
      <span class="badge-metric">en-IN NBFC</span>
    </div>
    <div class="card-p">
      <strong>Use Case:</strong> Small business loan qualification for an Indian NBFC.<br>
      <strong>Key Features:</strong> Browser webcall UI, adaptive VAD with 300ms pre-roll, 5-gate anti-hallucination grounding, strict numeric check, code-enforced legal disclosures, CSV rules data engine, CRM lead persistence.
    </div>
  </div>

  <div class="card">
    <div class="card-title">
      <span>Q2: Production Knowledge Base</span>
      <span class="badge-metric">RRF Hybrid</span>
    </div>
    <div class="card-p">
      <strong>Use Case:</strong> Ingestion of real listed NBFC (UGRO Capital) site + policy PDFs.<br>
      <strong>Key Features:</strong> Frequency boilerplate stripping, atomic rule chunking, PII redaction (PAN/Verhoeff Aadhaar), hybrid BM25 + Gemini Dense search fused via Reciprocal Rank Fusion ($k=60$).
    </div>
  </div>

  <div class="card">
    <div class="card-title">
      <span>Q3: Multilingual Voice Bots</span>
      <span class="badge-metric">PH & ID Locales</span>
    </div>
    <div class="card-p">
      <strong>Use Case:</strong> Philippines life insurance (Taglish) & Indonesia multifinance (Bahasa).<br>
      <strong>Key Features:</strong> Decoupled locale packs, Tagalog honorifics (<code>po</code>/<code>opo</code>), payment vocabulary (GCash/Indomaret), deterministic indirect refusal detection (<em>"Nanti saya kabari deh"</em>), ASR WER benchmark.
    </div>
  </div>

  <div class="card">
    <div class="card-title">
      <span>Q4: Live Call Nudges</span>
      <span class="badge-metric">&lt; 1s In-Call</span>
    </div>
    <div class="card-p">
      <strong>Use Case:</strong> Real-time agent assist during live contact center calls.<br>
      <strong>Key Features:</strong> 1.0x wall-clock replay, stereo channel split (Agent L / Customer R), two-tier signals (sub-ms rules + debounced LLMs), nudge cooldowns and duplicate suppression, live WebSocket dashboard.
    </div>
  </div>
</div>

<div class="page-break"></div>

<!-- ARCHITECTURE & HARD ENGINEERING CHALLENGES -->
<h2>3. Key Architecture & Grounding Decisions</h2>

<h3>The 5 Strict Grounding Gates (Anti-Hallucination Pipeline)</h3>
<p>When a customer asks a factual question, the response must pass 5 consecutive checks before the agent speaks:</p>
<ol style="margin: 4px 0 10px 18px; padding: 0; font-size: 8.5pt; color: #334155;">
  <li><strong>Retrieve First:</strong> No factual claim without an indexed retrieval match.</li>
  <li><strong>Confidence Floor:</strong> If top RRF score &lt; 0.35, the agent immediately admits ignorance and offers a human.</li>
  <li><strong>Explicit Model Decline:</strong> The LLM is explicitly allowed and instructed to return <code>false</code>.</li>
  <li><strong>Citation Verification:</strong> All returned <code>record_id</code>s must exist in the retrieved chunks.</li>
  <li><strong>Strict Numeric Guard:</strong> <em>Every number in the generated response must appear verbatim in the source chunk</em>, explicitly preventing loan rate smearing (e.g. turning "14.5% to 16.0%" into "around 15%").</li>
</ol>

<h3>Why No Vector Database?</h3>
<p>
  With 352 searchable records, brute-force cosine similarity over a 768-dimensional NumPy float32 matrix executes in <strong>&lt; 0.2 ms</strong>—three orders of magnitude faster than an ASR API call. Introducing external vector databases (Chroma/Pinecone) would add unnecessary operational overhead with zero latency benefit.
</p>

<!-- HARD BUGS SOLVED -->
<h2>4. Critical Engineering Hurdles Solved</h2>

<table>
  <thead>
    <tr>
      <th>Challenge Discovered</th>
      <th>Root Cause Traced</th>
      <th>Robust Engineering Solution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Whisper Hallucinating "Thank You." on Silence</strong></td>
      <td>In quiet rooms, the adaptive noise floor decayed from 0.0229 to 0.0012. Soft breathing triggered VAD; Whisper transcribes near-silence into subtitle training artifacts.</td>
      <td>1) Clamped noise floor bounds.<br>2) Added absolute audio energy gate.<br>3) Subtitle artifact filter across 10 languages.</td>
    </tr>
    <tr>
      <td><strong>Noise Floor Latching in Loud Rooms</strong></td>
      <td>In loud environments (&gt; 0.013 RMS), continuous tone was classified as speech, freezing the adapter and locking VAD permanently open.</td>
      <td>Rewrote noise estimation to sample a low percentile over a rolling 15-second sliding window.</td>
    </tr>
    <tr>
      <td><strong>Acoustic Feedback / Self-Talk</strong></td>
      <td>Microphone picked up synthesized agent audio from speakers, creating a self-loop.</td>
      <td>Elevated VAD threshold dynamically during agent audio playback. Quiet bleed is ignored; loud user barge-in is preserved.</td>
    </tr>
    <tr>
      <td><strong>Indonesian Indirect Refusals</strong></td>
      <td>Polite phrasing (<em>"Nanti saya kabari deh"</em> — "I'll let you know later") was heard by LLMs as a promise to pay, corrupting collections queues.</td>
      <td>Engine detects cultural indirect refusal patterns deterministically in code.</td>
    </tr>
  </tbody>
</table>

<!-- HOW TO TEST LOCALLY -->
<h2>5. Quick Start & Local Verification</h2>

<p>The entire system is self-contained with no external dependencies required for testing:</p>

<pre><code># 1. Setup & install (Python 3.11+)
git clone https://github.com/CASPER0022/Darwix-Assignment.git
cd Darwix-Assignment && pip install -e ".[scrape,dev]"

# 2. Run the 235 unit tests (Zero API keys required, 100% offline)
pytest

# 3. Launch interactive web applications
python -m darwix.server.app
# Open http://localhost:8000 in your browser:
# -> /webcall   : Interactive voice agent with live speech & citation inspector
# -> /kb        : Knowledge base search with BM25/Dense scoring breakdown
# -> /dashboard : Live agent assist dashboard with real-time audio playback</code></pre>

<!-- FOOTER -->
<div style="margin-top: 24px; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 8pt; color: #64748b; display: flex; justify-content: space-between;">
  <span>Darwix AI Engineer Assessment — Submission Package</span>
  <span>Repository: github.com/CASPER0022/Darwix-Assignment</span>
</div>

</body>
</html>
"""

async def generate():
    out_pdf = Path("Darwix_AI_Assessment_Submission_Overview.pdf")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(HTML_CONTENT, wait_until="networkidle")
        await page.pdf(
            path=str(out_pdf),
            format="A4",
            print_background=True,
            margin={"top": "14mm", "bottom": "14mm", "left": "14mm", "right": "14mm"}
        )
        await browser.close()
    print(f"Successfully generated PDF: {out_pdf.absolute()} ({out_pdf.stat().st_size} bytes)")

if __name__ == "__main__":
    asyncio.run(generate())
