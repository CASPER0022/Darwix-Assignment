# Darwix AI assessment - everything runnable from one place.
# Windows: use `make` from Git Bash, or run the python commands directly.

PY ?= python

.PHONY: help install setup kb kb-offline crawl eval serve sim sim-q1 sim-q3 asr-bench q4 live q4-eval test all check

help:
	@echo "Setup"
	@echo "  make install    install dependencies (editable) + playwright chromium"
	@echo "  make setup      seed synthetic source documents"
	@echo ""
	@echo "Knowledge base (Q2)"
	@echo "  make crawl      re-crawl the source website + policy PDFs (needs network)"
	@echo "  make kb         clean -> build -> index the knowledge base"
	@echo "  make kb-offline rebuild with no API key (lexical retrieval only)"
	@echo "  make eval       run the retrieval test set -> evaluation/retrieval_tests.md"
	@echo ""
	@echo "Voice agent (Q1, Q3)"
	@echo "  make serve      start the web call interface + KB search + dashboard"
	@echo "  make sim        run every scripted test call (records audio + transcripts)"
	@echo "  make sim-q1     the five Q1 scenarios only"
	@echo "  make sim-q3     the four Q3 scenarios only"
	@echo "  make asr-bench  ASR provider comparison -> evaluation/asr_benchmark.md"
	@echo ""
	@echo "Live nudges (Q4)"
	@echo "  make q4         render the Q4 scenario recordings"
	@echo "  make live FILE=data/recordings/q4_compliance_gap.wav"
	@echo "  make q4-eval    run all Q4 scenarios in real time -> latency + FP reports"
	@echo ""
	@echo "  make test       unit tests (no API key needed)"
	@echo "  make check      tests + KB rebuild + retrieval eval"

install:
	$(PY) -m pip install -e ".[scrape,dev]"
	$(PY) -m playwright install chromium

setup:
	$(PY) -m darwix.kb.ingest.seed_internal_docs
	$(PY) -m darwix.kb.ingest.seed_market_docs

crawl:
	$(PY) -m darwix.kb.ingest.web
	$(PY) -m darwix.kb.ingest.web --discover
	$(PY) -m darwix.kb.ingest.web --pdfs

kb:
	$(PY) -m darwix.kb.clean
	$(PY) -m darwix.kb.build
	$(PY) -m darwix.kb.index

# Rebuild with no API key at all: no LLM structuring, no embeddings.
kb-offline:
	$(PY) -m darwix.kb.clean
	$(PY) -m darwix.kb.build --no-llm
	$(PY) -m darwix.kb.index --no-embeddings

eval:
	$(PY) -m darwix.kb.evaluate

serve:
	$(PY) -m darwix.server.app

sim:
	$(PY) -m darwix.simulator.caller

sim-q1:
	$(PY) -m darwix.simulator.caller q1_cooperative q1_objection q1_conflicting q1_out_of_scope q1_human_request

sim-q3:
	$(PY) -m darwix.simulator.caller q3_ph_cooperative q3_ph_objection q3_id_colloquial q3_id_javanese

q4:
	$(PY) -m darwix.simulator.scenario_recorder

FILE ?= data/recordings/q4_compliance_gap.wav
live:
	$(PY) -m darwix.realtime.pipeline "$(FILE)"

q4-eval:
	$(PY) -m darwix.realtime.evaluate

asr-bench:
	$(PY) -m darwix.evaluation.asr_benchmark

test:
	$(PY) -m pytest tests/ -q

check: test kb eval
	@echo "all checks complete"

all: setup kb eval q4 sim q4-eval
	@echo "full pipeline complete"
