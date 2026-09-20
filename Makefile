PY := .venv/bin/python
PIP := .venv/bin/pip
MODEL := qwen2.5-coder:7b-instruct-q4_K_M

.PHONY: setup fixtures dev test eval external smoke lint api web clean tunnel deploy deploy-web

setup:
	python3.11 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"
	cd web && npm install
	ollama pull $(MODEL)

fixtures:
	$(PY) eval/corrupt.py --out eval/fixtures
	$(PY) eval/make_genres.py

dev:
	@echo "api on :8000, web on :5173"
	@trap 'kill 0' EXIT; \
	$(PY) -m uvicorn darwinbox.api.app:app --app-dir backend --reload --port 8000 & \
	cd web && npm run dev & \
	wait

api:
	$(PY) -m uvicorn darwinbox.api.app:app --app-dir backend --reload --port 8000

web:
	cd web && npm run dev

test:
	$(PY) -m pytest -q
	cd web && npm test

lint:
	.venv/bin/ruff check backend eval

eval:
	$(PY) eval/score_genres.py
	$(PY) eval/run_eval.py --stage ingest
	$(PY) eval/run_eval.py --stage relationships
	$(PY) eval/run_eval.py --stage qa --repeat 3
	$(PY) eval/run_eval.py --stage qa --holdout --repeat 3

# The external check: real open data, questions and answers derived from the files
# rather than written by me. Needs network once to build the corpus.
external:
	$(PY) eval/fetch_gov.py --wide
	$(PY) eval/make_external_questions.py
	$(PY) eval/run_eval.py --stage qa --external --repeat 3

# Drives the real page in a browser. Needs `make dev` running in another terminal.
smoke:
	cd web && npx playwright install chromium
	node web/e2e/smoke.mjs

clean:
	rm -rf sessions .pytest_cache .ruff_cache

# Expose the local dev server so the hosted frontend can reach this machine.
# cloudflared over localtunnel: the latter dropped repeatedly and shows an
# interstitial that every cross-origin request has to work around.
tunnel:
	cloudflared tunnel --url http://localhost:5173 --no-autoupdate

# --- Hosting -------------------------------------------------------------- #
#
# Two deployments. The frontend is static and goes to Firebase. The API needs a
# process, so it runs on Render with DARWINBOX_LLM=gemini -- there is no GPU
# there, and the LLMClient protocol is what makes that a configuration change
# rather than a rewrite.
#
# Nothing in the hosted path can be charged for, which was a requirement rather
# than a nicety. Each piece refuses when it runs out instead of billing:
#
#   Render, free plan    suspends at 750 instance-hours a month
#   Firebase Hosting     the Spark plan, no billing account attached
#   Gemini 2.5 Flash     the AI Studio free tier, rate-limited not billed
#
# That distinction is why this is not Cloud Run. Its free tier is the most
# generous of the three, but the project behind it has billing enabled, so going
# past the tier is charged rather than refused -- and "free unless you exceed it"
# is not the same guarantee.
#
# The API is deployed from render.yaml by connecting the repo once in Render's
# dashboard; there is no CLI step to put here. Get the Gemini key from
# https://aistudio.google.com/apikey in a project with no billing account, so the
# free tier is not merely the cheapest path but the only one available to it.

# Publish the frontend against a given API. API_BASE is the Render URL (or a tunnel).
deploy-web deploy:
	cd web && VITE_API_BASE=$(API_BASE) npm run build
	firebase deploy --only hosting --project darwinbox-assignment-app
