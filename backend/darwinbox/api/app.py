"""FastAPI application wiring.

Everything here is assembly: build the store, mount the routers, install the error
handlers. No business logic passes through this file.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from darwinbox import samples as sample_sets
from darwinbox.api import errors, routes_query, routes_schema, routes_sessions
from darwinbox.llm.client import client_from_env
from darwinbox.models import SampleSet
from darwinbox.session import SessionStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# The hosted frontend is served from Firebase while the API stays on the machine with
# the GPU, so those origins are cross-site and must be allowed explicitly.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://darwinbox-assignment-app.web.app",
    "https://darwinbox-assignment-app.firebaseapp.com",
]


def create_app(client_factory=client_from_env) -> FastAPI:
    """Build the app. Tests pass a factory returning FakeLLMClient, so no GPU is needed."""
    app = FastAPI(
        title="Darwinbox Assignment",
        version="0.1.0",
        description="Ask plain-English questions across several messy CSV/Excel files.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.store = SessionStore(client_factory)

    app.include_router(routes_sessions.router)
    app.include_router(routes_schema.router)
    app.include_router(routes_query.router)
    errors.register(app)

    @app.get("/api/samples", tags=["meta"], response_model=list[SampleSet])
    def list_samples() -> list[SampleSet]:
        """The bundled datasets a session can be pre-loaded with."""
        return sample_sets.SAMPLES

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "sessions": len(app.state.store)}

    return app


app = create_app()
