"""FastAPI middleware: rate limiting, correlation ID, structured logging."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Rate limit store: ip -> list of request timestamps
_rate_store: defaultdict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 200
RATE_LIMIT_WINDOW_SECONDS = 60


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request and response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = corr_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Correlation-ID"] = corr_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
        logger.info(
            "method=%s path=%s status=%d latency_ms=%.2f corr_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            corr_id,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter per client IP."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Health check exempt
        if request.url.path in ("/health", "/api/v1/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SECONDS

        timestamps = _rate_store[client_ip]
        timestamps[:] = [t for t in timestamps if t > window_start]

        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            logger.warning("Rate limit exceeded for IP=%s", client_ip)
            return Response(
                content='{"detail":"Rate limit exceeded. Max 200 requests/minute."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        timestamps.append(now)
        return await call_next(request)
