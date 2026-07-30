"""SearXNG search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Same JSON
API call (``/search?format=json``), same result normalization. The legacy
in-tree module ``tools.web_providers.searxng`` was removed in the same
commit that moved this code under ``plugins/``; this file is now the
canonical implementation.

Search-only — SearXNG aggregates results from upstream engines but does not
fetch/extract arbitrary URLs. ``supports_extract()`` returns False.

Config keys this provider responds to::

    web:
      search_backend: "searxng"     # explicit per-capability
      backend: "searxng"            # shared fallback

Env var::

    SEARXNG_URL=http://localhost:8080
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _searxng_url() -> str:
    """Return SEARXNG_URL from Hermes config-aware env, falling back to process env."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value("SEARXNG_URL")
    except Exception:
        val = None
    if val is None:
        val = os.getenv("SEARXNG_URL", "")
    return (val or "").strip()


class SearXNGWebSearchProvider(WebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def display_name(self) -> str:
        return "SearXNG"

    def is_available(self) -> bool:
        """Return True when ``SEARXNG_URL`` is set."""
        return bool(_searxng_url())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the configured SearXNG instance."""
        import httpx

        base_url = _searxng_url().rstrip("/")
        if not base_url:
            return {"success": False, "error": "SEARXNG_URL is not set"}

        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }

        try:
            resp = httpx.get(
                f"{base_url}/search",
                params=params,
                timeout=15,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"SearXNG returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach SearXNG at {base_url}: {exc}",
            }

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SearXNG response parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse SearXNG response as JSON",
            }

        if not isinstance(data, dict):
            logger.warning("SearXNG returned a non-object JSON response")
            return {
                "success": False,
                "error": "SearXNG returned an invalid JSON response",
            }

        if "results" not in data or data["results"] is None:
            logger.warning("SearXNG response is missing its results field")
            return {
                "success": False,
                "error": "SearXNG response is missing a results list",
            }

        raw_results = data["results"]
        if not isinstance(raw_results, list):
            logger.warning("SearXNG returned a non-list results field")
            return {
                "success": False,
                "error": "SearXNG returned an invalid results field",
            }

        unresponsive_engines = data.get("unresponsive_engines") or []
        if not isinstance(unresponsive_engines, list):
            unresponsive_engines = [unresponsive_engines]

        diagnostics = {
            "number_of_results": data.get("number_of_results"),
            "unresponsive_engines": unresponsive_engines,
        }

        # SearXNG deliberately returns HTTP 200 when one or more upstream
        # engines fail. Do not let an empty aggregate look like a successful
        # search in that case: the caller otherwise has no way to distinguish
        # "no match" from "the configured engine pool is broken".
        if not raw_results and unresponsive_engines:
            failures = []
            for entry in unresponsive_engines:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    failures.append(f"{entry[0]} ({entry[1]})")
                else:
                    failures.append(str(entry))
            failure_text = ", ".join(failures)
            logger.warning(
                "SearXNG returned no results; unresponsive engines: %s",
                failure_text,
            )
            return {
                "success": False,
                "error": (
                    "SearXNG returned no results because upstream engine(s) "
                    f"are unavailable: {failure_text}. "
                    "Check the SearXNG engine configuration or retry."
                ),
                "diagnostics": diagnostics,
            }

        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(
            raw_results,
            key=lambda r: float(r.get("score", 0)),
            reverse=True,
        )[:limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "")),
                "position": i + 1,
            }
            for i, r in enumerate(sorted_results)
        ]

        logger.info(
            "SearXNG search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        response: Dict[str, Any] = {
            "success": True,
            "data": {"web": web_results},
        }
        # Preserve partial-engine diagnostics without turning a usable result
        # set into a failure. This is useful when an instance still returns
        # results but has a degraded upstream pool.
        if unresponsive_engines:
            response["diagnostics"] = diagnostics
        return response

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "SearXNG",
            "badge": "free · self-hosted",
            "tag": "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "env_vars": [
                {
                    "key": "SEARXNG_URL",
                    "prompt": "SearXNG instance URL (e.g. http://localhost:8080)",
                    "url": "https://searx.space/",
                },
            ],
        }
