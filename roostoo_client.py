"""
Roostoo REST API client.

Handles request signing (HMAC-SHA256 / RCL_TopLevelCheck) and exposes
typed helpers for every endpoint described in the Roostoo API docs.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any, Optional
from urllib.parse import urlencode

import requests

import config

logger = logging.getLogger(__name__)


# ── Low-level helpers ──────────────────────────────────────────────────────────


def _timestamp() -> str:
    """Return the current UTC time as a 13-digit millisecond timestamp string."""
    return str(int(time.time() * 1000))


def _sign(payload: dict[str, Any], secret: str) -> tuple[dict[str, str], str]:
    """
    Build the HMAC-SHA256 signature required for RCL_TopLevelCheck endpoints.

    The canonical string is built by sorting parameter keys, joining each
    key=value pair with '&', then computing HMAC-SHA256 over the result
    using *secret* as the key.

    Returns
    -------
    headers : dict
        HTTP headers to include in the request (``RST-API-KEY``,
        ``MSG-SIGNATURE``).
    total_params : str
        The urlencoded query / body string (to be sent verbatim).
    """
    payload["timestamp"] = _timestamp()
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted(payload))

    signature = hmac.new(
        secret.encode("utf-8"),
        total_params.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "RST-API-KEY": config.API_KEY,
        "MSG-SIGNATURE": signature,
    }
    return headers, total_params


# ── Public API client class ────────────────────────────────────────────────────


class RoostooClient:
    """Thin wrapper around the Roostoo mock-exchange REST API."""

    def __init__(
        self,
        api_key: str = config.API_KEY,
        secret_key: str = config.SECRET_KEY,
        base_url: str = config.BASE_URL,
        timeout: int = 10,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._timeout = timeout

    # ── Internal request helpers ───────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        url = f"{self._base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("GET %s failed: %s", path, exc)
            return None

    def _signed_get(self, path: str, payload: Optional[dict] = None) -> Optional[dict]:
        if payload is None:
            payload = {}
        headers, total_params = _sign(payload, self._secret_key)
        url = f"{self._base_url}{path}"
        try:
            resp = self._session.get(
                url,
                headers=headers,
                params=dict(pair.split("=") for pair in total_params.split("&")),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Signed GET %s failed: %s", path, exc)
            return None

    def _signed_post(self, path: str, payload: Optional[dict] = None) -> Optional[dict]:
        if payload is None:
            payload = {}
        headers, total_params = _sign(payload, self._secret_key)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        url = f"{self._base_url}{path}"
        try:
            resp = self._session.post(
                url,
                headers=headers,
                data=total_params,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Signed POST %s failed: %s", path, exc)
            return None

    # ── Public endpoints ───────────────────────────────────────────────────────

    def get_server_time(self) -> Optional[dict]:
        """GET /v3/serverTime – unauthenticated connectivity check."""
        return self._get("/v3/serverTime")

    def get_exchange_info(self) -> Optional[dict]:
        """GET /v3/exchangeInfo – trading pairs and rules."""
        return self._get("/v3/exchangeInfo")

    def get_ticker(self, pair: Optional[str] = None) -> Optional[dict]:
        """GET /v3/ticker – market ticker for one or all pairs.

        Parameters
        ----------
        pair:
            If supplied (e.g. ``"BTC/USD"``), returns data for that pair only.
            If omitted, returns all pairs.
        """
        params: dict[str, Any] = {"timestamp": _timestamp()}
        if pair:
            params["pair"] = pair
        return self._get("/v3/ticker", params=params)

    # ── Authenticated endpoints ────────────────────────────────────────────────

    def get_balance(self) -> Optional[dict]:
        """GET /v3/balance – wallet balances (requires API key)."""
        return self._signed_get("/v3/balance")

    def get_pending_count(self) -> Optional[dict]:
        """GET /v3/pending_count – number of open (pending) orders."""
        return self._signed_get("/v3/pending_count")

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> Optional[dict]:
        """POST /v3/place_order – submit a new order.

        Parameters
        ----------
        pair:
            Trading pair, e.g. ``"BTC/USD"``.
        side:
            ``"BUY"`` or ``"SELL"``.
        quantity:
            Amount of the base coin to trade.
        order_type:
            ``"MARKET"`` (default) or ``"LIMIT"``.
        price:
            Required for LIMIT orders; ignored for MARKET orders.
        """
        order_type = order_type.upper()
        side = side.upper()

        if order_type == "LIMIT" and price is None:
            logger.error("LIMIT orders require a price.")
            return None

        payload: dict[str, Any] = {
            "pair": pair,
            "side": side,
            "type": order_type,
            "quantity": str(quantity),
        }
        if order_type == "LIMIT" and price is not None:
            payload["price"] = str(price)

        return self._signed_post("/v3/place_order", payload)

    def query_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None,
        pending_only: Optional[bool] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Optional[dict]:
        """POST /v3/query_order – retrieve order history or pending orders."""
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        else:
            if pair:
                payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if offset is not None:
                payload["offset"] = str(offset)
            if limit is not None:
                payload["limit"] = str(limit)
        return self._signed_post("/v3/query_order", payload)

    def cancel_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None,
    ) -> Optional[dict]:
        """POST /v3/cancel_order – cancel pending order(s).

        Omitting both *order_id* and *pair* cancels **all** pending orders.
        """
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
        return self._signed_post("/v3/cancel_order", payload)
