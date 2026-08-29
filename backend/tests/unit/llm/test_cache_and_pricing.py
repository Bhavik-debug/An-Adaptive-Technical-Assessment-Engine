"""The cache key, and the cost arithmetic.

Both are small enough to test exhaustively, and both are the kind of code where
a quiet mistake is invisible until it has been wrong for a month.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.llm.cache import CachedCall, NullCache, RedisResponseCache, cache_key
from app.llm.pricing import PRICES, ModelPrice, price_call

BASE = dict(
    task="grade_answer",
    prompt_version="v7",
    prompt_fingerprint="abc123def456",
    schema_fingerprint="0011223344556677",
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.0,
    top_p=1.0,
    inputs={"answer": "a cache is a store of recent results"},
)


# --- the key ---------------------------------------------------------------


def test_the_same_request_always_hashes_the_same():
    assert cache_key(**BASE) == cache_key(**BASE)


def test_input_key_order_does_not_change_the_hash():
    """``{"a":1,"b":2}`` and ``{"b":2,"a":1}`` are the same request."""
    a = cache_key(**{**BASE, "inputs": {"a": 1, "b": 2}})
    b = cache_key(**{**BASE, "inputs": {"b": 2, "a": 1}})
    assert a == b


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "follow_up_probe"),
        ("prompt_version", "v8"),
        ("prompt_fingerprint", "ffffffffffff"),
        ("schema_fingerprint", "ffffffffffffffff"),
        ("model", "some/other-model"),
        ("temperature", 0.4),
        ("top_p", 0.95),
        ("inputs", {"answer": "something else"}),
    ],
)
def test_anything_that_could_change_the_answer_changes_the_key(field, value):
    assert cache_key(**{**BASE, field: value}) != cache_key(**BASE)


def test_an_edited_prompt_invalidates_the_cache_even_without_a_version_bump():
    """The mistake worth catching: a prompt changed, its label was not."""
    edited = cache_key(**{**BASE, "prompt_fingerprint": "999999999999"})
    assert edited != cache_key(**BASE)


def test_keys_are_namespaced_so_they_can_be_flushed_selectively():
    assert cache_key(**BASE).startswith("llm:v1:")


# --- the backends ----------------------------------------------------------


async def test_null_cache_never_hits():
    cache = NullCache()
    await cache.set("k", CachedCall(payload={"a": 1}, provider="p", model="m"))
    assert await cache.get("k") is None


class FakeRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.fail = fail
        self.ttls: dict[str, int | None] = {}

    async def get(self, key: str) -> Any:
        if self.fail:
            raise RedisConnectionError("redis is down")
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self.fail:
            raise RedisConnectionError("redis is down")
        self.store[key] = value
        self.ttls[key] = ex


async def test_redis_cache_round_trips_and_applies_a_ttl():
    redis = FakeRedis()
    cache = RedisResponseCache(redis, ttl_s=3600)  # type: ignore[arg-type]

    await cache.set("k", CachedCall(payload={"a": 1}, provider="nvidia", model="m"))
    hit = await cache.get("k")

    assert hit is not None
    assert hit.payload == {"a": 1}
    assert hit.provider == "nvidia"
    assert redis.ttls["k"] == 3600


async def test_a_dead_redis_is_a_miss_not_a_failure():
    """Losing the cache must make the system slower, never broken."""
    cache = RedisResponseCache(FakeRedis(fail=True), ttl_s=60)  # type: ignore[arg-type]

    assert await cache.get("k") is None
    await cache.set("k", CachedCall(payload={"a": 1}, provider="p", model="m"))  # no raise


async def test_an_unreadable_entry_is_discarded():
    redis = FakeRedis()
    redis.store["k"] = "not json at all"
    cache = RedisResponseCache(redis, ttl_s=60)  # type: ignore[arg-type]
    assert await cache.get("k") is None


async def test_an_entry_from_an_older_shape_is_discarded():
    redis = FakeRedis()
    redis.store["k"] = json.dumps({"payload": {"a": 1}})  # no provider/model
    cache = RedisResponseCache(redis, ttl_s=60)  # type: ignore[arg-type]
    assert await cache.get("k") is None


# --- pricing ---------------------------------------------------------------


def test_the_selected_model_is_in_the_price_table():
    """An unpriced production model is a cost dashboard that reads $0 forever."""
    assert "nvidia/nemotron-3.5-lightning-30b-a3b" in PRICES


def test_the_nvidia_endpoint_is_recorded_as_free_with_a_reason():
    price = PRICES["nvidia/nemotron-3.5-lightning-30b-a3b"]
    assert price.usd_per_million_input == Decimal("0")
    assert price.note  # the reason it is zero, not an empty row


def test_cost_is_computed_per_million_tokens(monkeypatch):
    monkeypatch.setitem(
        PRICES,
        "test/priced-model",
        ModelPrice(usd_per_million_input=Decimal("0.30"), usd_per_million_output=Decimal("2.50")),
    )
    cost = price_call(model="test/priced-model", input_tokens=1_450, output_tokens=390)

    # 1450 * 0.30/1e6 + 390 * 2.50/1e6 = 0.000435 + 0.000975
    assert cost.usd == Decimal("0.001410")
    assert cost.price_known is True


def test_cost_uses_decimal_not_float(monkeypatch):
    """Fractions of a cent summed over thousands of calls is where floats lie."""
    monkeypatch.setitem(
        PRICES,
        "test/priced-model",
        ModelPrice(usd_per_million_input=Decimal("0.10"), usd_per_million_output=Decimal("0.10")),
    )
    total = sum(
        price_call(model="test/priced-model", input_tokens=1, output_tokens=0).usd
        for _ in range(10)
    )
    assert isinstance(total, Decimal)


def test_an_unknown_model_is_flagged_rather_than_reported_as_free():
    cost = price_call(model="some/model-nobody-priced", input_tokens=1000, output_tokens=1000)
    assert cost.usd == Decimal("0")
    assert cost.price_known is False


def test_zero_tokens_costs_zero():
    cost = price_call(
        model="nvidia/nemotron-3.5-lightning-30b-a3b", input_tokens=0, output_tokens=0
    )
    assert cost.usd == Decimal("0.000000")
    assert cost.price_known is True
