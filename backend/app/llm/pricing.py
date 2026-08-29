"""Token prices, and the cost of one call.

Plan section 14.2: "maintain a ``PRICES`` table in code, compute ``cost_usd`` on
every call".  Cost per interview is a headline number for this project, and a
number you cannot compute is a number you cannot put on a resume.

Two rules this module exists to enforce:

* **Decimal, not float.**  Fractions of a cent summed over thousands of calls is
  precisely the arithmetic binary floating point gets wrong.
* **Unknown is not free.**  A model missing from the table reports
  ``price_known=False`` alongside a 0.0 cost.  Silently charging $0 for a paid
  model is how a cost dashboard reads perfect right up until the invoice.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

#: Prices are quoted per million tokens, which is how every vendor publishes them.
_PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class ModelPrice:
    usd_per_million_input: Decimal
    usd_per_million_output: Decimal
    note: str = ""


@dataclass(frozen=True, slots=True)
class CallCost:
    usd: Decimal
    price_known: bool


#: Keyed by the provider's own model id.
PRICES: dict[str, ModelPrice] = {
    # NVIDIA's hosted evaluation endpoint (integrate.api.nvidia.com) is served
    # from developer credits rather than metered per-token billing, so the
    # marginal cost of a call is genuinely zero.  Recorded explicitly, with the
    # reason, so that "cost was $0" is a documented fact and not a missing row.
    "nvidia/nemotron-3.5-lightning-30b-a3b": ModelPrice(
        usd_per_million_input=Decimal("0"),
        usd_per_million_output=Decimal("0"),
        note="NVIDIA hosted evaluation endpoint - credit-metered, not per-token billed",
    ),
    # The Day-5 offline provider. Listed explicitly so that a stubbed call
    # reports `price_known=True, cost=0` - "free because nothing was bought" -
    # rather than `price_known=False`, which means "we do not know what this
    # cost". Those are different statements and a cost report must not blur
    # them; it is also what lets an offline test exercise the real cost
    # arithmetic instead of skipping it.
    "stub/deterministic-v1": ModelPrice(
        usd_per_million_input=Decimal("0"),
        usd_per_million_output=Decimal("0"),
        note="offline replay provider - no request is made, so nothing is charged",
    ),
}


def price_call(*, model: str, input_tokens: int, output_tokens: int) -> CallCost:
    """Cost of one completion.

    Reasoning tokens are not a separate line item: providers bill them as output
    tokens and report them inside ``output_tokens``, so counting them again here
    would double-charge.
    """
    price = PRICES.get(model)
    if price is None:
        return CallCost(usd=Decimal("0"), price_known=False)

    usd = (
        Decimal(input_tokens) * price.usd_per_million_input
        + Decimal(output_tokens) * price.usd_per_million_output
    ) / _PER_MILLION
    # Six places: the plan's per-interview figures live in the fourth decimal,
    # so anything coarser rounds the interesting digits away.
    return CallCost(usd=usd.quantize(Decimal("0.000001")), price_known=True)
