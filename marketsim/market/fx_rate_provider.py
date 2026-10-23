from __future__ import annotations

from decimal import Decimal
from typing import Mapping


class FXRateProvider:
    """Day-indexed FX rates: one unit of base buys `rate` units of quote.

    For example, set_rate(0, 'USD', 'PLN', '3.80') means 1 USD = 3.80 PLN.
    Rates must be positive and finite. No implicit carry-forward or triangulation.
    """

    def __init__(
        self, rates: Mapping[tuple[int, str, str], Decimal | str | int | float] | None = None
    ):
        self._rates: dict[tuple[int, str, str], Decimal] = {}
        for (day, base, quote), rate in (rates or {}).items():
            self.set_rate(day=day, base=base, quote=quote, rate=rate)

    @staticmethod
    def _currency(value: str) -> str:
        if not isinstance(value, str) or len(value.strip()) != 3 or not value.strip().isalpha():
            raise ValueError(f"Invalid currency code: {value!r}")
        return value.strip().upper()

    def set_rate(
        self, *, day: int, base: str, quote: str, rate: Decimal | str | int | float
    ) -> None:
        if not isinstance(day, int) or isinstance(day, bool) or day < 0:
            raise ValueError("day must be a nonnegative integer")
        base, quote = self._currency(base), self._currency(quote)
        value = Decimal(str(rate))
        if not value.is_finite() or value <= 0:
            raise ValueError("FX rate must be positive and finite")
        if base == quote and value != 1:
            raise ValueError("Same-currency FX rate must be 1")
        self._rates[(day, base, quote)] = value

    def get_rate(self, *, day: int, base: str, quote: str) -> Decimal:
        base = self._currency(base)
        quote = self._currency(quote)

        if base == quote:
            return Decimal(1)

        # 1. Direct rate
        direct = self._rates.get((day, base, quote))
        if direct is not None:
            return direct

        # 2. Inverse rate
        inverse = self._rates.get((day, quote, base))
        if inverse is not None:
            return Decimal(1) / inverse

        # 3. Triangulation through PLN
        pivot = "PLN"

        if base != pivot and quote != pivot:
            base_to_pivot = self.get_rate(
                day=day,
                base=base,
                quote=pivot,
            )

            quote_to_pivot = self.get_rate(
                day=day,
                base=quote,
                quote=pivot,
            )

            return base_to_pivot / quote_to_pivot

        raise KeyError(f"No FX rate for {base}/{quote} on day {day}")

    def convert(
        self, amount: Decimal | str | int | float, *, day: int, base: str, quote: str
    ) -> Decimal:
        value = Decimal(str(amount))
        if not value.is_finite():
            raise ValueError("Amount must be finite")
        return value * self.get_rate(day=day, base=base, quote=quote)
