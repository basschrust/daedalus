from decimal import ROUND_HALF_UP, Decimal

# TODO: check precision, currently is 28 (default) print(getcontext().prec)


def D(value):
    return Decimal(str(value))


class Price(Decimal):
    TICK_SIZE = Decimal("0.01")  # TODO: make it configurable on Market level

    def __new__(cls, value: Decimal | float | int):

        d = Decimal(str(value))
        if not d.is_finite():
            raise ValueError("Price must be finite")

        # Optional sanity check
        if abs(d) > Decimal("100_000_000"):
            raise ValueError(f"Unreasonable price: {value}")

        # TODO: this slows doowwwwwwnnnnn....   print(f"d: {d}")
        return super().__new__(cls, d.quantize(cls.TICK_SIZE, rounding=ROUND_HALF_UP))

    def __mul__(self, other):
        if isinstance(other, float):
            other = Decimal(str(other))
        result = Decimal(super().__mul__(other))
        return Price(result)

    def __rmul__(self, other):
        if isinstance(other, float):
            other = Decimal(str(other))

        return Price(other * Decimal(self))
