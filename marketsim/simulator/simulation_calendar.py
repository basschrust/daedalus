from datetime import date


class SimulationCalendar:
    def __init__(self, dates: list[date]):
        if not dates:
            raise ValueError("Simulation calendar cannot be empty")

        if dates != sorted(set(dates)):
            raise ValueError(
                "Trading dates must be unique and chronological"
            )

        self.dates = dates
        self._days = {
            trading_date: day
            for day, trading_date in enumerate(dates)
        }

    def to_date(self, day: int) -> date:
        return self.dates[day]

    def to_day(self, trading_date: date) -> int:
        return self._days[trading_date]

    def __len__(self) -> int:
        return len(self.dates)
