from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from marketsim.fourheap.order import Order


class EventQueue:
    def __init__(self, rand_seed: int = None):
        self.rand = random.Random(rand_seed)
        self.scheduled_activities = defaultdict(list)

    def schedule_activity(self, order: Order):
        t = order.time
        self.scheduled_activities[t].append(order)

    def step(self) -> List[Order]:
        raise

    def get_activities(self, current_time: int) -> list[Order]:
        activities = self.scheduled_activities.pop(current_time, [])
        self.rand.shuffle(activities)
        return activities
