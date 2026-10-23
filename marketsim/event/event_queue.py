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
        # self.current_time = 0 # is it even needed here? we just store the time im simulation
        # we will have to think about optimization later, when the dict is not enough and we'll have to
        # save mamory usage

    def schedule_activity(self, order: Order):
        t = order.time

        self.scheduled_activities[t].append(order)

    def step(self) -> List[Order]:
        raise
        random.shuffle(self.scheduled_activities[self.current_time])
        self.current_time += 1  # TODO: AK: isn't it redundant? step in market, step in queue.
        # that's why matched order has time t+1, while placed order t
        # without it nothing happens on the market, we have to align it somehow

        return self.scheduled_activities[self.current_time - 1]

    def get_activities(self, current_time: int) -> List[Order]:
        # replaces old step() method to use synchronized time with the simulation
        random.shuffle(self.scheduled_activities[current_time])
        return self.scheduled_activities[current_time]

    # def get_current_time(self):
    #     return self.current_time

    # def set_time(self, t):
    #     self.current_time = t
