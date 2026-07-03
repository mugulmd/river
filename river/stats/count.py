from __future__ import annotations

import typing

from river import stats


class Count(stats.base.Univariate):
    """A simple counter.

    Attributes
    ----------
    n : int
        The current number of observations.

    """

    def __init__(self) -> None:
        self.n: int = 0

    def update(self, x: typing.Any = None) -> None:
        self.n += 1

    def get(self) -> float:
        return self.n
