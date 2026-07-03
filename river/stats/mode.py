from __future__ import annotations

import collections
from collections.abc import Hashable

from typing_extensions import TypeVar

from river import stats

__all__ = ["Mode"]

HashableT = TypeVar("HashableT", bound=Hashable, default=Hashable)


class Mode(stats.base.Univariate[HashableT, HashableT]):
    """Running mode.

    The mode is simply the most common value. An approximate mode can be computed by setting the
    number of first unique values to count.

    Parameters
    ----------
    k
        Only the first `k` unique values will be included. If `k` equals -1, the exact mode is
        computed.

    Examples
    --------

    >>> from river import stats

    >>> X = ['sunny', 'cloudy', 'cloudy', 'rainy', 'rainy', 'rainy']
    >>> mode = stats.Mode(k=2)
    >>> for x in X:
    ...     mode.update(x)
    ...     print(mode.get())
    sunny
    sunny
    cloudy
    cloudy
    cloudy
    cloudy

    >>> mode = stats.Mode(k=-1)
    >>> for x in X:
    ...     mode.update(x)
    ...     print(mode.get())
    sunny
    sunny
    cloudy
    cloudy
    cloudy
    rainy

    """

    def __init__(self, k: int = 25) -> None:
        self.k = k
        self.counts: collections.defaultdict[HashableT, int] = collections.defaultdict(int)

    @property
    def name(self) -> str:
        return "mode"

    def update(self, x: HashableT) -> None:
        if self.k == -1 or x in self.counts or len(self.counts) < self.k:
            self.counts[x] += 1

    def get(self) -> HashableT:
        if not self.counts:
            raise stats.base.NotEnoughSamples(f"{self.name} hasn't seen any value")
        return max(self.counts, key=self.counts.__getitem__)


class RollingMode(stats.base.RollingUnivariate[HashableT, HashableT]):
    """Running mode over a window.

    The mode is the most common value.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Attributes
    ----------
    counts : collections.defaultdict
        Value counts.

    Examples
    --------

    >>> from river import stats

    >>> X = ['sunny', 'sunny', 'sunny', 'rainy', 'rainy', 'rainy', 'rainy']
    >>> rolling_mode = stats.RollingMode(window_size=2)
    >>> for x in X:
    ...     rolling_mode.update(x)
    ...     print(rolling_mode.get())
    sunny
    sunny
    sunny
    sunny
    rainy
    rainy
    rainy

    >>> rolling_mode = stats.RollingMode(window_size=5)
    >>> for x in X:
    ...     rolling_mode.update(x)
    ...     print(rolling_mode.get())
    sunny
    sunny
    sunny
    sunny
    sunny
    rainy
    rainy

    """

    def __init__(self, window_size: int):
        self.window: collections.deque[HashableT] = collections.deque(maxlen=window_size)
        self.counts: collections.defaultdict[HashableT, int] = collections.defaultdict(int)

    @property
    def window_size(self) -> int:
        assert self.window.maxlen is not None
        return self.window.maxlen

    def update(self, x: HashableT) -> None:
        if len(self.window) >= self.window_size:
            # Subtract the counter of the last element
            first_in = self.window[0]
            self.counts[first_in] -= 1

            # No need to store the value if it's counter is 0
            if self.counts[first_in] == 0:
                self.counts.pop(first_in)

        self.counts[x] += 1
        self.window.append(x)

    def get(self) -> HashableT:
        if not self.counts:
            raise stats.base.NotEnoughSamples(f"{self.name} hasn't seen any value")
        return max(self.counts, key=self.counts.__getitem__)
