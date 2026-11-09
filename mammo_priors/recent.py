"""A bounded memo for what one patient's extraction asks for more than once.

Images are expensive to decode and large to hold, and an extraction walks
thousands of patients of which only the one in hand is asked about again, so the
memo forgets the oldest result instead of growing for the whole run.
"""

from collections import OrderedDict


class Recent:
    """The last `kept` results of `compute`, computed on first request."""

    def __init__(self, compute, kept: int):
        self.compute = compute
        self.kept = kept
        self._held = OrderedDict()

    def __getitem__(self, key):
        if key in self._held:
            self._held.move_to_end(key)
        else:
            self._held[key] = self.compute(key)
            if len(self._held) > self.kept:
                self._held.popitem(last=False)

        return self._held[key]
