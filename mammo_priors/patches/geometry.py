"""Where a patch sits, in the pixels of the image it is cut from.

Windows are cut at the image's native resolution and never resampled, so the
morphology of microcalcifications is kept as acquired.
"""

from dataclasses import dataclass
from math import ceil


def clamped(start: float, size: int, limit: int) -> int:
    """A window start moved just far enough to keep the whole window in the image.

    Clamped rather than padded: a crop is then always full size and always real
    pixels, while a zero-padded edge would be a black band a model could learn.
    """
    return int(round(min(max(start, 0), max(0, limit - size))))


@dataclass(frozen=True)
class PixelWindow:
    """A square window on one image, named by its top-left corner (row, column)."""

    origin_yx: tuple
    size_px: int

    @property
    def bounds_yx(self) -> tuple:
        y, x = self.origin_yx
        return (y, x, y + self.size_px, x + self.size_px)

    @property
    def centre_yx(self) -> tuple:
        return (self.origin_yx[0] + self.size_px / 2, self.origin_yx[1] + self.size_px / 2)

    def grown(self, size_px: int, grid: tuple) -> "PixelWindow":
        """The same centre at `size_px`, clamped into an image of `grid`.

        Growing to its own size is how a window is clamped into another grid.
        """
        centre_y, centre_x = self.centre_yx

        return PixelWindow((clamped(centre_y - size_px / 2, size_px, grid[0]),
                            clamped(centre_x - size_px / 2, size_px, grid[1])), size_px)

    def mirrored(self, grid: tuple) -> "PixelWindow":
        """The window covering the same pixels once the image is mirrored left-right."""
        y, x = self.origin_yx
        return PixelWindow((y, grid[1] - x - self.size_px), self.size_px)


def _origins(low: int, high: int, size: int, limit: int) -> list:
    """Window starts along one axis that together cover [low, high).

    One window centred on the box when it fits. Otherwise the first starts at the
    low edge, the last ends at the high edge and the rest are spaced evenly, so
    they overlap rather than leave a seam through the lesion.
    """
    count = max(1, ceil((high - low) / size))
    if count == 1:
        starts = [(low + high - size) / 2]
    else:
        slack = (high - low) - size
        starts = [low + slack * step / (count - 1) for step in range(count)]

    return [clamped(start, size, limit) for start in starts]


def tile(box_yx: tuple, grid: tuple, size_px: int) -> list:
    """The windows covering one box: several overlapping ones when it is larger.

    Tiled rather than clipped, so a large lesion is seen whole instead of being
    quietly turned into a small one. Every tile of a patient belongs to the same
    sample, so tiling never adds samples.
    """
    y_min, x_min, y_max, x_max = box_yx
    height, width = grid

    return [PixelWindow((y, x), size_px)
            for y in _origins(y_min, y_max, size_px, height)
            for x in _origins(x_min, x_max, size_px, width)]
