"""Which window of which image every cell of one patient's grid is cut from.

The axes are (timepoint, view, tile): t = 0 is the index exam and t grows into
the past, the views are CC and MLO, and the tiles are the windows covering the
findings of that view. A tile means the same place at every timepoint, so a
prior is cut wherever the correspondence method carries that index window.
"""

from dataclasses import dataclass

from mammo_priors.data.cohort import views_of
from mammo_priors.data.tables import VIEWS
from mammo_priors.patches.geometry import PixelWindow, tile


@dataclass(frozen=True)
class Cell:
    """One patch: its address in the grid and what to cut for it."""

    timepoint: int
    view: int
    tile: int
    image: str
    laterality: str
    window: PixelWindow
    grid: tuple
    # Whether the prior faces the other way and its patch must be mirrored back.
    mirrored: bool = False


@dataclass(frozen=True)
class IndexTile:
    """One window on the index exam, which every timepoint is cut at."""

    image: str
    laterality: str
    window: PixelWindow


class SampleGrid:
    """Every cell of one patient, and the rectangular shape they form.

    The tile axis is as long as the busiest view needs, so a patient with two CC
    windows and one MLO window has a hole rather than a ragged row: a hole is
    what the presence mask is for.
    """

    def __init__(self, sample, reader, correspondence, tile_px: int):
        self.sample = sample
        self.reader = reader
        self.correspondence = correspondence
        self.tiles = [self._tiled(findings, tile_px) for findings in views_of(sample)]

    def _tiled(self, findings: list, tile_px: int) -> list:
        index = self.sample.exams[0]
        tiles = []
        for finding in findings:
            image = index.image_of(finding.laterality, finding.view)
            tiles += [IndexTile(image, finding.laterality, window)
                      for window in tile(finding.box_yx, self.reader.grid(image), tile_px)]

        return tiles

    @property
    def shape(self) -> tuple:
        return (len(self.sample.exams), len(self.tiles),
                max(len(tiles) for tiles in self.tiles))

    def _cell(self, timepoint: int, view: int, at: int) -> Cell:
        """One cell, or None where that exam holds no image of the breast and view."""
        index_tile = self.tiles[view][at]
        if timepoint == 0:
            return Cell(0, view, at, index_tile.image, index_tile.laterality,
                        index_tile.window, self.reader.grid(index_tile.image))

        exam = self.sample.exams[timepoint]
        image = exam.image_of(index_tile.laterality, VIEWS[view])
        if image is None:
            return None

        return Cell(timepoint, view, at, image, index_tile.laterality,
                    self.correspondence.locate(index_tile.window, index_tile.image, image),
                    self.reader.grid(image),
                    self.correspondence.mirrored(index_tile.image, image))

    def cells(self) -> list:
        """Every cell that exists, sorted by image so each file is decoded once."""
        found = [self._cell(timepoint, view, at)
                 for timepoint in range(self.shape[0])
                 for view, tiles in enumerate(self.tiles)
                 for at in range(len(tiles))]

        return sorted((cell for cell in found if cell is not None), key=lambda cell: cell.image)
