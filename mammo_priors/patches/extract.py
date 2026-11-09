"""Cut every patient of the cohort into the cache of one correspondence method.

Resumable: a patient whose file exists is skipped, so an interrupted extraction
continues where it stopped. Each correspondence method is its own cache folder,
because its name is part of the patch config the folder is named by.
"""

import time

from mammo_priors.config import ComponentConfig, PatchConfig
from mammo_priors.patches.cache import PatchCache
from mammo_priors.patches.correspondence import CORRESPONDENCES, SAME_LOCATION
from mammo_priors.patches.sources import ImageReader


def patch_config(cohort, correspondence: str = SAME_LOCATION) -> PatchConfig:
    """The cache a cohort is cut into: its revision is part of the name."""
    return PatchConfig(correspondence=ComponentConfig(correspondence),
                       cohort_revision=cohort.revision())


def extract(cohort, cache_root: str, image_root: str,
            correspondence: str = SAME_LOCATION) -> PatchCache:
    cache = PatchCache(cache_root, patch_config(cohort, correspondence))
    reader = ImageReader(image_root)
    placing = CORRESPONDENCES.build(cache.config.correspondence, reader=reader)

    started, cut, skipped, patches = time.time(), 0, 0, 0
    for at, sample in enumerate(cohort.samples, start=1):
        if cache.holds(sample):
            skipped += 1
            continue
        patches += cache.write(sample, reader, placing)
        cut += 1
        if at % 50 == 0:
            print(f"  {at}/{len(cohort.samples)} patients, {time.time() - started:.0f} s")

    print(f"cut {cut} patients ({patches} patches), {skipped} already cut, "
          f"priors placed by {correspondence}, into {cache.folder}")

    return cache
