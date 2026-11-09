"""An ablation ladder for what prior mammograms add to benign/malignant classification.

Every arm of the ladder runs the same encoder, aggregator, loss and training
schedule on the same patients; the arms differ only in which cells of a patient's
(exam x view x tile) grid are masked. A difference between two arms is therefore
a difference in input alone.
"""

__version__ = "0.1.0"
