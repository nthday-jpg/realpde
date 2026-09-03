from datasets.pde_dataset import PDEDataset

# Dummy to satisfy `accelerate` which does `from datasets import IterableDataset`.
# Our local `datasets/` shadows HF `datasets`; since HF is not used, a dummy is enough.
class IterableDataset:
    pass

__all__ = ["PDEDataset", "IterableDataset"]
