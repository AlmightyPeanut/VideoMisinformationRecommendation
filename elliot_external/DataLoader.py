import typing as t
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class DataLoader(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace):
        pass

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        pass

    def filter(self, users: t.Set[int], items: t.Set[int]):
        pass

    def create_namespace(self) -> SimpleNamespace:
        pass
