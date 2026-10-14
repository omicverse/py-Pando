"""Class API: the same workflow as method chaining on one object."""
from __future__ import annotations

from . import grn as _grn
from .graph import get_network_graph
from .objects import GRNData, coef, gof, NetworkModules


class Pando:
    """Method-chaining wrapper around :class:`~pypando.objects.GRNData`.

    >>> p = Pando(rna, atac, annotation).find_motifs(genome="hg38.fa").infer_grn(genes=targets).find_modules()
    >>> p.coefs.head()
    """

    def __init__(self, rna, atac, annotation, **initiate_kwargs):
        self.data: GRNData = _grn.initiate_grn(rna, atac, annotation, **initiate_kwargs)

    def find_motifs(self, **kwargs) -> "Pando":
        _grn.find_motifs(self.data, **kwargs)
        return self

    def infer_grn(self, **kwargs) -> "Pando":
        _grn.infer_grn(self.data, **kwargs)
        return self

    def find_modules(self, **kwargs) -> "Pando":
        _grn.find_modules(self.data, **kwargs)
        return self

    def network_graph(self, **kwargs) -> "Pando":
        get_network_graph(self.data, **kwargs)
        return self

    @property
    def coefs(self):
        return coef(self.data)

    @property
    def gof(self):
        return gof(self.data)

    @property
    def modules(self):
        return NetworkModules(self.data)

    def __repr__(self):
        return repr(self.data)
