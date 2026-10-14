"""Design matrices for Pando's per-gene model formulas, with R's naming.

Pando builds, for every target gene, a formula of the form::

    target ~ peak1 <op> tfA + peak1 <op> tfB + peak2 <op> tfA + ...

with ``<op>`` one of ``:`` (default), ``*`` or ``+``. R turns this into
columns differently depending on the fitting function, and the column names
(``term`` in Pando's output) follow from that:

* ``stats::model.matrix`` (glm, xgb, bagging/bayesian ridge): terms are
  de-duplicated, main effects come before interactions, and an interaction's
  label lists its variables in the order the variables first appear in the
  formula (so ``peak2:tfA`` is labelled ``tfA:peak2`` if ``tfA`` appeared
  earlier). An ``(Intercept)`` column comes first.
* ``glmnetUtils`` (glmnet, cv.glmnet): each additive term is expanded on
  its own (``~ 0 + term``) and the pieces are bound together without
  de-duplication; labels keep the order written in that term.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

__all__ = ["PandoFormula", "is_syntactic"]

_SYNTACTIC = re.compile(r"^((([A-Za-z]|[.][._A-Za-z])[._A-Za-z0-9]*)|[.])$")
_RESERVED = {
    "if", "else", "repeat", "while", "function", "for", "next", "break", "TRUE", "FALSE",
    "NULL", "Inf", "NaN", "NA", "NA_integer_", "NA_real_", "NA_character_", "in",
}


def is_syntactic(name: str) -> bool:
    """Whether ``name`` parses as a bare R symbol (so ``as.formula`` accepts it)."""
    return bool(_SYNTACTIC.match(name)) and name not in _RESERVED


@dataclass
class PandoFormula:
    """``target ~ sum over (peak, tf) pairs of peak <op> tf``.

    ``pairs`` lists ``(peak_name, tf_name)`` in formula order, names already
    sanitised (``-`` -> ``_``).
    """

    target: str
    pairs: list
    interaction: str = ":"

    def __post_init__(self):
        if self.interaction not in (":", "*", "+"):
            raise ValueError("interaction_term must be one of ':', '*', '+'")

    @classmethod
    def from_string(cls, formula: str) -> "PandoFormula":
        """Parse an R formula of Pando's form, e.g. ``"GENE ~ p1:TFA + p2*TFB"``.

        Every additive term must be ``peak:tf`` or ``peak*tf`` (one operator for
        the whole formula, as Pando writes it).
        """
        lhs, rhs = (x.strip() for x in formula.split("~", 1))
        terms = [t.strip() for t in rhs.split("+") if t.strip()]
        ops = {":" if ":" in t else "*" if "*" in t else None for t in terms}
        if len(ops) != 1 or None in ops:
            raise ValueError("every term must be 'peak:tf' or 'peak*tf' with a single operator")
        op = ops.pop()
        pairs = [tuple(x.strip() for x in t.split(op)) for t in terms]
        return cls(lhs, pairs, op)

    # -- variables in order of first appearance (R's all.vars) --------------
    def variables(self) -> list:
        seen = {self.target: 0}
        for p, t in self.pairs:
            for v in (p, t):
                if v not in seen:
                    seen[v] = len(seen)
        return list(seen)

    def check_parse(self):
        for v in self.variables():
            if not is_syntactic(v):
                raise ValueError(
                    f"R cannot parse the formula: {v!r} is not a syntactic name "
                    "(Pando's as.formula() would fail here as well)"
                )

    # -- stats::model.matrix ---------------------------------------------------
    def model_matrix_terms(self) -> list:
        """Unique terms in model.matrix order, each a tuple of variable names
        ordered as in the label."""
        order = {v: i for i, v in enumerate(self.variables())}
        mains, inters = [], []
        seen_m, seen_i = set(), set()

        def add_main(v):
            if v not in seen_m:
                seen_m.add(v)
                mains.append((v,))

        def add_inter(a, b):
            key = frozenset((a, b))
            if a == b:
                add_main(a)
                return
            if key not in seen_i:
                seen_i.add(key)
                inters.append(tuple(sorted((a, b), key=order.__getitem__)))

        for p, t in self.pairs:
            if self.interaction == ":":
                add_inter(p, t)
            elif self.interaction == "*":
                add_main(p)
                add_main(t)
                add_inter(p, t)
            else:
                add_main(p)
                add_main(t)
        return mains + inters

    def model_matrix(self, data: dict, intercept: bool = True):
        """Return ``(X, colnames)`` as ``stats::model.matrix(formula, data)``."""
        cols, names = [], []
        n = len(next(iter(data.values())))
        if intercept:
            cols.append(np.ones(n))
            names.append("(Intercept)")
        for term in self.model_matrix_terms():
            v = np.asarray(data[term[0]], dtype=np.float64)
            for other in term[1:]:
                v = v * np.asarray(data[other], dtype=np.float64)
            cols.append(v)
            names.append(":".join(term))
        return np.column_stack(cols), names

    # -- glmnetUtils makeModelComponents --------------------------------------
    def glmnet_matrix(self, data: dict):
        """Return ``(X, colnames)`` as glmnetUtils builds it (no intercept column)."""
        cols, names = [], []
        for p, t in self.pairs:
            if self.interaction == ":":
                if p == t:
                    pieces = [(p,)]
                else:
                    pieces = [(p, t)]
            elif self.interaction == "*":
                pieces = [(p,), (t,), (p, t)] if p != t else [(p,)]
            else:  # '+': two separate additive terms
                pieces = [(p,), (t,)]
            for term in pieces:
                v = np.asarray(data[term[0]], dtype=np.float64)
                for other in term[1:]:
                    v = v * np.asarray(data[other], dtype=np.float64)
                cols.append(v)
                names.append(":".join(term))
        return np.column_stack(cols), names
