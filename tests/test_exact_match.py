"""The pre-registered parity gate (``data/manifest.yaml``) as a pytest test.

Needs the R reference written by ``tests/r_reference_driver.R`` and the hg38
FASTA matching ``BSgenome.Hsapiens.UCSC.hg38``::

    PYPANDO_REF_DIR=/path/to/ref PYPANDO_FASTA=/path/to/hg38.fa pytest tests/test_exact_match.py

Skipped when either variable is unset. Set ``PYPANDO_CAND_DIR`` to reuse a
finished candidate run instead of recomputing it (``tests/_run_candidate.py``).
"""
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REF = os.environ.get("PYPANDO_REF_DIR")
FASTA = os.environ.get("PYPANDO_FASTA")

pytestmark = pytest.mark.skipif(not (REF and FASTA), reason="needs PYPANDO_REF_DIR and PYPANDO_FASTA")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    import parity_check

    cand = os.environ.get("PYPANDO_CAND_DIR")
    if not cand:
        import _run_candidate

        cand = str(tmp_path_factory.mktemp("cand"))
        methods = ",".join(m for m in ("glm", "glmnet", "cv.glmnet", "xgb", "bagging_ridge", "bayesian_ridge")
                           if (Path(REF) / f"coefs_{m}.tsv").exists())
        _run_candidate.main(REF, cand, FASTA, methods)
    return parity_check.main(REF, cand)


GATES = [
    "candidate_regions", "peaks_near_genes_signac", "peaks_near_genes_great", "motif_matrix",
    "glm_keys", "glm_coefficients", "glm_pvalues", "glmnet_coefficients", "cvglmnet_coefficients",
    "xgb_gain", "bagging_ridge", "bayesian_ridge", "modules",
]


@pytest.mark.parametrize("gate", GATES)
def test_gate(results, gate):
    if gate not in results:
        pytest.skip(f"no reference output for {gate}")
    assert results[gate]["pass"], f"{gate}: {results[gate]}"
