"""Python side of the get_tf_network gates (data/manifest.yaml, amendment 2026-10-14).

    python tests/_run_tf_network.py <ref_dir> <out_dir> <hg38.fa>

Runs initiate_grn -> find_motifs -> infer_grn(glm) -> find_modules -> get_network_graph on the
fixture (reusing <ref_dir>/bench_grn.pkl when present), then get_tf_network for the 12 TFs with the
largest out-degree in the module graph, in the variants the R driver writes (PANDO_TF_NETWORK=1).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pypando as pp  # noqa: E402

# R driver variant -> py-Pando arguments. R_default is not reproduced: py-Pando's default is the fix.
VARIANTS = {
    "R_all": dict(keep_all_edges=True, order=np.inf),
    "intent_default": dict(),
    "intent_all": dict(keep_all_edges=True, order=3),
    "intent_default_o2": dict(order=2),
}


def top_tfs(edges: pd.DataFrame, n: int = 12) -> list:
    deg = edges["from_node"].value_counts()
    return sorted(deg.index, key=lambda t: (-int(deg[t]), t.encode()))[:n]


def main(ref, out, fasta):
    import bench_infer_grn as b

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    grn, targets = b._prepared(ref, fasta)
    g = pp.infer_grn(grn, genes=targets, method="glm", verbose=False)
    pp.find_modules(g, verbose=False)
    pp.get_network_graph(g, verbose=False)
    edges = pp.NetworkGraph(g)["edges"]
    edges.to_csv(out / "module_graph_edges.tsv", sep="\t", index=False, float_format="%.17g")
    tfs = top_tfs(edges)
    (out / "tf_network_tfs.txt").write_text("\n".join(tfs) + "\n")
    rows, status = [], {}
    for v, kw in VARIANTS.items():
        for tf in tfs:
            try:
                pp.get_tf_network(g, tf=tf, verbose=False, **kw)
            except ValueError as e:
                status[f"{v}/{tf}"] = f"error: {e}"
                continue
            e = pp.NetworkGraph(g, graph="tf_graphs")[tf]["edges"].copy()
            e["from_name"], e["to_name"] = e["from_node"], e["to_node"]
            e["variant"], e["root"], e["edge_index"] = v, tf, np.arange(1, len(e) + 1)
            rows.append(e)
            status[f"{v}/{tf}"] = f"ok: {len(e)} edges, {len(pp.NetworkGraph(g, graph='tf_graphs')[tf]['nodes'])} nodes"
    pd.concat(rows).to_csv(out / "tf_networks.tsv", sep="\t", index=False, float_format="%.17g")
    (out / "tf_networks_status.json").write_text(json.dumps(status, indent=1))
    print("done")


if __name__ == "__main__":
    main(*sys.argv[1:4])
