"""TRAMIL: residual semantic prior bipartite MIL."""

__implementation__ = {
    "revision": {
        "proposal": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
        "experiments": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v04.md",
    },
    "premises": {
        "prediction": "binary depression label (PHQ-8 >= 10) per subject",
        "statisticalUnit": "subject",
        "metric": "AUROC",
        "direction": "higher",
    },
}
__backend__ = "torch"
