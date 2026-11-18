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

# Ordered rung ladder for the position sequence: rehearsal-scale pilot, then the declared full scale.
__levels__ = ["pilot", "full"]

__steps__ = {
    "extract-features": {
        "module": "TRAMIL.extract_run",
        "function": "run",
        "advances": 1,
        "reads": [],
        "produces": ["Data/features"],
        "placement": "remote",
        "job": "extract-features",
        "service": "kaggle",
    },
    "e1-full": {
        "module": "TRAMIL.e1_run",
        "function": "run",
        "advances": 1,
        "reads": ["Data/features"],
        "produces": ["Results/E1"],
        "placement": "remote",
        "job": "e1-full",
        "service": "kaggle",
    },
}
