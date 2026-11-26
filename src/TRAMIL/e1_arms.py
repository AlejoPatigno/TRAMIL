"""The neural arms of v07 E1, E2 and E5 as zero-argument model constructors for item_training.train.

build(arm, k, p_drop, vbar, Q): k = PCA dimension (the k = 768 reference of Ec. 103 is "full" at k = 768), p_drop =
dropout of Ec. 14, vbar = frontend.rc_scale of the fold's training bags (mu_R = 1 / vbar for the penalized arms),
Q = the 8 x 768 PHQ-8 descriptor embeddings (Symptom-Guided only). SOURCE names the arms whose input is not the
audio: the prompt-only model reads the uncentered prompt rows (frontend.fold_bags(..., k=8, source="R"))."""
import functools

from .item_comparators import ABMIL8, R06Items, SymptomGuided
from .surrogate import SurrogateModel

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v07.md",
    "sections": ["E1", "E2", "E5"],
    "experiments": ["E1", "E2", "E3", "E4", "E5"],
    "invariants": [
        "every_e1_arm_builds_trains_one_epoch_and_predicts_finite_items",
        "comparators_and_untargeted_arm_train_without_the_penalty",
    ],
}

E1_ARMS = ("full", "permuted", "untargeted", "r12", "prompt_only", "abmil8", "gated_abmil8", "abmil_r8", "r06_8",
           "symptom_guided")  # the registry of E1 (the default arm set of e1v07_run)
ARMS = {
    "full": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v),
    "permuted": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v, permuted=True),
    "untargeted": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=0.0),
    "r12": lambda k, p, v, Q: SurrogateModel(k, p, prior="text"),
    "prompt_only": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v, log_n=True),
    "abmil8": lambda k, p, v, Q: ABMIL8(k, p),
    "gated_abmil8": lambda k, p, v, Q: ABMIL8(k, p, gated=True),
    "abmil_r8": lambda k, p, v, Q: ABMIL8(k, p, with_R=True),
    "r06_8": lambda k, p, v, Q: R06Items(k, p),
    "symptom_guided": lambda k, p, v, Q: SymptomGuided(Q, k, p),
    # E2 to E5 (e2e5_run). *_x re-run an E1 arm only to store extras (attention, fidelity, graph diagnostics);
    # permuted_quint is the permuted-target model whose target rows are the quintile-derangement donors (E5 d);
    # full_nohist is the full model on bags without the mental-health-history segments (E5 e).
    "graph_corr": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v, graph="corr"),
    "graph_prior": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v, graph="prior"),
    "binary_sup": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v, binary=True),
    "free_desc": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v, free="desc", Q=Q),
    "free_zero": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v, free="zero"),
    "permuted_quint": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v, permuted=True),
    "full_nohist": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v),
    "full_x": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v),
    "permuted_x": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=1 / v, permuted=True),
    "untargeted_x": lambda k, p, v, Q: SurrogateModel(k, p, mu_R=0.0),
}
# The other E1 arms re-run as *_x for the same reason (their E1 real units persist no model): E6 and E7 read frozen models.
ARMS.update({a + "_x": ARMS[a] for a in ("r12", "prompt_only", "abmil8", "gated_abmil8", "abmil_r8", "r06_8", "symptom_guided")})
E2E5_ARMS = tuple(a for a in ARMS if a not in E1_ARMS)
ALIAS = {a + "_x": a for a in ("full", "permuted", "untargeted", "r12", "prompt_only", "abmil8", "gated_abmil8", "abmil_r8",
                               "r06_8", "symptom_guided")}  # E1 arm whose k*, p* they use
SOURCE = {"prompt_only": "R", "prompt_only_x": "R"}


def build(arm, k, p_drop, vbar, Q=None):
    return functools.partial(ARMS[arm], k, p_drop, vbar, Q)
