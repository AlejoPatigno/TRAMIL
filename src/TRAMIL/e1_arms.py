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
    "experiments": ["E1", "E2", "E5"],
    "invariants": [
        "every_e1_arm_builds_trains_one_epoch_and_predicts_finite_items",
        "comparators_and_untargeted_arm_train_without_the_penalty",
    ],
}

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
}
SOURCE = {"prompt_only": "R"}


def build(arm, k, p_drop, vbar, Q=None):
    return functools.partial(ARMS[arm], k, p_drop, vbar, Q)
