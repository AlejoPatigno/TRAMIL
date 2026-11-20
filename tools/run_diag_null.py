"""One-off: permutation null for bag_svm. Usage: python tools/run_diag_null.py wavlm|egemaps N"""
import json, sys, time
import numpy as np
from TRAMIL import diag_instance_svm as d, e1_run, exp_e1

ac, n = sys.argv[1], int(sys.argv[2])
f = e1_run._find(e1_run._default_roots(ac), acoustic=ac)
items, _ = exp_e1.load_items(f["labels"], f["instances"].parent, f["features"], acoustic=ac, text_dir=f.get("text"))
t = time.time(); obs = d.bag_svm_val_auroc(items); print(ac, "obs", obs, "s/run", time.time() - t, flush=True)
null = []
for s in range(n):
    null.append(d.bag_svm_val_auroc(items, seed=1000 + s))
    if s % 10 == 9: print(s + 1, time.time() - t, flush=True)
json.dump({"acoustic": ac, "obs": obs, "null": null, "n_subjects": len(items)}, open(f"TRAMIL/Results/diag_null_bag_svm_{ac}.json", "w"))
