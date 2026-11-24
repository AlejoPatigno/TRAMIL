# Training diagnostic (T7 Step 1): can the graph-free readout learn the item task?

Diagnostic only, no PoC criterion; test is never read. a = 0.5, synthetic data, PoC settings.

Readings of the pre-registered T7 spec, fixed before any T7 run:
- Step 1 uses a = 0.5 and the graph-free model (H~ = H: encoder, attention pool W_o, item heads), batch 8, Adam.
  Test is never read in Step 1 (only train and val bags are generated).
- D1: the first 16 train bags of seed 0; v3 config (lr 1e-2, W_o = 0), 400 epochs; train MSE on those 16 bags after
  every epoch. "Well below the constant" read as min train MSE <= 0.5 x the constant's train MSE on the same bags.
  If D1 fails, D2 is not run and the readout defect is reported.
- D2: grid lr {1e-3, 3e-3, 1e-2} x W_o init {zero, normal}; 5 seeds, full 256 train bags; up to 400 epochs, val
  item MSE after every epoch, best epoch on val (ties -> earliest) per run. Config score = seed mean of the best
  val MSE; the config with the lowest score is chosen. "N(0, 0.01)" read in the document's N(mean, variance)
  notation (as N(0, 0.005^2) for the prompt noise): sd 0.1.
- Gate V3: chosen config's score < constant val MSE (seed mean of the per-item training-mean predictor on val).
  V3 fails -> v3c is not run.
- v3c "training config" = chosen lr and W_o init, with the per-run rule "up to 400 epochs, best epoch on val (item
  MSE without the mu term)" for EVERY arm (full per mu, prior-only, acoustic-only). mu* = argmin over mu of the
  val item MSE at each run's best epoch. ORACLE-SEG is not computed in v3c (V2 now uses ORACLE-BAG).
- ORACLE-BAG: per bag, the mean of all raw segment features X (no item routing), the same vector for all 8 items;
  one OLS head (with intercept) per item, fit on train (gelsd, relative rcond 1e-6, as ORACLE-NB).
- Runs are independent (each seeds itself) and are executed in parallel worker processes with 1 torch thread each;
  results do not depend on the parallelism.

## D1 capacity (16 train bags, lr 1e-2, W_o = 0)

| epoch | 1 | 10 | 25 | 50 | 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|---|---|---|
| train MSE | 2.8630 | 1.0081 | 0.7600 | 0.1139 | 0.0005 | 0.0000 | 0.0013 | 0.0007 |

Constant train MSE on these bags 1.0962; min train MSE 0.0000 at epoch 209 (needs <= 0.5481). **D1 PASS**

## D2 sweep (val only; best epoch on val per run; 5 seeds)

| config | best val MSE (mean +- sd) | best epoch per seed |
|---|---|---|
| lr=0.001, W_o=zero | 1.1315 +- 0.0599 | [33, 33, 34, 28, 43] |
| lr=0.001, W_o=normal | 1.1592 +- 0.0428 | [16, 34, 49, 34, 45] |
| lr=0.003, W_o=zero | 1.1192 +- 0.0551 | [11, 11, 13, 11, 16] |
| lr=0.003, W_o=normal | 1.1461 +- 0.0582 | [5, 10, 15, 16, 19] |
| lr=0.01, W_o=zero | 1.1079 +- 0.0597 | [5, 5, 5, 3, 6] |
| lr=0.01, W_o=normal | 1.1316 +- 0.0606 | [8, 5, 10, 4, 6] |

Constant val MSE 1.2566. Chosen config: **lr=0.01, W_o=zero**.

## Gate V3: chosen val MSE 1.1079 < constant val MSE 1.2566. **PASS**
