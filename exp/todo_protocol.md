Objective:
Compare loss-free activation/co-statistical pruning criteria against loss-aware Fisher/Taylor-Fisher criteria for structured hidden-neuron pruning. The goal is not to claim Hebbian pruning is new, but to study when cheap activation co-statistics approximate stronger loss-aware saliency, where they fail layer-wise, and how their computational cost compares.

Core research questions:

1. Can loss-free activation/co-statistical criteria prune neurons as safely as Fisher-based criteria?
2. At what epoch do neuron rankings become stable enough to prune?
3. Do criteria behave differently across layers?
4. Does Fisher explain failure cases where activation-only criteria perform worse than random?
5. Is the extra cost of Fisher justified by better pruning performance?
6. How sensitive are criteria to dataset, architecture depth, pruning percentage, and calibration subset size?

Datasets:
Use at minimum:

* MNIST
* Fashion-MNIST
* KMNIST or EMNIST

Optional stress dataset:

* CIFAR-10 flattened to 3072-dimensional vectors

Architectures:
Implement at least three MLP depths.

Small:

* input → 256 → 128 → num_classes

Medium:

* input → 512 → 256 → 128 → 64 → num_classes

Deep:

* input → 512 → 512 → 256 → 256 → 128 → 64 → num_classes

For CIFAR-10 flattened:

* 3072 → 1024 → 512 → 512 → 256 → 128 → 10

Training:

* Standard supervised training with cross-entropy.
* Do not modify training loss for pruning criteria.
* Use Adam or SGD consistently.
* Save checkpoints at every epoch.
* Track train loss, validation loss, validation accuracy, and test accuracy.
* Use multiple random seeds, at least 3.
* Use a fixed validation set and a separate calibration set for pruning scores.
* Calibration data must not be the test set.

Required pruning criteria:

A. Baselines:

1. Random neuron ranking
2. Output weight norm

   * Score neuron j by norm of outgoing weights.
3. Mean activation

   * Score neuron j by average post-activation value.
4. Firing rate

   * Score neuron j by percentage of samples where activation exceeds threshold.
5. Activation variance

   * Score neuron j by variance of post-activation values.

B. Loss-free supervised/statistical criteria:
6. Class selectivity

* Estimate class-conditioned activation means.
* Score neurons by how different their activation profile is across classes.

7. Redundancy / correlation

   * Compute pairwise activation correlation within each layer.
   * Penalize neurons that are highly redundant with many others.
8. Current Hebbian/co-statistical criterion

   * Use the existing criterion already implemented in the previous experiments.
   * Keep it loss-free.

C. Loss-aware criteria:
9. Activation-Fisher / Taylor-Fisher neuron saliency

* For each hidden neuron j:
  score_j = mean over calibration batches of (activation_j * dL/dactivation_j)^2
* Use post-activation values.
* Run backward passes on calibration batches.
* Do not call optimizer.step().
* Do not update weights.
* Use model.eval() unless dropout/batchnorm behavior requires controlled handling.
* Ensure gradients are enabled.
* Clear gradients between batches.

10. Weight-Fisher aggregated to neuron level

* Estimate diagonal empirical Fisher using squared gradients.
* For each neuron, aggregate w^2 * grad_w^2 over incoming and/or outgoing weights.
* Report clearly whether incoming, outgoing, or both were used.

D. Expensive reference:
11. Non-destructive single-neuron ablation sensitivity

* Temporarily mask one neuron at a time.
* Measure validation loss/accuracy change.
* This is not a practical criterion for large models, but serves as an empirical reference.

Implementation design:
Create a modular pipeline with the following components:

1. Dataset module

   * Loads datasets.
   * Creates train/val/test/calibration splits.
   * Supports flattened image inputs.
   * Stores dataset metadata.

2. Model module

   * Defines MLP architectures.
   * Hidden layers should expose names such as a1, a2, a3, etc.
   * Use ReLU activations unless otherwise configured.
   * Include neuron masks per hidden layer for structured pruning.

3. Training module

   * Standard training loop.
   * Saves checkpoints.
   * Logs metrics per epoch.
   * Does not know about pruning except optional hooks.

4. Activation collection module

   * Registers hooks on post-activation tensors.
   * Collects activations per layer on calibration data.
   * Supports memory-safe streaming aggregation.
   * Avoid storing all activations when not needed.

5. Fisher scoring module

   * Registers hooks or uses retained activations.
   * For each calibration batch:

     * forward pass
     * compute cross-entropy loss
     * backward pass
     * collect activation gradients
     * accumulate (activation * activation_gradient)^2
   * No optimizer step.
   * No weight update.
   * Return per-layer neuron scores.
   * Track scoring wall-clock time and number of forward/backward passes.

6. Pruning module

   * Takes per-layer neuron rankings.
   * Supports pruning percentages: 5%, 10%, 20%, 30%, 40%, 50%, 70%.
   * Structured pruning should mask complete hidden neurons.
   * Evaluate both:
     a. non-finetuned accuracy immediately after pruning
     b. accuracy after short fine-tuning, e.g. 1, 3, and 5 epochs
   * Use identical pruning protocol for all criteria.

7. Evaluation module

   * Computes validation/test accuracy.
   * Computes validation/test loss.
   * Computes per-class accuracy.
   * Estimates remaining parameters and approximate MACs/FLOPs.
   * Measures inference time if possible.

8. Logging module

   * Save all results as CSV/Parquet/JSON.
   * Every result row should include:
     dataset
     architecture
     seed
     epoch
     layer
     criterion
     pruning_percentage
     fine_tune_epochs
     validation_accuracy
     test_accuracy
     validation_loss
     test_loss
     parameter_count
     parameter_reduction
     estimated_macs
     scoring_time_seconds
     scoring_memory_mb
     num_forward_passes
     num_backward_passes
     calibration_size

Important methodological rules:

* Do not compare criteria using different calibration data.
* Do not use test data for scoring.
* Do not let Fisher update the model.
* Do not mix ranking stability metrics:

  * Spearman/Kendall for full ranking correlation.
  * Jaccard/top-k overlap for top-k set stability.
* Report layer-wise results separately before aggregating.
* Always compare against random pruning with multiple random draws.
* Use confidence intervals across seeds.
* Keep loss-free criteria clearly separated from loss-aware criteria.
* Treat non-destructive ablation as an expensive empirical reference, not as a deployable baseline.

Primary plots to generate:

1. Accuracy vs pruning percentage for each criterion.
2. Accuracy drop vs pruning percentage.
3. Criterion improvement over random pruning.
4. Layer-wise pruning sensitivity heatmap.
5. Ranking stability across epochs using Spearman or Kendall.
6. Top-k Jaccard overlap across epochs.
7. Fisher score vs current co-statistical score scatter plot.
8. Correlation matrix between all pruning criteria.
9. Cost-benefit Pareto plot: scoring time vs accuracy retained.
10. Scoring memory vs accuracy retained.
11. Fine-tuning recovery curves after pruning.
12. Per-class accuracy degradation after pruning.
13. Score distribution per layer and criterion.
14. Calibration subset size sensitivity.
15. AUC-style pruning damage summary per criterion.
16. Layer failure analysis: cases where criterion pruning is worse than random.
17. Fisher advantage plot: Fisher accuracy minus loss-free criterion accuracy.
18. Rank agreement between Fisher and loss-free criteria across epochs.
19. Parameter reduction vs actual inference-time speedup.
20. Architecture-depth sensitivity plot.

Expected outputs:

* Clean experiment code.
* Reproducible configs.
* Stored raw logs.
* Summary tables.
* Publication-style figures.
* A written analysis answering:

  1. Which criteria work best?
  2. Which criteria are cheapest?
  3. Which criteria are stable earliest?
  4. Which layers are easiest/hardest to prune?
  5. Where does the loss-free co-statistical criterion fail?
  6. Does Fisher explain those failures?
  7. Is the extra Fisher cost worth it?
  8. Does the conclusion change with dataset or architecture depth?

Main thesis to test:
Loss-aware Fisher criteria should often be stronger, but they require backward passes and supervised loss information. The key question is whether cheaper loss-free activation/co-statistical criteria are close enough to be useful, and whether their failure modes can be detected through ranking stability, layer sensitivity, and disagreement with Fisher.
