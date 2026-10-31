# Asteroid PMSQE training loss

These upstream files are unmodified and pinned to the revision and SHA256 values in
`PROVENANCE.json`. Copyright and MIT terms are retained in `LICENSE` and `pmsqe.py`.
Only the 16kHz Bark matrix is included; this isolated copy does not support8kHz.

Source: https://github.com/asteroid-team/asteroid/tree/c15708a04d3d28e9a1cd50553c456299dbc6d236/asteroid/losses

`experiments/causal_pmsqe_objective.py` adds PMSQE to the existing physical loss for
a controlled training experiment. It uses physical power reconstructed from the
unchanged512/160/320 Hann frontend, corrects window energy, excludes batch padding
and silent-reference utterances, and keeps upstream perceptual equalization inside
the loss. It changes no model output, evaluation metric, normalization or streaming
state. PMSQE is a differentiable surrogate; the production metric remains the
existing independent wideband `pesq` implementation on all600 comparison samples.
