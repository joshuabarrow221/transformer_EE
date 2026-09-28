# Scientific method and boundaries

One point is a complete neutrino interaction. A frozen `Transformer_EE_MV`
checkpoint, including single-target uses of that architecture, maps measured
particle vectors and global scalars to physical predictions. Forward pre-hooks
record its actual regression-head input and penultimate representation. The
exporter honors saved padding, masks, normalization, and maximum prong count;
truncations are counted in metadata. The optimized path is checked against the
repository's original Dataset/forward path for each chunk, including leading
events and representatives of every observed prong multiplicity.

The code uses `eval()` and `inference_mode()` with no new inference noise.
Noise/no-noise denotes the training condition. No optimizer or training step
is run. MV bundles predict energy and three momentum components; composed SV
bundles concatenate independent head-input blocks for those same four targets.
The original selected networks have 272-dimensional blocks, hence 1,088 for SV;
the portable extractor derives actual widths from the checkpoint outputs.

## Event selection and labels

Select the first requested number of usable records in each CSV, preserving
source row and Event_Index. All models applied to that source share the same
selection. Validation uses the union of selected model input fields, all four
truth kinematics, finite numeric vectors, aligned particle-list lengths, exact
topology strings, and agreement with PDG multiplicities. Invalid rows are excluded
and replaced by later usable records. Duplicate selected event IDs are rejected.
The prefix scan does not claim to audit the full source population. There is no
energy cut based on a filename and no topology balancing or prediction-based cut.

Read topology as decimal strings, never float32. Its format is one neutrino-flavor
digit followed by two-digit proton, pi+, pi-, and pi0 counts separated by `00`.
The colors are `0p1pi, 1p0pi, 2p0pi, 1p1pi, 2p1pi, 1p2pi, 2p2pi, NpNpi, Other`.
`NpNpi` means at least three protons **or** at least three summed pions. `Other`
retains 0p0pi and 0p2pi. Neutrons and other hadrons are not encoded in this label.

The inspected GENIE/NuWro preprocessing increments proton/charged-pion counts
above configured kinetic-energy thresholds; pi0 counts have no kinetic-energy
cut. Thus this is a preprocessing-selected truth label, not a detector-efficiency
or reconstructed-topology measurement. The inspected configurations used 25 MeV
for protons and 70 MeV for charged pions, but the CSV alone cannot establish the
runtime threshold configuration that produced it. See the original
[topology implementation](https://github.com/joshuabarrow221/transformer_EE/blob/d28eee3d524f826b6b6fb45ea47c2d20a9f1c029/preprocessing_GENIE_files/codes/VectorLept_wNC.C#L833-L938).

## Embedding

For each fixed model bundle/training row, fit all selected generators jointly.
Columns in that row share one coordinate system. Different training rows are
independent fits; their orientations, island positions, and axes cannot be
compared directly. Adding a generator requires fitting that joint row again.

For SV only, center each network block and divide by its RMS norm, measured
across the joint row. Reduce with unwhitened PCA to at most 50 dimensions.
Use Euclidean neighbors, PCA initialization, openTSNE FFT optimization, seed 42,
perplexity 30, 250 early-exaggeration iterations plus 750 ordinary iterations.
Annoy supplies approximate neighbors for latent features. All selected points
participate in fitting and plotting; there is no landmark-only approximation.
Alternate seed 7/perplexity 50 is a separate sensitivity fit. openTSNE and
scikit-learn have different optimization conventions and are not interchangeable
for exact reproduction. Hardware/library changes can also change coordinates.

Colors never enter PCA/t-SNE. The selected models have zero topology-loss weight;
the portable runner rejects a nonzero topology weight or topology input field.
Particle identities and multiplicities already enter the network, so topology
separation does not by itself establish a learned topology objective or better
kinematic accuracy. Cluster areas, distant-island separations, and orientations
are not calibrated physical quantities. Source topology frequencies can differ.
The supplied source names do not prove held-out membership or train/test
independence. This is descriptive representation analysis.

## Historical selection notes

The example study uses GENIE AR23-trained networks on four GENIE beam inputs plus
NuWro, and four atmospheric GENIE inputs where checkpoints exist. The three
non-AR23 atmospheric inputs supplied for the extension were byte-identical to
earlier copies labeled `HondaDUNEOsc_*`. Their use follows the user's explicit
source designation; a directory named `Natural_Spectra` does not establish an
unoscillated flux. The atmospheric natural/noise replica remains supplemental.

The completed SV manifests include subsequently verified noise bundles. In the
flat/noise MAPE case, Px explicitly uses J1 (`model_166b0be5...`), not the different
C1 replica selected by an older timestamp-based composition. Model hashes and
bundle membership now make this choice explicit. Replicas are never selected
automatically by filename, newest modification time, or loss name alone.
