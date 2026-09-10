"""The ML tab's consolidated help panel.

The tab is one procedure — a nested cross-validation whose outer folds are the honest held-out splits
and whose inner folds are where the optional hyperparameter search runs — with the data and feature
picks on the left and the graph, console and Save on the right. Per-control hover tooltips stay, but
they can't carry the *why* or how the outer and inner loops relate. This dialog is the single place
that explains the tab end to end: what the run does, why nesting the search matters, what the model
sweep costs, and the meaning of the trickier knobs. It is opened by the **Info** button on the run
panel and is purely informational (read-only, modeless so it can sit beside the tab).
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout

# The help body. Plain, sectioned HTML rendered in a scrollable browser — kept here (not inline in the
# page) so the page code stays about wiring and this stays about explanation.
_HELP_HTML = """
<h2>ML — how this tab works</h2>
<p>There is <b>one run</b>. The field is partitioned into <b>outer folds</b>; for each fold the model
is built on that fold's training plots only and scored on the plots it never saw. That gives an honest
error estimate plus a per-fold spread, rather than one lucky number. When the hyperparameter search is
on, it runs on <b>inner folds</b> <i>inside</i> each outer fold, so the tuning never sees the held-out
plots either — that nesting is the whole point, and it is what makes the reported error trustworthy.</p>
<p>The left column is your data and feature picks; the middle column is the run's settings; the right
column holds the <b>graph</b>, the <b>console</b> and <b>Save</b>.</p>

<h3>Data &amp; features (left)</h3>
<p>The features (X) and targets (y) workbooks load from the project. Tick the feature columns to use;
each shows its training-split <b>range</b> and a <b>direct R</b> (signed single-feature correlation
with the target, so its direction shows). The <b>standardize</b> toggle standardises a feature for
scale-sensitive models (linear/Lasso/Ridge/ElasticNet, SVR, KNN, GPR, PLS); tree models ignore it. The
ticked set is used as-is by every fold — this tab does not choose features for you.</p>

<h3>Model</h3>
<p>Pick the prediction <b>Model</b>; its tunable hyperparameters appear in the form below. Tick
<b>Try every model in sequence and keep the best</b> to run the <i>same</i> configured procedure once
per model in the registry — same features, same splits, same augmented-data settings, so the numbers
are directly comparable — and keep the one with the lowest mean held-out rRMSE. A ranked comparison
table of every model is printed to the console, the winner is selected in the dropdown, and <b>Save</b>
ships that winner. Each model is scored from its own defaults, so leave the search on: otherwise every
model is judged at whatever values happen to be in the form, which means nothing for most of them. The
sweep costs one full run per model.</p>

<h3>Hyperparameters</h3>
<p><b>Search for the best hyperparameters automatically</b> (on by default) runs an Optuna (Bayesian
TPE) search inside every outer fold, scored on that fold's inner cross-validation, and uses the winner
for that fold's held-out fit. The manual form is greyed out because the search is exactly what would
overwrite it; after the run the best fold's values are written into it, so you can see and re-use what
won. <b>Trials</b> is how many combinations each fold tries (total cost = trials × outer folds ×
models) and <b>CPU threads</b> fans a trial's inner folds across cores — speed only, not the result.</p>
<p>Turn the search <b>off</b> to use exactly the values in the form in every fold. The run is then a
plain honest cross-validation of that fixed pipeline, and the inner-CV settings are unused.</p>

<h3>Outer folds &amp; inner cross-validation</h3>
<p>You set the <b>number of outer folds</b> and the <b>outer split ratio</b> freely and pick a
partition mode — <b>Sequential</b> (contiguous blocks), <b>Systematic</b> (every n-th plot, interleaved
so each fold samples the whole field) or <b>Random Systematic</b> (shuffled blocks). All three are
without-replacement partitions (no plot held out twice), which is what makes the estimate honest; plain
Random is deliberately not offered here, because its overlapping draws are not a clean partition. Each
fold holds out <i>ratio × plots</i> plots, so the count and ratio are mutually capped to
<b>splits · ratio ≤ 1</b>; at the product 1 the folds tile the field exactly (every plot held out once)
and below 1 the surplus plots are simply not tested.</p>
<p>The <b>inner</b> controls govern the search's cross-validation <i>inside</i> each outer fold: you set
its split count and ratio and pick <b>Sequential</b> (the default), <b>Systematic</b> or <b>Random
Systematic</b> — making the whole procedure deterministic end to end — or <b>Random</b>, whose
independent draws give inner noise-averaging with an unbounded ratio.</p>

<h3>Target</h3>
<p><b>Fit on the log of the target</b> fits every fold on <code>log1p(target)</code> rather than the raw
value. That is the natural shape for biomass: the quantity is multiplicative and right-skewed, its error
grows with the plot, and on the raw scale a model spends most of its capacity on the few largest plots.
<code>log1p</code> (not <code>log</code>) is used so exact zeros are allowed; a target below −1 is
rejected before the run starts.</p>
<p>Predictions are always brought back to the target's own units, so <b>every</b> metric — here, in the
console, and on every Results view — is in the original dimension. Nothing you read is in log units.</p>
<p>The <b>back-transform</b> choice is where the usual mistake lives. Simply exponentiating a log-scale
prediction returns the conditional <i>median</i>, not the mean: it is low by roughly
<code>exp(σ²/2)</code> for a residual spread σ — a <i>systematic</i> shortfall that does not average out
and lands straight in rRMSE and MAPE. <b>Smearing (Duan)</b>, the default, multiplies by
<code>mean(exp(training residual))</code>, a non-parametric estimate of exactly that missing factor,
computed on each fold's <i>training</i> rows only, so it stays leakage-safe. <b>Plain exponential</b> is
the uncorrected version, kept only so you can see the bias for yourself.</p>
<p>The choice is recorded in the saved bundle and restated on the Results tab's <b>Selection
stability</b> panel, so a model's units are never ambiguous later.</p>

<h3>Augmented data</h3>
<p>The two toggles decide, independently, whether augmented copies count toward <b>fitting</b> and
toward <b>scoring the held-out plots</b>. Scoring on originals only (the default) keeps the headline
number an estimate on real data. Every split is by plot, so a plot's original and augmented copies
never straddle a split — the core leakage guard.</p>

<h3>Graph, console &amp; Save (right)</h3>
<p>The <b>graph</b> has a tab per metric — <b>rRMSE / R² / R / MAPE</b> — each showing the train vs
held-out value per outer fold. (A sweep plots only its first model: several models' curves on one axis
would read as one run's spread — the console table is the sweep's real report.)</p>
<p>The <b>console</b> shows the train and held-out rRMSE while a run is in progress; when it finishes it
prints the full rRMSE / R² / R / MAPE for both, averaged across the outer folds. The small red bin
clears it. <b>Pause</b> and <b>Stop</b> act on the running job; a Stop keeps every fold (or model) that
already finished and discards the one in progress, so a partial result is never biased.</p>
<p><b>Save</b> writes the finished run as a self-describing bundle in the project's <code>model/</code>
folder for the Results tab: its outer folds become the model's splits and the mean held-out rRMSE is
its honest label. The name field is pre-filled with a suggested name (model, averaged rRMSE and a
timestamp); edit it if you like. On the <b>Results</b> tab you then get the full per-fold split
consistency plus a <b>Selection stability</b> tab — the features the model uses and the spread of each
tuned hyperparameter across folds (stable ⇒ trust the number; jumpy ⇒ it leaned on which plots happened
to be held out).</p>
"""


class MLHelpDialog(QDialog):
    """Modeless, read-only help dialog explaining the whole ML tab."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ML — help")
        self.resize(720, 760)
        layout = QVBoxLayout(self)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(_HELP_HTML)
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
