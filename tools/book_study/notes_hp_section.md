
## Ch.9 hyperparameter search (run_hp_search.py, verify_hp_search.py, test_hp_search.py)

Declared before any result. Read-only diagnostic; no model, threshold, calibration file or execution code is touched.

Design:
- Source: the events.csv of an existing 14-feature run (`--run`). Search set = events whose labels end before the walk-forward cutoff (2024-01-01), i.e. exactly the walk-forward training set. Nothing at or after the cutoff is used to choose.
- Grid, fixed in the script: min_weight_fraction_leaf in {0.005, 0.01, 0.025, 0.05, 0.1, 0.2} x max_features in {sqrt, 0.5} = 12 configs (ids 0-11, grid order). 100 trees, sequential bootstrap and max_draws 10000 as in the pipeline. The pipeline setting (0.05, sqrt) is config 6.
- Score: purged 5-fold CV (meta_model.purged_kfold, 1% embargo), pooled out-of-fold sample-weighted log loss. Lowest wins, first in grid order on a tie. AUC is reported, not used to choose. The forest seed of a fold is the same for every config (search seed 20261201 + fold index).
- Post-cutoff: the default and the chosen config are each fitted once on the whole search set and scored on the events starting on or after the cutoff, for the five seeds, with the June 2026 AI-correction and the sell-off (2026-09-27 to 2026-10-08) windows reported separately. If the chosen config is the default there is no comparison.
- Gate: the default at the source run's own seed must reproduce the source walk_forward_predictions.csv within 1e-6 or the run stops (checked in the cloud copy: max difference 5.3e-13).
- Trial count: one new config reaches the post-cutoff data, so this is look 108 (prior 107). If the whole grid should count (12), say so; DSR inputs would change. The post-cutoff period was seen in every earlier run, so it is not a pristine holdout.
- Known risk of the criterion: log loss can rank a near-constant model first when the features carry little signal (the unit test shows this property on synthetic data). Out-of-fold AUC of the quick test was about 0.50. If the winner sits at the grid edge (min_weight_fraction_leaf 0.2) read that as "the search found nothing to tune", not as a better model. AUC is printed next to it for this reason.
- The checker (stdlib only) rebuilds the folds, recomputes every score, rank, post-cutoff metric (not the placebo, which is a seeded shuffle) and every cell of hp_summary.md. The forests are not refitted by the checker.

Tests: planted-error harness on a quick output: 59 of 59 planted errors caught (metrics, scores, ranks, oof, folds, settings, md5, predictions, summary cells, missing/truncated files); one declared blind spot (placebo column) confirmed undetected. test_hp_search.py: 12 tests (10 unit, 2 pipeline with HP_TEST_RUN, one of them with a 2-worker pool).

Commands (project root, conda env javier):
    python tools/book_study/run_hp_search.py --run "$(ls -d runtime/book_seed_pairs/volume_vs_14/ctrl_20261017/*/ | tail -1)" --jobs 2
    python tools/book_study/verify_hp_search.py "$(ls -d runtime/book_hp_search/*/ | tail -1)"

Results: not run yet on the Mac.
