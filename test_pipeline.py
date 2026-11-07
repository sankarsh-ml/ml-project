"""Scientific and leakage safeguards for the improved real-data workflow."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from search_pipeline import detrend, search, detection_accepted
from transit_features import centroid_features, phase_time, FEATURES
from training_workflow import match_ephemeris
from model_selection import feature_matrix, grouped_folds, train as fit_model
from target_selector import TargetIndex, parse_tic
from experiment import RUN_DIR


class PipelineTests(unittest.TestCase):
    def test_transit_preserving_detrend_with_gap(self):
        t = np.arange(0, 20, 2/1440)
        t = t[(t < 8) | (t > 8.7)]
        inside = np.abs(phase_time(t, 2.7, .5)) < .06
        flux = 1+.01*np.sin(2*np.pi*t/8)-.005*inside+np.random.default_rng(7).normal(0, .0003, len(t))
        flat, trend = detrend(t, flux, inside)
        recovered = np.median(flat[~inside])-np.median(flat[inside])
        self.assertAlmostEqual(recovered, .005, delta=.001)
        self.assertTrue(np.isfinite(trend).all())

    def test_blind_bls_recovers_known_period(self):
        t = np.arange(0, 20, 2/1440)
        flux = 1+np.random.default_rng(12).normal(0, .0005, len(t))
        flux[np.abs(phase_time(t, 2.713, .37)) < .06] -= .008
        candidates, _ = search(t, flux, np.full(len(t), .0005), lambda _: None)
        self.assertAlmostEqual(candidates[0]["period"], 2.713, delta=.01)
        self.assertGreater(candidates[0]["snr"], 20)

    def test_catalog_association_uses_bjd_conversion_and_rejects_wrong_phase(self):
        catalog = {"pl_orbper": 3., "pl_tranmid": 2458325., "pl_trandurh": 2.,
                   "pl_orbpererr1": 1e-6, "pl_tranmiderr1": .001}
        feature = {"period_days": 3., "epoch_btjd": 1325.}
        self.assertTrue(match_ephemeris(feature, catalog)["catalog_match"])
        self.assertTrue(match_ephemeris({**feature, "period_days": 1.5}, catalog)["catalog_match"])
        self.assertFalse(match_ephemeris({**feature, "epoch_btjd": 1326.}, catalog)["catalog_match"])

    def test_centroid_shift_flags_blended_signal(self):
        t = np.arange(0, 15, 2/1440)
        phase = phase_time(t, 2.4, .3)
        noise = np.random.default_rng(4).normal(0, .0001, len(t))
        shifted = noise+.002*(np.abs(phase) < .06)
        strong = centroid_features(t, [shifted, noise], phase, .12)
        weak = centroid_features(t, [noise, noise], phase, .12)
        self.assertGreater(strong["centroid_significance"], weak["centroid_significance"]+20)

    def test_actual_frozen_assignment_is_star_disjoint(self):
        plan = json.loads((RUN_DIR / "split_plan.json").read_text())
        train, test = set(plan["train_tic_ids"]), set(plan["test_tic_ids"])
        self.assertFalse(train & test)
        self.assertFalse(test & set(plan["legacy_seen_tic_ids"]))
        self.assertFalse(train & set(plan["quarantined_development_tic_ids"]))

    def test_model_inputs_exclude_catalog_labels_and_identifiers(self):
        row = {name: 1. for name in FEATURES}
        row.update(period_days=2., duration_days=.1, depth=.01, flux_std=.001, duty_cycle=.05,
                   tic_id="123", label=1, catalog_period_days=999.)
        x = feature_matrix(pd.DataFrame([row]))
        self.assertEqual(list(x), FEATURES)
        self.assertNotIn("tic_id", x)
        self.assertNotIn("catalog_period_days", x)

    def test_grouped_cv_keeps_repeated_stars_together(self):
        groups = np.repeat(np.arange(40).astype(str), 2)
        y = np.repeat(np.arange(40) % 2, 2)
        for fit, valid in grouped_folds(np.zeros((80, 2)), y, groups):
            self.assertFalse(set(groups[fit]) & set(groups[valid]))

    def test_selector_accepts_new_manual_ids_and_filters_known_records(self):
        self.assertEqual(parse_tic("TIC 000123456"), "123456")
        with self.assertRaises(ValueError):
            parse_tic("TIC -2")
        index = TargetIndex()
        matches = index.search("149603")
        self.assertIn("149603524", [row["tic_id"] for row in matches])
        self.assertIn(1, index.get("149603524")["sectors"])
        self.assertIsNone(index.get("9999999999999999"))

    def test_epoch_projection_uses_catalog_period_over_many_years(self):
        catalog = {"pl_orbper": 3., "pl_tranmid": 2461325., "pl_trandurh": 2.,
                   "pl_orbpererr1": 1e-8, "pl_tranmiderr1": .001}
        # A 0.03% one-sector period error accumulates ~0.9 d over 1000 cycles.
        # The real transit at BTJD 1325 must still match the catalog ephemeris.
        for period, epoch in [(3.0009, 1325.), (1.50045, 1325.), (6.0018, 1328.)]:
            self.assertTrue(match_ephemeris({"period_days": period, "epoch_btjd": epoch}, catalog)["catalog_match"])

    def test_weak_signals_fail_predeclared_gate(self):
        f = {"bls_snr": 4., "pink_snr": 2., "observed_transits": 3, "event_fraction": .8}
        self.assertFalse(detection_accepted(f))
        self.assertTrue(detection_accepted({**f, "bls_snr": 20., "pink_snr": 10.}))

    def test_training_refuses_an_already_open_independent_protocol(self):
        if not (RUN_DIR / "evaluation_lock.json").exists():
            self.skipTest("The protocol has not been opened yet.")
        with self.assertRaisesRegex(RuntimeError, "independent test is already open"):
            fit_model(pd.DataFrame(), "outputs/forbidden_retrain.joblib")


if __name__ == "__main__":
    unittest.main()
