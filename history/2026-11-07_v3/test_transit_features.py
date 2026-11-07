"""Physical sanity checks for flux-only vetting measurements."""
import unittest
import numpy as np
from tess_pipeline import vetting_features


class TransitFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(0, 20, 2/1440)
        self.period, self.duration, self.depth = 2.0, 0.12, 0.01
        self.phase = (self.time + 1) % 2 - 1
        self.inside = np.abs(self.phase) < self.duration/2
        noise = np.random.default_rng(42).normal(0, 0.0001, len(self.time))
        self.flux = 1 + noise - self.depth * self.inside

    def features(self, flux):
        return vetting_features(self.time, flux, self.period, 0, self.duration, self.depth)

    def test_secondary_eclipse_is_detected(self):
        binary = self.flux.copy()
        binary[np.abs(np.abs(self.phase) - 1) < self.duration/2] -= 0.008
        self.assertGreater(self.features(binary)["secondary_ratio"], 0.7)
        self.assertLess(self.features(self.flux)["secondary_ratio"], 0.02)

    def test_alternating_eclipse_depths_raise_odd_even_significance(self):
        binary = self.flux.copy()
        cycle = np.round(self.time / self.period).astype(int)
        binary[self.inside & (cycle % 2 == 1)] -= 0.005
        self.assertGreater(self.features(binary)["odd_even_significance"],
                           self.features(self.flux)["odd_even_significance"] + 20)

    def test_v_shape_differs_from_flat_bottom(self):
        v_shape = self.flux.copy()
        v_shape[self.inside] += self.depth * np.abs(self.phase[self.inside]) / (self.duration/2)
        self.assertGreater(self.features(v_shape)["shape_ratio"],
                           self.features(self.flux)["shape_ratio"] * 2)

    def test_overlapping_transit_windows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "out-of-transit"):
            vetting_features(self.time, self.flux, 0.6, 0, 0.32, self.depth)


if __name__ == "__main__":
    unittest.main()
