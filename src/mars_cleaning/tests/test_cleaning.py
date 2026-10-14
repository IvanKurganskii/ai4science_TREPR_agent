import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from .. import CleaningEngine, MarsCleaningTools, TOOL_SCHEMAS
from .._runtime import numpy
from ..data import EPRDataset, NumpyUnpickler


class CleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = numpy()
        cls.engine = CleaningEngine()
        b, t = cls.np.linspace(.33, .37, 61), cls.np.linspace(-1e-6, 5e-6, 25)
        shape = cls.np.zeros(61)
        shape[24:37] = cls.np.sin(cls.np.linspace(0, 2*cls.np.pi, 13))
        weights = cls.np.zeros(25)
        weights[5:22] = cls.np.exp(-cls.np.arange(17)/7)
        cls.true = weights[:, None]*shape[None, :]
        background = .2+.04*cls.np.cos(cls.np.arange(61)/4)
        baseline = cls.np.zeros_like(cls.true)
        for i in range(5, 22):
            baseline[i] = (i/20)*(1+3*(b-.35))+.1
        cls.raw = EPRDataset.from_arrays(b, t, cls.true+background+baseline, name="synthetic")
        cls.options = {"field_bounds_T": [.345, .355], "time_bounds_s": [float(t[5]), float(t[21])], "poly_order": 1}

    def test_baseline_recovers_signal_both_modes(self):
        for mode in ("scaled_reference", "independent"):
            with self.subTest(mode=mode):
                cleaned, diagnostic = self.engine.correct_baseline(self.raw, {**self.options, "baseline_mode": mode})
                self.np.testing.assert_allclose(cleaned.signal, self.true, atol=2e-8)
                self.np.testing.assert_allclose(self.raw.signal-cleaned.signal,
                    diagnostic["background_profile"][None, :]+diagnostic["baseline_surface"], atol=2e-8)

    def test_one_sided_background_no_crop(self):
        options = {"correct_baseline": False, "time_bounds_s": [0.0, None]}
        cleaned, _ = self.engine.correct_baseline(self.raw, options)
        self.np.testing.assert_allclose(cleaned.signal, self.raw.signal-self.raw.signal[self.raw.times_s < 0].mean(axis=0))
        self.assertEqual(cleaned.signal.shape, self.raw.signal.shape)
        with self.assertRaises(ValueError):
            self.engine.correct_baseline(self.raw, {**options, "time_bounds_s": [None, None]})

    def test_all_filters_match_mars_with_correct_transposes(self):
        from mars.spectra_processing import filter_noise_2d
        data = self.raw.copy(self.true+self.np.random.default_rng(42).normal(0, .01, self.true.shape))
        variants = [{"method": "svd", "svd_rank": 2}, {"method": "gaussian", "sigma_t": 1., "sigma_b": .5},
                    {"method": "wavelet", "wavelet": "db2", "wavelet_level": 1, "wavelet_threshold": .01},
                    {"method": "median", "median_kernel": 3},
                    {"method": "combined", "svd_rank": 2, "sigma_t": 1., "sigma_b": .5}]
        for options in variants:
            with self.subTest(method=options["method"]):
                actual = self.engine.denoise(data, options)
                expected = filter_noise_2d(data.fields_T, data.times_s, data.signal.T, **options).T
                self.np.testing.assert_allclose(actual.signal, expected, rtol=1e-11, atol=1e-12)

    def test_pipeline_equals_sequential_and_preserves_input(self):
        original = self.raw.signal.copy()
        sequential, _ = self.engine.correct_baseline(self.raw, self.options)
        sequential = self.engine.denoise(sequential, {"method": "svd", "svd_rank": 1})
        together, diag = self.engine.clean(self.raw, baseline=self.options, denoising={"method": "svd", "svd_rank": 1})
        self.np.testing.assert_allclose(sequential.signal, together.signal)
        self.np.testing.assert_array_equal(self.raw.signal, original)
        self.assertIn("baseline_corrected_signal", diag)
        self.assertEqual([s["stage"] for s in together.metadata["history"]], ["baseline", "denoising"])

    def test_layout_sorting_and_validation(self):
        transposed = EPRDataset.from_arrays(self.raw.fields_T[::-1], self.raw.times_s[::-1], self.raw.signal[::-1, ::-1].T, layout="field_time")
        self.np.testing.assert_array_equal(transposed.signal, self.raw.signal)
        with self.assertRaises(ValueError):
            EPRDataset.from_arrays([1, 2], [0, 1], [[1, 2], [3, 4]], layout="auto")
        for options in ({"method": "svd", "svd_rank": 0}, {"method": "median", "median_kernel": 4},
                        {"method": "svd", "sigma_t": 2}, {"method": "wavelet", "wavelet": "invalid"}):
            with self.assertRaises(ValueError):
                self.engine.denoise(self.raw, options)

    def test_csv_pkl_npz_roundtrip_and_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.np.savez_compressed(root/"input.npz", fields_T=self.raw.fields_T, times_s=self.raw.times_s,
                signal=self.raw.signal, metadata_json=self.np.asarray(json.dumps({"dataset": "synthetic", "parameters": {"p": {"Temperature": 80}}})))
            tools = MarsCleaningTools(working_dir=root)
            result = tools.export_epr_2d({"path": "input.npz"}, output={"formats": ["npz", "pkl", "csv"], "save_plot": False})
            for extension in ("npz", "pkl", "csv"):
                source = {"path": result["artifacts"][extension]}
                if extension == "csv":
                    source["time_unit"] = "ns"
                loaded = tools.loader.load(source)
                self.np.testing.assert_allclose(loaded.fields_T, self.raw.fields_T)
                self.np.testing.assert_allclose(loaded.times_s, self.raw.times_s, atol=1e-20)
                self.np.testing.assert_allclose(loaded.signal, self.raw.signal)
                if extension != "csv":
                    self.assertEqual(loaded.metadata["parameters"]["p"]["Temperature"], 80)
                    self.assertEqual(loaded.metadata["source_history"][0]["path"], str(root/"input.npz"))

    def test_all_six_tools_artifacts_and_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.np.savez_compressed(root/"input.npz", fields_T=self.raw.fields_T, times_s=self.raw.times_s, signal=self.raw.signal)
            tools = MarsCleaningTools(working_dir=root)
            self.assertEqual(tools.output_dir, root/"results"/"mars_simulations")
            self.assertEqual(len(TOOL_SCHEMAS), 6)
            for schema in TOOL_SCHEMAS:
                name = schema["function"]["name"]
                args = {"source": {"path": "input.npz"}}
                if name == "inspect_epr_2d":
                    args["singular_values"] = 5
                else:
                    args["output"] = {"save_plot": name == "clean_epr_2d"}
                if name in ("clean_epr_2d", "correct_epr_2d_baseline"):
                    args["baseline"] = self.options
                if name in ("clean_epr_2d", "denoise_epr_2d"):
                    args["denoising"] = {"method": "svd", "svd_rank": 1}
                result = tools.dispatch(name, json.dumps(args))
                self.assertTrue(result["ok"], result)
                json.dumps(result, allow_nan=False)
                if name == "clean_epr_2d":
                    self.assertTrue(Path(result["artifacts"]["plot"]).is_file())
            self.assertFalse(tools.dispatch("import_epr_2d", {"source": {"path": "input.npz"}, "output": {"name": "../elsewhere"}})["ok"])

    def test_pickle_rejects_arbitrary_classes(self):
        with self.assertRaises(pickle.UnpicklingError):
            NumpyUnpickler(io.BytesIO(pickle.dumps(Path("not-an-array")))).load()


if __name__ == "__main__":
    unittest.main()
