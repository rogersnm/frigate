"""Tests for the Apple Neural Engine detector reached through lighter."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from frigate.detectors.detector_config import ModelConfig
from frigate.detectors.plugins import lighter_ane


def detector_config(**fields):
    return lighter_ane.LighterANEDetectorConfig(
        type="lighter_ane", model=ModelConfig(), **fields
    )


class TestLighterANEConfig(unittest.TestCase):
    def test_the_library_defaults_to_where_the_device_places_it(self):
        with patch.dict(os.environ, {}, clear=True):
            config = detector_config()

        self.assertEqual(config.library, lighter_ane.DEFAULT_LIBRARY)

    def test_the_library_follows_the_path_the_device_exports(self):
        with patch.dict(os.environ, {"LIGHTER_ANE_EP": "/opt/ep.so"}):
            config = detector_config()

        self.assertEqual(config.library, "/opt/ep.so")

    def test_the_library_can_be_set_in_the_config(self):
        self.assertEqual(detector_config(library="/x/ep.so").library, "/x/ep.so")


class TestLighterANEDetector(unittest.TestCase):
    def _config(self, library: str):
        return detector_config(library=library)

    def test_a_missing_library_says_how_to_start_the_container(self):
        with self.assertRaisesRegex(RuntimeError, "lighter.sh/ane"):
            lighter_ane.LighterANEDetector(self._config("/nonexistent/ep.so"))

    def test_a_model_the_library_cannot_wrap_is_reported(self):
        with (
            tempfile.NamedTemporaryFile() as library,
            tempfile.NamedTemporaryFile() as model,
        ):
            lib = MagicMock()
            lib.lighter_ane_wrap.return_value = -1

            with patch.object(lighter_ane.ctypes, "CDLL", return_value=lib):
                with self.assertRaisesRegex(RuntimeError, "not an ONNX model"):
                    lighter_ane.neural_engine_session(library.name, model.name)

    def test_the_wrapped_model_runs_with_the_library_registered(self):
        with tempfile.NamedTemporaryFile() as model:
            model.write(b"model")
            model.flush()
            lib = MagicMock()
            lib.lighter_ane_wrap.return_value = 0
            ort = MagicMock()

            with (
                patch.object(lighter_ane.ctypes, "CDLL", return_value=lib),
                patch.object(lighter_ane.ctypes, "string_at", return_value=b"wrapped"),
                patch.object(lighter_ane, "ort", ort),
            ):
                lighter_ane.neural_engine_session("/ep.so", model.name)

            ort.SessionOptions.return_value.register_custom_ops_library.assert_called_once_with(
                "/ep.so"
            )
            self.assertEqual(ort.InferenceSession.call_args.args[0], b"wrapped")
            lib.lighter_ane_free.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
