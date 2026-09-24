"""Apple Neural Engine detector, for Frigate running under lighter on a Mac.

lighter (https://github.com/fieldwork-ai/lighter) is a container runtime for
macOS. A container started with `--device lighter.sh/ane=all` gets a library
that runs ONNX models on the Mac's Neural Engine as an ONNX Runtime custom
operator: the library wraps the model into one node, and the node runs the
whole model on the Neural Engine. Everything after the session (model types,
post-processing, warmup) is the ONNX detector's.
"""

import ctypes
import logging
import os
from typing import Literal

import onnxruntime as ort
from pydantic import ConfigDict, Field

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detection_runners import ONNXModelRunner
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.detectors.plugins.onnx import ONNXDetector

logger = logging.getLogger(__name__)

DETECTOR_KEY = "lighter_ane"

# where lighter's device places its library (it also exports the path as
# LIGHTER_ANE_EP)
DEFAULT_LIBRARY = "/usr/lib/lighter/liblighter_ane_ep.so"


class LighterANEDetectorConfig(BaseDetectorConfig):
    """Apple Neural Engine detector for ONNX models, through lighter's lighter.sh/ane device."""

    model_config = ConfigDict(
        title="Apple Neural Engine (lighter)",
    )

    type: Literal[DETECTOR_KEY]
    library: str = Field(
        default_factory=lambda: os.environ.get("LIGHTER_ANE_EP", DEFAULT_LIBRARY),
        title="Neural Engine library",
        description="Path to lighter's Neural Engine library, which the lighter.sh/ane device places in the container.",
    )


def neural_engine_session(library: str, path: str) -> ort.InferenceSession:
    """Open a model on the Neural Engine.

    Args:
        library: Path to lighter's library
        path: Path to the ONNX model

    Returns:
        A session whose one node runs the whole model on the Neural Engine
    """
    lib = ctypes.CDLL(library)
    lib.lighter_ane_wrap.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.lighter_ane_free.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

    with open(path, "rb") as f:
        model = f.read()

    out = ctypes.POINTER(ctypes.c_uint8)()
    size = ctypes.c_size_t()

    if lib.lighter_ane_wrap(model, len(model), ctypes.byref(out), ctypes.byref(size)):
        raise RuntimeError(f"{path} is not an ONNX model lighter can run.")

    wrapped = ctypes.string_at(out, size.value)
    lib.lighter_ane_free(out, size.value)

    options = ort.SessionOptions()
    options.register_custom_ops_library(library)
    return ort.InferenceSession(wrapped, options, providers=["CPUExecutionProvider"])


class LighterANEDetector(DetectionApi):
    type_key = DETECTOR_KEY
    supported_models = [
        ModelTypeEnum.dfine,
        ModelTypeEnum.rfdetr,
        ModelTypeEnum.yolonas,
        ModelTypeEnum.yologeneric,
        ModelTypeEnum.yolox,
    ]

    # the ONNX detector's input handling and post-processing, unchanged; only
    # the session it runs on differs
    detect_raw = ONNXDetector.detect_raw
    _warmup = ONNXDetector._warmup

    def __init__(self, detector_config: LighterANEDetectorConfig):
        super().__init__(detector_config)

        path = detector_config.model.path
        logger.info(f"lighter_ane: loading {path}")

        if not os.path.exists(detector_config.library):
            raise RuntimeError(
                f"lighter's Neural Engine library was not found at {detector_config.library}. Start the container with --device lighter.sh/ane=all."
            )

        self.runner = ONNXModelRunner(
            neural_engine_session(detector_config.library, path),
            detector_config.model.model_type,
        )

        self.onnx_model_type = detector_config.model.model_type
        self.onnx_model_px = detector_config.model.input_pixel_format
        self.onnx_model_shape = detector_config.model.input_tensor

        if self.onnx_model_type == ModelTypeEnum.yolox:
            self.calculate_grids_strides()

        self._warmup(detector_config)
        logger.info(f"lighter_ane: {path} loaded on the Neural Engine")
