"""Regenerates the tiny stand-in models for the offline browser test: pip install onnx numpy; python make_stubs.py

They ignore the picture and always answer "one car, Tesla Model S". That is enough to prove the plumbing (download, storage, wasm,
tensors, post-processing, safety decision) without the real 75 MB of model files. They say nothing about accuracy.
"""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).parent
NAMES = (HERE / "../../../data/classnames.txt").resolve().read_text(encoding="utf-8").strip().splitlines()
TESLA = NAMES.index("Tesla Model S Sedan 2012")
DIM = 8


def save(name, input_name, input_shape, nodes, outputs, inits):
    graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)], outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, HERE / name)


# Detector: 1 x 84 x 8400, one confident "car" (COCO class 2) in the first column, zeros elsewhere.
column = np.zeros((1, 84, 1), np.float32)
column[0, :4, 0] = [320, 320, 500, 300]
column[0, 4 + 2, 0] = 0.9
save("yolo11s.onnx", "images", [1, 3, 640, 640],
     [helper.make_node("Pad", ["column", "pads"], ["output0"], mode="constant")],
     [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])],
     [numpy_helper.from_array(column, "column"), numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 8399], np.int64), "pads")])

# Classifier: logits peaked on the Tesla, and an embedding identical to the first training embeddings (unknown score 0).
logits = np.zeros((1, len(NAMES)), np.float32)
logits[0, TESLA] = 12
emb = np.zeros((1, DIM), np.float32)
emb[0, 0] = 1
save("classifier_int8.onnx", "input", [1, 3, 384, 384],
     [helper.make_node("Identity", ["l"], ["logits"]), helper.make_node("Identity", ["e"], ["embedding"])],
     [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, len(NAMES)]), helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, DIM])],
     [numpy_helper.from_array(logits, "l"), numpy_helper.from_array(emb, "e")])

bank = np.zeros((4, DIM), np.float32)
bank[:, 0] = 1
np.save(HERE / "train_emb.npy", bank)
(HERE / "config.json").write_text(json.dumps({
    "class_names": NAMES, "img": 384, "temperature": 1.0, "knn_k": 2, "unknown_threshold": 0.5, "hv_threshold": 0.05,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
}), encoding="utf-8")
