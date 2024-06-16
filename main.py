import cshogi
import numpy as np
import onnxruntime

from cshogi._cshogi import _dlshogi_FEATURES1_NUM as FEATURES1_NUM
from cshogi._cshogi import _dlshogi_FEATURES2_NUM as FEATURES2_NUM
from cshogi.dlshogi import make_input_features

board = cshogi.Board()
feature1 = np.empty((1, FEATURES1_NUM, 9, 9), dtype='f4')
feature2 = np.empty((1, FEATURES2_NUM, 9, 9), dtype='f4')
make_input_features(board, feature1[0], feature2[0])

session = onnxruntime.InferenceSession("model.onnx")
io_binding = session.io_binding()
io_binding.bind_cpu_input("input1", feature1)
io_binding.bind_cpu_input("input2", feature2)
io_binding.bind_output("output_policy")
io_binding.bind_output("output_value")
session.run_with_iobinding(io_binding)

policy, value = io_binding.copy_outputs_to_cpu()
print(value)