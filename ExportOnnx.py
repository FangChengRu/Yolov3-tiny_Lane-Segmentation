from lib.models import get_net
from lib.config import cfg
import torch
import onnx

weights_path = './weights/version2.pth'

export = True  # Export without yolo layer
# Load model
model = get_net(export)
checkpoint = torch.load(weights_path, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
print(model)

dummy_input = torch.randn(1, 3, 416, 416).to('cpu')
torch.onnx.export(model, dummy_input, 'yolov3Lane_v2.onnx', opset_version=11,
                      output_names=['yolo', 'yolo', 'drive_area_seg', 'lane_line_seg'])
print('convert', 'yolov3Lane_v2.onnx', 'to onnx finish!!!')
# Checks
model_onnx = onnx.load('yolov3Lane_v2.onnx')  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model
print(onnx.helper.printable_graph(model_onnx.graph))  # print
