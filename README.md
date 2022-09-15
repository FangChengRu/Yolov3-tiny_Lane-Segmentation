### Object Detection + Segmentation model (reference by YOLOP model)

This model is Yolov3-tiny (backbone + detection head) + Segmentation head

And can convert to ONNX and Kneron

Success inference on Kneron KL520

### Model structure

![model](pictures/yolov3Lane.opt.onnx.png)

### Result (Inference on KL520)

![result_day](output_0ace96c3-48481887.jpg)

![result_night](output_3c0e7240-96e390d2.jpg)
