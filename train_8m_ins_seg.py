from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='my_dataset_seg.yaml', epochs=100, imgsz=640, device=[1])
results = model.train(data='my_dataset_seg.yaml', epochs=300, imgsz=640, device=[1],
                      hsv_h=0.015,
                      hsv_s=0.7,
                      hsv_v=0.4,
                      degrees=0.3,
                      translate=0.1,
                      scale=0.2,
                      mosaic=0.3)

# Export the model to ONNX format
success = model.export(format='onnx')
