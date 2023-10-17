from ultralytics import YOLO

# Load a model
model = YOLO('model/best_yolo.pt')  # load a custom trained model

# Export the model
model.export(format='engine')
