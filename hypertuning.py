from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
yaml_path = "data.yaml" 
# Train the model
if __name__ == '__main__':
    model.tune(data=yaml_path, epochs=2, iterations=5, optimizer="AdamW", plots=False, save=False, val=False)