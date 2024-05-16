from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="data.yaml", epochs=2, imgsz=512, device='mps', conf=0, iou=0.7, show=True)  # train the model
metrics = model.val()  # evaluate model performance on the validation set

print(metrics)
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

# model = YOLO("yolov8n.yaml")  # build a new model from scratch

