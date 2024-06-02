from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("runs/segment/train5/weights/last.pt")
# Use the model
model.train(data="/srv/submission/stenosis/data.yaml", epochs=50, imgsz=512, conf=0.01, iou=0.99, device=0)  # train the model
metrics = model.val()  # evaluate model performance on the validation set

print(metrics)
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

# model = YOLO("yolov8n.yaml")  # build a new model from scratch

