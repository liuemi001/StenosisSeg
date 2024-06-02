from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("runs/segment/train5/weights/last.pt")
# Use the model
<<<<<<< HEAD:stenosis_yolo.py
model.train(data="/srv/submission/stenosis/data.yaml", epochs=50, imgsz=512, conf=0.01, iou=0.99, device=0)  # train the model
=======
model.train(data="data.yaml", epochs=2, imgsz=512, device='mps', conf=0, iou=0.7, show=True)  # train the model
>>>>>>> da22f9c76ef62bfd7d20d36e6c76cdf12fd11e0f:stenosis/stenosis_yolo.py
metrics = model.val()  # evaluate model performance on the validation set

print(metrics)
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

# model = YOLO("yolov8n.yaml")  # build a new model from scratch

