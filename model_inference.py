# coding:utf-8
import os
import torch
import torchvision
from models.backbone import Backbone
from utils import transforms
from imutils.video import fps
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch import nn

classes = ["0", "1"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = ""

val_transforms_list = [
    torchvision.transforms.Resize(size=(224, 224)),
    transforms.ZeroOneNormalize()
]
val_transforms = torchvision.transforms.Compose(val_transforms_list)

backbone = Backbone(out_dimension=len(classes), model_name="resnet50")
model, _, _ = backbone.build_model()
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
model.to(device)
model.eval()

image_path = ""
fps = fps.FPS()
fps.start()

with torch.no_grad():
    for image_name in os.listdir(image_path):
        file_path = os.path.join(image_path, image_name)
        img = read_image(file_path, mode=ImageReadMode.RGB)
        img = img.to(device)
        img = img.unsqueeze(dim=0)
        img = val_transforms(img).to(device)

        res = model(img)
        cls_index = res.argmax(dim=1)
        cls_prob = nn.functional.softmax(res, dim=1)

        pred_prob = cls_prob[0][cls_index].item()
        pred_cls = classes[cls_index]

        print(pred_prob, pred_cls)

        fps.update()
fps.stop()
print("FPS: {}".format(fps.fps()))
print("time: {}".format(fps.elapsed()))
