import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import os
import torch
from model import TwinLite as net
import cv2


def Run(model, img):
    img = cv2.resize(img, (1280, 720))
    img_rs = img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    x0 = img_out[0]
    x1 = img_out[1]

    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    img_rs[DA > 100] = [255, 0, 0]
    img_rs[LL > 100] = [0, 255, 0]

    return img_rs


model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('pretrained/best.pth'))
model.eval()

# cap = cv2.VideoCapture('/home/cvlab/Datasets/video  .mp4')  # video setting
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
# fps = 15.0
# delay = round(1000 / fps)
#
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('./ouput.avi', fourcc, fps, (int(width), int(height)))
# while True:
#     retval, frame = cap.read()
#
#     if retval == False:
#         break
#     result = Run(model, frame)
#
#     cv2.imshow("run_model", result)
#
#     if cv2.waitKey(10) == 27:
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()

image_path = "/home/cvlab/Datasets/night3.png"       #night.png

img = cv2.imread(image_path)
img = Run(model, img)
cv2.imshow('result', img)
cv2.imwrite('/home/cvlab/Datasets/OnlyTwin_night.png', img)
# cv2.destroyAllWindows()
# image_list = os.listdir('images')
# shutil.rmtree('results')
# os.mkdir('results')
# for i, imgName in enumerate(image_list):
#     img = cv2.imread(os.path.join('images', imgName))
#     img = Run(model, img)
#     cv2.imwrite(os.path.join('results', imgName), img)