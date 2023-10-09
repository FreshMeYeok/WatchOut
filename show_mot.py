import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import torch
from model import TwinLite as net
import cv2


def Run(model, img):
    img = cv2.resize(img, (640, 360))
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

if __name__ == '__main__':

    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('pretrained/best.pth'))    # test_/model_1.pth
    model.eval()

    fps = 30.0
    delay = round(1000/ fps)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('./mot_result_0000f77c-cb820c98.avi', fourcc, fps,(640, 360))

    # 이미지 파일을 읽어올 폴더 경로 설정
    folder_path = '/home/cvlab/Datasets/bdd100k/images/track/train/0000f77c-cb820c98'

    # 폴더 내의 모든 파일 리스트를 가져옵니다.
    file_list = os.listdir(folder_path)
    file_list.sort()
    # jpg 및 png 파일만 처리하려면 다음과 같이 반복문을 사용합니다.
    for file_name in file_list:
        # 파일의 확장자를 가져옵니다.
        file_extension = file_name.split('.')[-1].lower()

        # jpg 또는 png 파일인 경우에만 처리
        if file_extension in ['jpg', 'jpeg', 'png']:
            # 파일의 전체 경로 생성
            file_path = os.path.join(folder_path, file_name)
            try:
                image = cv2.imread(file_path)
                # 여기서 이미지에 대한 작업을 수행하면 됩니다.
                result = Run(model, image)
                # cv2.imshow('image', image)
                #
                # cv2.imshow('result', result)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # 예시: 이미지 크기 출력
                print(f'이미지 파일: {file_name}, 크기: {image.shape}')
                out.write(result)
            except Exception as e:
                # 이미지 열기에 실패한 경우 처리
                print(f'{file_name} 열기 실패: {str(e)}')