from PIL import Image
from ultralytics import YOLO
import os
import cv2

import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm

if __name__ == '__main__':
    # Load a pretrained YOLOv8n model
    model = YOLO('input/your/model.pt')

    directory_path = "input/your/video.mp4"
    file_paths = []

    # os.walk를 사용하여 디렉토리 안의 모든 파일 및 하위 디렉토리의 경로를 읽어옵니다.
    for root, directories, files in os.walk(directory_path):
        for filename in files:
            # 각 파일의 전체 경로를 생성합니다.
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)

    file_paths.sort()
    results = model(file_paths)  # results list

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    w = 1280
    h = 720

    # 웹캠으로 찰영한 영상을 저장하기
    fps = 10.
    # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
    out = cv2.VideoWriter('input/dir/result.avi', fourcc, fps, (w, h))

    # Show the results
    for r in results:
        im_array = r.plot(masks=True, boxes=False)
        out.write(im_array)
