from __future__ import division
import numpy as np
import torch
from model import TwinLite as net
import cv2
from ultralytics import YOLO
import time
import math
from shapely.geometry import Polygon
import os
import time
import csv
import signal
import sys

default_space = np.array([[320, 720], [960, 720], [700, 450], [500, 450]])
half_w = default_space[0][0] + (default_space[1][0] - default_space[0][0]) // 2

def Run(model,img):
    # img = cv2.resize(img, (640, 360))

    img_rs=img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    # x0=img_out[0]
    x1=img_out[1]

    # _,da_predict=torch.max(x0, 1)
    _,ll_predict=torch.max(x1, 1)

    # DA = da_predict.byte().cpu().data.numpy()[0]*255
    LL = ll_predict.byte().cpu().data.numpy()[0]*255
    # img_rs[DA>100]=[255,0,0]
    img_rs[LL>100]=[0,255,0]
    
    return img_rs\
# Color
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
deepgray = (43, 43, 43)
dark = (1, 1, 1)
cyan = (255, 255, 0)
magenta = (255, 0, 255)
lime = (0, 255, 128)
purple = (255, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

# Global 함수 초기화
l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0
next_frame = (0, 0, 0, 0, 0, 0, 0, 0)

def save_video(filename, frame=20.0):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(filename, fourcc, frame, (1280,720))
    return out

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

hei = 25
# alpha = int(args.alpha)
font_size = 1

""" 현재 영상 프레임 표시 """
def show_fps(image, frames, start, color = white):
    now_fps = round(frames / (time.time() - start), 2)
    cv2.putText(image, "FPS : %.2f"%now_fps, (10, hei), font, 0.8, color, font_size)

"""Bird's eye view 적용을 위한 ROI image 반환"""
def ROI_BEV(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    # vertiecs로 만든 polygon으로 이미지의 ROI를 정하고 ROI 이외의 영역은 모두 검정색으로 정한다.
    masked_image = cv2.bitwise_and(img, mask)
    # cv2.imshow('masked', masked_image)
    return masked_image

"""roi image를 BEV이미지으로 변환"""
def roi2bev(roi_img, vertices):     # Bird's eye view
    pts1 = np.float32([vertices[1], vertices[0], vertices[2], vertices[3]])
    pts2 = np.float32([[425, 0], [425, 720], [855, 0], [855, 720]])
    M = cv2.getPerspectiveTransform(pts1, pts2) + np.random.rand(3, 3) * 1e-9
    dst = cv2.warpPerspective(roi_img, M, (1280, 720))
    # cv2.imshow("bdffdd", dst)
    return dst, M

"""bev image를 roi image로 변환"""
def bev2roi(bev_img, M, vertices):
    bev2roi_start = time.time()
    linV = np.linalg.inv(M)
    lindst = cv2.warpPerspective(bev_img, linV, (1280, 720))

    # Create a mask of the region of interest using vertices
    mask = np.zeros_like(lindst)
    cv2.fillPoly(mask, [vertices], white)  # Fill the region with white

    # Apply the mask to the lindst image
    lindst = cv2.bitwise_and(lindst, mask)


    vertices, points = find_white_contour_vertices(lindst, vertices)
    # cv2.imshow('lindst', lindst)
    bev2roi_end = time.time()
    bev2roi_time = round(bev2roi_end - bev2roi_start, 3)
    print(f'bev2roi run time : {bev2roi_time}')
    return lindst, vertices, points


def find_white_contour_vertices(image, vertices):
    copy_frame = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask of white areas
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    # Detect corners using cornerHarris
    dst = cv2.cornerHarris(thresh, 2, 3, 0.04)

    # Find white contours
    white_contours = np.argwhere(dst > 0.01 * dst.max())

    y_list = [y for y, x in white_contours if y >= vertices[1][1]]

    if not y_list:
        points = default_space
    else:
        max_y_value = max(y_list)
        min_y_value = min(y_list)

        bottom_candidate = [x for y, x in white_contours if y == max_y_value]
        top_candidate = [x for y, x in white_contours if y == min_y_value]

        if not bottom_candidate or not top_candidate:
            points = default_space
        elif len(bottom_candidate) == 1 and len(top_candidate) > 1:
            bottom_x = bottom_candidate[0]
            bottom_x_another = half_w - (bottom_x - half_w) if bottom_x >= half_w else half_w + (half_w - bottom_x)
            max_bottom = [bottom_x, max_y_value]
            min_bottom = [bottom_x_another, max_y_value]
            max_top = [max(top_candidate), min_y_value]
            min_top = [min(top_candidate), min_y_value]
            points = np.array([max_bottom, min_bottom, min_top, max_top])
        elif len(bottom_candidate) > 1 and len (top_candidate) == 1:
            top_x = top_candidate[0]
            top_x_another = half_w - (top_x - half_w) if top_x >= half_w else half_w + (half_w - top_x)
            max_bottom = [max(bottom_candidate), max_y_value]
            min_bottom = [min(bottom_candidate), max_y_value]
            max_top = [top_x, min_y_value]
            min_top = [top_x_another, min_y_value]
            points = np.array([max_bottom, min_bottom, min_top, max_top])
        elif len(bottom_candidate) == 1 and len(top_candidate) == 1:
            bottom_x = bottom_candidate[0]
            top_x = top_candidate[0]
            bottom_x_another = half_w - (bottom_x - half_w) if bottom_x >= half_w else half_w + (half_w - bottom_x)
            top_x_another = half_w - (top_x - half_w) if top_x >= half_w else half_w + (half_w - top_x)
            max_bottom = [bottom_x, max_y_value]
            min_bottom = [bottom_x_another, max_y_value]
            max_top = [top_x, min_y_value]
            min_top = [top_x_another, min_y_value]
            points = np.array([max_bottom, min_bottom, min_top, max_top])
        else:
            max_bottom = [max(bottom_candidate), max_y_value]
            min_bottom = [min(bottom_candidate), max_y_value]
            max_top = [max(top_candidate), min_y_value]
            min_top = [min(top_candidate), min_y_value]
            points = np.array([max_bottom, min_bottom, min_top, max_top])

    return copy_frame, points

def auto_canny(image, kernel_size ,sigma = 0.33):
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    v = np.mean(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def preprocess(img):

    kernel_size = 3

    rho = 2
    theta = np.pi / 180
    thresh = 50  # 100
    min_line_len = 100  # 50
    max_line_gap = 150
    gray_image = grayscale(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_green = np.array([30, 80, 80], dtype="uint8")
    upper_green = np.array([70, 255, 255], dtype="uint8")

    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_green)  # Grayscale로 변환한 원본 이미지에서 흰색과 노란색만 추출

    canny_edges = auto_canny(mask_yw_image, kernel_size)

    preprocess_start = time.time()
    line_image, lane_space = convert_hough(canny_edges, rho, theta, thresh, min_line_len, max_line_gap, vertices)
    # cv2.imshow('line_image', line_image)

    preprocess_end = time.time()
    preprocess_time = round(preprocess_end - preprocess_start, 3)
    print(f"preprocess 실행 시간: {preprocess_time} 초")
    result = weighted_img(line_image, img, α=1., β=1., λ=0.)
    # cv2.polylines(result, vertices, True, (0, 255, 255)) # ROI mask

    return result, line_image, lane_space

def convert_hough(img, rho, theta, threshold, min_line_len, max_line_gap, vertices):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # hough_start = time.time()
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lane_space = draw_lanes(line_img, lines, vertices)
    # hough_end = time.time()
    # hough_run = round(hough_end - hough_start, 3)
    # print(f"hough 실행 시간: {hough_run} 초")
    return line_img, lane_space

def ransac_line_fit(points_start, points_end, iterations=100, threshold=10):
    best_line = None
    best_inliers = []

    for _ in range(iterations):
        # 임의로 두 점 선택
        sample = np.random.randint(len(points_start))
        # 선 모델 매개변수 계산 (여기서는 기울기와 y 절편)
        x1, y1 = points_start[sample]
        x2, y2 = points_end[sample]
        if x1 == x2:
            continue  # 기울기가 무한대인 경우 건너뜀
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1

        # 모델을 사용하여 모든 점을 평가하고 임계값 이내의 점을 인라이어로 선택
        inliers = []

        for point in points_start:
            x, y = point
            distance = abs(y - (slope * x + y_intercept))
            if distance < threshold:
                inliers.append(point)

        for point in points_end:
            x, y = point
            distance = abs(y - (slope * x + y_intercept))
            if distance < threshold:
                inliers.append(point)

        # 현재 모델의 인라이어 수가 최고 모델보다 많으면 업데이트
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (slope, y_intercept)

    return best_line, best_inliers
def draw_lanes(img, lines ,vertices):
    global cache
    global first_frame
    global next_frame
    # global prev_lane

    # y_global_min = img.shape[0]
    y_global_min = 0
    y_max = img.shape[0]
    l_slope, r_slope = [], []
    l_lane, r_lane = [], []
    l_lane_start, l_lane_end = [], []
    r_lane_start, r_lane_end = [], []

    det_slope = 0.5
    α = 0.2

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = get_slope(x1, y1, x2, y2)
                if slope == math.inf or slope == -math.inf:
                    pass
                elif slope > det_slope:
                    r_slope.append(slope)
                    r_lane.append((x1,y1))
                    r_lane.append((x2,y2))
                    r_lane_start.append((x1, y1))
                    r_lane_end.append((x2, y2))
                elif slope < -det_slope:
                    l_slope.append(slope)
                    l_lane.append((x1, y1))
                    l_lane.append((x2, y2))
                    l_lane_start.append((x1, y1))
                    l_lane_end.append((x2, y2))
        # y_global_min = min(y1, y2, y_global_min)
    else:
        cv2.fillPoly(img, [vertices], (102, 000, 51))
    if (len(l_lane) == 0 or len(r_lane) == 0):  # 오류 방지
        return 1

    left_line, left_inliers = ransac_line_fit(np.array(l_lane_start), np.array(l_lane_end), iterations=100, threshold=3)
    right_line, right_inliers = ransac_line_fit(np.array(r_lane_start), np.array(r_lane_end), iterations=100, threshold=3)

    if left_line is not None and right_line is not None:
        l_slope, l_y_intercept = left_line
        r_slope, r_y_intercept = right_line

        l_x1 = int((y_global_min - l_y_intercept) / l_slope)
        l_x2 = int((y_max - l_y_intercept) / l_slope)
        r_x1 = int((y_global_min - r_y_intercept) / r_slope)
        r_x2 = int((y_max - r_y_intercept) / r_slope)

        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype="float32")
    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0
    else:
        prev_frame = cache
        next_frame = (1-α)*prev_frame+α*current_frame

    lane_space = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]],
     [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], dtype=int)
    cv2.fillPoly(img, [lane_space], (204, 255, 204))

    cache = next_frame

    return lane_space

def set_safetyzone(points, divider):
    max_bottom, min_bottom, min_top, max_top = points
    x1, y1 = max_top
    x2, y2 = max_bottom
    a = (y2 - y1)/(x2 - x1)
    b = y1 - a*x1
    new_left_x = min(min_bottom[0], min_top[0]) + int(abs(min_top[0] - min_bottom[0]) / divider)
    new_y = max_bottom[1] - int((max_bottom[1] - max_top[1]) / divider)
    new_right_x = (new_y - b) / a
    if not math.isnan(new_right_x):
        new_right_x = int(new_right_x)
    else:
        new_right_x = 360
    new_left = [new_left_x, new_y]
    new_right = [new_right_x, new_y]
    new_points = np.array([max_bottom, min_bottom, new_left, new_right])
    return new_points

def sigint_handler(signal, frame):
    # Ctrl+C로 종료 시 CSV 파일로 데이터 저장
    with open('execution_times.csv', 'w', newline='') as csv_file:
        fieldnames = ["Process Time", "Yolo Time", "Twin Time"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    sys.exit(0)

def find_center_of_points(points):
    """
    주어진 좌표 배열의 중심점을 찾는 함수
    :param points: (N, 2) 형태의 NumPy 배열. 각 행은 (x, y) 좌표를 나타냄.
    :return: 중심점 좌표 (x_center, y_center)
    """
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    return x_mean, y_mean

if __name__ == '__main__':
    # SIGINT 시그널 핸들러 설정
    signal.signal(signal.SIGINT, sigint_handler)

    vertices = np.array([                  # test.mp4
        [30, 360],       # 좌하
        [240, 200],     # 좌상
        [370, 200],    # 우상
        [600, 360]     # 우하
    ])
    #
    # vertices = np.array([  # video.mp4
    #     [30, 360],  # 좌하
    #     [210, 200],  # 좌상
    #     [420, 200],  # 우상
    #     [600, 360]  # 우하
    # ])

    # vertices = np.array([               #project_video.mp4
    #     [70, 360],       # 좌하
    #     [240, 220],     # 좌상
    #     [400, 220],    # 우상
    #     [600, 360]     # 우하
    # ])

    # vertices = np.array([                   # video_night.mp4
    #     [0, 360],       # 좌하
    #     [160, 180],     # 좌상
    #     [450, 180],    # 우상
    #     [640, 360]     # 우하
    # ])

    data = []
    font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 선택
    font_scale = 1  # 폰트 스케일
    font_thickness = 2  # 폰트 두께
    text_color = (0, 255, 0)  # 텍스트 색상 (BGR)
    text_position = (30, 50)  # 텍스트 위치 (x, y)
    circle_position = (550, 40)
    radius = 20

    first_frame = 1
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('pretrained/best.pth'))    # test_/model_1.pth
    model.eval()
    detection_model = YOLO('best 3.pt')

    cap = cv2.VideoCapture('/home/cvlab/Datasets/test.mp4')    # video setting
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fps = 15.0
    delay = round(1000/ fps)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('./ouput.avi', fourcc, fps,(int(width/2), int(height/2)))
    while True:

        process_start = time.time()
        retval, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))
        if retval == False:
            break
        start_time_twin = time.time()
        result = Run(model, frame)
        # cv2.imshow("run_twin", result)
        end_time_twin = time.time()
        twin_time = round(end_time_twin - start_time_twin, 3)
        print(f"TwinLiteNet 함수 실행 시간: {twin_time} 초")
        """--------------------------------- Lane Detection---------------------------"""
        cpframe = frame.copy()
        # cv2.imshow("run_model", result)

        roi_view = ROI_BEV(result, vertices)
        cv2.imshow("roi_view", roi_view)

        dst, M = roi2bev(roi_view, vertices)

        x = time.time()
        result, line_image, lane_space = preprocess(dst)
        xend = time.time()
        xrun = round(xend - x, 3)
        print(f"x 실행 시간: {xrun} 초")
        # cv2.imshow("resultss", result)
        final_image, real_lane, points = bev2roi(result, M, vertices)


        # cv2.imshow("roi", final_image)
        # real_image = weighted_img(real_lane, frame, α=1., β=1., λ=0.)
        # lane_detection = weighted_img(final_image, frame, α=1., β=1., λ=0.)
        # cv2.fillPoly(cpframe, [points], cyan)
        # cv2.imshow("frmfe", cpframe)
        # cv2.imshow('lane_detection', lane_detection)
        # cv2.imshow('real_image', real_image)
        """--------------------------------- Object Detection---------------------------"""
        start_time_yolo = time.time()
        results = detection_model(cpframe)
        end_time_yolo = time.time()
        yolo_time = round(end_time_yolo - start_time_yolo, 3)
        print(f"Yolov8 함수 실행 시간: {yolo_time} 초")
        annotated_frame = results[0].plot(boxes=False)

        """---------------------------------- Result -----------------------------"""
        lane_points = points
        # x = time.time()
        points = set_safetyzone(points, 1.1)
        point_center_x , point_center_y = find_center_of_points(points)

        if results[0].masks is None:
            continue
        object_polygon_xy = results[0].masks.xy
        object_class = results[0].names
        iou_class = results[0].boxes.cls
        iou_sum = []
        points_poly = Polygon(points)
        points_poly = points_poly.buffer(0)
        detect_obj = []

        distance_list = []
        for i ,obj in enumerate(object_polygon_xy):
            if len(obj) < 4 :
                continue
            obj_center_x, obj_center_y = find_center_of_points(obj)
            object_polygon = Polygon(obj)
            object_polygon = object_polygon.buffer(0)
            intersect = object_polygon.intersection(points_poly).area
            try:
                iou = intersect / points_poly.area
            except ZeroDivisionError:
                iou = 0
            # print(iou)  # iou = 0.5
            detect_class = object_class[int(iou_class[i])]
            if iou > 0:
                iou_sum.append(iou)
                detect_obj.append(detect_class)
            else:
                distance = math.sqrt((obj_center_x - point_center_x)**2 + (obj_center_y - point_center_y) ** 2)
                distance_list.append(distance)
        im_array = results[0].plot(masks=True, boxes=False)  # plot a BGR numpy array of predictions
        warning_text = ""

        if len(iou_sum) != 0:
            warning_text = "Watch out the " + ", ".join(detect_obj)
            iou_mean = np.sum(iou_sum) + 2
            im_array[:, :, 2] = im_array[:, :, 2] * iou_mean
            cv2.putText(im_array, warning_text, text_position, font, font_scale,text_color, font_thickness)
            cv2.circle(im_array, circle_position, radius, red, -1)
        elif any(x <= 100 for x in distance_list):
            cv2.circle(im_array, circle_position, radius, yellow, -1)
        else:
            cv2.circle(im_array, circle_position, radius, green, -1)

        cv2.polylines(im_array, [lane_points], isClosed=True, color=(0,255,0), thickness=2)
        cv2.fillPoly(im_array, [points], (255, 255, 255))
        # restore_img = cv2.resize(im_array, (1280, 720))
        # im_array = im_array
        out.write(im_array)

        process_end = time.time()
        process_time = round(process_end - process_start, 3)
        print(f"End-to-End 실행 시간: {process_time} 초")

        # out.write(annotated_frame)
        cv2.imshow('Lane and Object Detection', im_array)

        time_data = {
            "Process Time": process_time,
            "Yolo Time": yolo_time,
            "Twin Time": twin_time
        }
        data.append(time_data)
        if cv2.waitKey(10) == 27:
            break
    # # 디렉토리 경로
    # directory_path = '/home/cvlab/DoTA_dataset/frames/0qfbmt4G8Rw_001602/images'
    #
    # # 디렉토리 내의 모든 파일 목록 가져오기
    # file_list = os.listdir(directory_path)
    #
    # # jpg 파일만 필터링
    # jpg_files = [file for file in file_list if file.endswith('.jpg')]
    #
    # # 각 jpg 파일을 읽어와서 처리
    # for jpg_file in jpg_files:
    #     file_path = os.path.join(directory_path, jpg_file)
    #     image = cv2.imread(file_path)
    #     cv2.destroyAllWindows()
    with open('execution_times.csv', 'w', newline='') as csv_file:
        fieldnames = ["Process Time", "Yolo Time", "Twin Time"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
