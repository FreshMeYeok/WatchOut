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
    
    return img_rs


vertices = np.array([                  # video.mp4
    [100, 720],       # 좌하
    [420, 400],     # 좌상
    [840, 400],    # 우상
    [1150, 720]     # 우하
])

# vertices = np.array([               #project_video.mp4
#     [100, 720],       # 좌하
#     [450, 450],     # 좌상
#     [800, 450],    # 우상
#     [1150, 720]     # 우하
# ])
# vertices = np.array([                   # video_night.mp4
#     [0, 720],       # 좌하
#     [240, 360],     # 좌상
#     [720, 360],    # 우상
#     [1280, 720]     # 우하
# ])
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
def bev2roi(bev_img, M, lane_infor, vertices):
    linV = np.linalg.inv(M)
    lindst = cv2.warpPerspective(bev_img, linV, (1280, 720))

    # Create a mask of the region of interest using vertices
    mask = np.zeros_like(lindst)
    cv2.fillPoly(mask, [vertices], white)  # Fill the region with white

    # Apply the mask to the lindst image
    lindst = cv2.bitwise_and(lindst, mask)
    vertices, points = find_white_contour_vertices(lindst, vertices)
    # cv2.imshow('lindst', lindst)
    return lindst, vertices, points


def find_white_contour_vertices(image, vertices):
    coordinate_list = []
    copy_frame= image.copy()
    h, w, _ = image.shape
    half_w = w / 2
    default_space = np.array([[320,720], [960,720], [700, 450], [500,450]])

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # Threshold the grayscale image to create a binary mask of white areas
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', thresh)
    dst = cv2.cornerHarris(thresh, 5, 3, 0.04)
    dst = cv2.dilate(dst, None)

    image[dst > 0.01*dst.max()] = [0, 0, 255]
    four_corner = np.zeros(thresh.shape)
    four_corner[dst > 0.01*dst.max()] = 1
    y_list = []
    for i in range(len(four_corner)):
        for j in range(len(four_corner[0])):
            if four_corner[i][j] == 1 and i >= vertices[1][1]:
                coordinate_list.append([i, j])
                y_list.append(i)
    bottom_candidate = []
    top_candidate = []
    if len(y_list) == 0:
        points = default_space
    else:
        max_y_value = max(y_list)
        min_y_value = min(y_list)

        for coordinate in coordinate_list:
            i, _ = coordinate
            if i == max_y_value:
                bottom_candidate.append(coordinate[1])
            if i == min_y_value:
                top_candidate.append(coordinate[1])

        if len(bottom_candidate) == 0 or len(top_candidate) == 0:
            points = default_space
        elif len(bottom_candidate) == 1 and len(top_candidate) > 1:
            bottom_x = bottom_candidate[0]
            if bottom_x >= half_w:
                differ = bottom_x - half_w
                bottom_x_another = half_w - differ
                max_bottom = [bottom_x, max_y_value]
                min_bottom = [bottom_x_another, max_y_value]
            else:
                differ = half_w - bottom_x
                bottom_x_another = half_w + differ
                max_bottom = [bottom_x_another, max_y_value]
                min_bottom = [bottom_x, max_y_value]
            max_top = [max(top_candidate), min_y_value]
            min_top = [min(top_candidate), min_y_value]
            points = np.array([max_bottom, min_bottom, min_top, max_top])
        elif len(bottom_candidate) > 1 and len(top_candidate) == 1:
            top_x = top_candidate[0]
            if top_x >= half_w:
                differ = top_x - half_w
                top_x_another = half_w - differ
                max_top = [top_x, min_y_value]
                min_top = [top_x_another, min_y_value]
            else:
                differ = half_w - top_x
                top_x_another = half_w + differ
                max_top = [top_x_another, min_y_value]
                min_top = [top_x, min_y_value]
            max_bottom = [max(bottom_candidate), max_y_value]
            min_bottom = [min(bottom_candidate), max_y_value]
            points = np.array([max_bottom, min_bottom, min_top, max_top])
        elif len(bottom_candidate) == 1 and len(top_candidate) == 1:
            bottom_x = bottom_candidate[0]
            top_x = top_candidate[0]
            if bottom_x >= half_w:
                differ = bottom_x - half_w
                bottom_x_another = half_w - differ
                max_bottom = [bottom_x, max_y_value]
                min_bottom = [bottom_x_another, max_y_value]
            else:
                differ = half_w - bottom_x
                bottom_x_another = half_w + differ
                max_bottom = [bottom_x_another, max_y_value]
                min_bottom = [bottom_x, max_y_value]
            if top_x >= half_w:
                differ = top_x - half_w
                top_x_another = half_w - differ
                max_top = [top_x, min_y_value]
                min_top = [top_x_another, min_y_value]
            else:
                differ = half_w - top_x
                top_x_another = half_w + differ
                max_top = [top_x_another, min_y_value]
                min_top = [top_x, min_y_value]
            points = np.array([max_bottom, min_bottom, min_top, max_top])
        else:
            max_bottom = [max(bottom_candidate), max_y_value]
            min_bottom = [min(bottom_candidate), max_y_value]
            max_top = [max(top_candidate), min_y_value]
            min_top = [min(top_candidate), min_y_value]
            points = np.array([max_bottom, min_bottom, min_top, max_top])


    # cv2.fillPoly(copy_frame, [points], cyan)
    # cv2.imshow("mdkafdf", copy_frame)
    # cv2.imshow('four_corner', four_corner)
    # cv2.imshow('harris', image)
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
    # cv2.imshow('gray_image', gray_image)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # 더 넓은 폭의 노란색 범위를 얻기위해 HSV를 이용한다.
    # cv2.imshow('img_hsv', img_hsv)
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    # cv2.imshow('mask_yellow', mask_yellow)
    mask_white = cv2.inRange(gray_image, 140, 255)
    # cv2.imshow('mask_white', mask_white)

    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)  # 흰색과 노란색의 영역을 합친다.
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)  # Grayscale로 변환한 원본 이미지에서 흰색과 노란색만 추출

    # cv2.imshow('white and yello image', mask_yw_image)
    canny_edges = auto_canny(mask_yw_image, kernel_size)

    # cv2.imshow('canny_image_bev', canny_edges)
    line_image, lane_space = convert_hough(canny_edges, rho, theta, thresh, min_line_len, max_line_gap, vertices)
    # cv2.imshow('line_image', line_image)
    result = weighted_img(line_image, img, α=1., β=1., λ=0.)
    # cv2.polylines(result, vertices, True, (0, 255, 255)) # ROI mask

    return result, line_image, lane_space

def convert_hough(img, rho, theta, threshold, min_line_len, max_line_gap, vertices):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lane_space = draw_lanes(line_img, lines, vertices)
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
    prev_lane = lane_space
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
    new_right_x = int((new_y - b) / a)
    new_left = [new_left_x, new_y]
    new_right = [new_right_x, new_y]
    new_points = np.array([max_bottom, min_bottom, new_left, new_right])
    return new_points

if __name__ == '__main__':
    font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 선택
    font_scale = 1  # 폰트 스케일
    font_thickness = 2  # 폰트 두께
    text_color = (0, 255, 0)  # 텍스트 색상 (BGR)
    text_position = (50, 100)  # 텍스트 위치 (x, y)

    first_frame = 1
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('pretrained/best.pth'))    # test_/model_1.pth
    model.eval()
    detection_model = YOLO('best 3.pt')

    cap = cv2.VideoCapture('/home/cvlab/Datasets/video_night.mp4')    # video setting
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fps = 15.0
    delay = round(1000/ fps)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('./ouput.avi', fourcc, fps,(int(width), int(height)))
    while True:
        retval, frame = cap.read()

        if retval == False:
            break
        result = Run(model, frame)
        """--------------------------------- Lane Detection---------------------------"""
        cpframe = frame.copy()
        # cv2.imshow("run_model", result)
        roi_view = ROI_BEV(result, vertices)
        dst, M = roi2bev(roi_view, vertices)
        result, line_image, lane_space = preprocess(dst)
        final_image, real_lane, points = bev2roi(result, M, lane_space, vertices)
        # cv2.imshow("roi", final_image)
        real_image = weighted_img(real_lane, frame, α=1., β=1., λ=0.)
        lane_detection = weighted_img(final_image, frame, α=1., β=1., λ=0.)
        # cv2.fillPoly(cpframe, [points], cyan)
        # cv2.imshow("frmfe", cpframe)
        # cv2.imshow('lane_detection', lane_detection)
        # cv2.imshow('real_image', real_image)
        """--------------------------------- Object Detection---------------------------"""
        results = detection_model(cpframe)
        annotated_frame = results[0].plot(boxes=False)
        cv2.imshow('anoo!!', annotated_frame)
        """---------------------------------- Result -----------------------------"""
        lane_points = points
        points = set_safetyzone(points, 1)
        if results[0].masks is None:
            continue
        object_polygon_xy = results[0].masks.xy
        object_class = results[0].names
        iou_class = results[0].boxes.cls
        iou_sum = []
        points_poly = Polygon(points)
        points_poly = points_poly.buffer(0)
        detect_obj = []
        for i ,obj in enumerate(object_polygon_xy):
            if len(obj) < 4 :
                continue
            object_polygon = Polygon(obj)
            object_polygon = object_polygon.buffer(0)
            intersect = object_polygon.intersection(points_poly).area
            iou = intersect / points_poly.area
            # print(iou)  # iou = 0.5
            detect_class = object_class[int(iou_class[i])]
            if iou > 0.01:
                iou_sum.append(iou)
                detect_obj.append(detect_class)

        im_array = results[0].plot(masks=True, boxes=False)  # plot a BGR numpy array of predictions
        warning_text = ""

        if len(iou_sum) != 0:
            warning_text = warning_text.join(detect_obj) + " is close!!"
            iou_mean = np.sum(iou_sum) + 2
            im_array[:, :, 2] = im_array[:, :, 2] * iou_mean
            cv2.putText(im_array, warning_text, text_position, font, font_scale,text_color, font_thickness)
        cv2.polylines(im_array, [lane_points], isClosed=True, color=(0,255,0), thickness=2)
        cv2.fillPoly(im_array, [points], (255, 255, 255))
        out.write(im_array)

        # out.write(annotated_frame)
        cv2.imshow('Lane and Object Detection', im_array)
        if cv2.waitKey(10) == 27:
            break
    # 디렉토리 경로
    directory_path = '/home/cvlab/DoTA_dataset/frames/0qfbmt4G8Rw_001602/images'

    # 디렉토리 내의 모든 파일 목록 가져오기
    file_list = os.listdir(directory_path)

    # jpg 파일만 필터링
    jpg_files = [file for file in file_list if file.endswith('.jpg')]

    # 각 jpg 파일을 읽어와서 처리
    for jpg_file in jpg_files:
        file_path = os.path.join(directory_path, jpg_file)
        image = cv2.imread(file_path)

        # 여기서 image 변수를 사용하여 이미지를 처리하거나 원하는 작업을 수행합니다.

        # 필요한 작업을 수행한 후 이미지 객체를 해제합니다.
        cv2.destroyAllWindows()


    cap.release()
    out.release()
    cv2.destroyAllWindows()
