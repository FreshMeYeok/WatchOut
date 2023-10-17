from __future__ import division
import numpy as np
import torch
from model import TwinLite as net
import cv2
from ultralytics import YOLO
import math
import time


default_space = np.array([[100, 360], [540, 360], [360, 180], [280, 180]])
half_w = default_space[0][0] + (default_space[1][0] - default_space[0][0]) // 2


def Run(model, img):
    # img = cv2.resize(img, (640, 360))

    img_rs = img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    # x0=img_out[0]
    x1 = img_out[1]

    # _,da_predict=torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    # DA = da_predict.byte().cpu().data.numpy()[0]*255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    # img_rs[DA>100]=[255,0,0]
    img_rs[LL > 100] = [0, 255, 0]

    return img_rs


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
    out = cv2.VideoWriter(filename, fourcc, frame, (1280, 720))
    return out


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


hei = 25
# alpha = int(args.alpha)
font_size = 1

""" 현재 영상 프레임 표시 """


def show_fps(image, frames, start, color=white):
    now_fps = round(frames / (time.time() - start), 2)
    cv2.putText(image, "FPS : %.2f" % now_fps, (10, hei), font, 0.8, color, font_size)


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


def roi2bev(roi_img, vertices):  # Bird's eye view
    pts1 = np.float32([vertices[1], vertices[0], vertices[2], vertices[3]])
    pts2 = np.float32([[425, 0], [425, 720], [855, 0], [855, 720]])
    M = cv2.getPerspectiveTransform(pts1, pts2) + np.random.rand(3, 3) * 1e-9
    dst = cv2.warpPerspective(roi_img, M, (1280, 720))
    # cv2.imshow("bdffdd", dst)
    return dst, M


"""bev image를 roi image로 변환"""


def bev2roi(bev_img, M, vertices):
    linV = np.linalg.inv(M)

    lindst = cv2.warpPerspective(bev_img, linV, (640, 360))

    # Create a mask of the region of interest using vertices
    mask = np.zeros_like(lindst)
    cv2.fillPoly(mask, [vertices], white)  # Fill the region with white

    # Apply the mask to the lindst image
    lindst = cv2.bitwise_and(lindst, mask)

    vertices, points = find_white_contour_vertices(lindst, vertices)
    cv2.imshow('lindst', lindst)

    return lindst, vertices, points


def find_white_contour_vertices(image, vertices):
    copy_frame = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.fillPoly(image, [default_space], cyan)
    # cv2.imshow("temp", image)
    # Threshold the grayscale image to create a binary mask of white areas
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    # Detect corners using cornerHarris
    dst = cv2.cornerHarris(thresh, 2, 3, 0.04)

    # Find white contours
    white_contours = np.argwhere(dst > 0.01 * dst.max())
    # cv2.imshow("white", white_contours)
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
        elif len(bottom_candidate) > 1 and len(top_candidate) == 1:
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


def auto_canny(image, kernel_size, sigma=0.33):
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
    line_image, lane_space = convert_hough(canny_edges, rho, theta, thresh, min_line_len, max_line_gap, vertices)
    # cv2.imshow('line_image', line_image)

    result = weighted_img(line_image, img, α=1., β=1., λ=0.)
    # cv2.imshow('dfd', result)

    return result, line_image, lane_space


def convert_hough(img, rho, theta, threshold, min_line_len, max_line_gap, vertices):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

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


def draw_lanes(img, lines, vertices):
    global cache
    global first_frame
    global next_frame
    # global prev_lane
    # cv2.imshow("ereq", img)
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
                    r_lane.append((x1, y1))
                    r_lane.append((x2, y2))
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
    right_line, right_inliers = ransac_line_fit(np.array(r_lane_start), np.array(r_lane_end), iterations=100,
                                                threshold=3)

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
            next_frame = (1 - α) * prev_frame + α * current_frame
        lane_space = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]],
                               [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], dtype=int)
    else:
        lane_space = default_space

    cv2.fillPoly(img, [lane_space], (204, 255, 204))

    cache = next_frame

    return lane_space


def set_safetyzone(points, divider):
    max_bottom, min_bottom, min_top, max_top = points
    x1, y1 = max_top
    x2, y2 = max_bottom
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
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



def find_center_of_points(points):
    """
    주어진 좌표 배열의 중심점을 찾는 함수
    :param points: (N, 2) 형태의 NumPy 배열. 각 행은 (x, y) 좌표를 나타냄.
    :return: 중심점 좌표 (x_center, y_center)
    """
    x_mean = int(np.mean(points[:, 0]))
    y_mean = int(np.mean(points[:, 1]))
    return x_mean, y_mean


if __name__ == '__main__':
    first_frame = 1
    vertices = np.array([                  # test.mp4
        [30, 360],       # 좌하
        [250, 90],     # 좌상
        [390, 90],    # 우상
        [630, 360]     # 우하
    ])

    image_path = "/home/cvlab/Downloads/yellow_can.png"  # night.png
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('pretrained/best.pth'))  # test_/model_1.pth
    model.eval()
    detection_model = YOLO('model/best_yolo.pt')

    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 360))
    cpimg = img.copy()
    results = detection_model(cpimg)
    annotated_frame = results[0].plot(boxes=False)
    img = Run(model, img)
    roi_view = ROI_BEV(img, vertices)
    dst, M = roi2bev(roi_view, vertices)
    result, line_image, lane_space = preprocess(dst)
    final_image, real_lane, points = bev2roi(result, M, vertices)
    # cv2.imshow("roi", final_image)
    real_image = weighted_img(real_lane, img, α=1., β=1., λ=0.)
    lane_detection = weighted_img(final_image, img, α=1., β=1., λ=0.)
    cv2.imwrite('/home/cvlab/Datasets/opencv_and_Twin_yellow.png', lane_detection)
    cv2.fillPoly(img, [points], white)
    cv2.imshow('result', img)
    cv2.imwrite('/home/cvlab/Datasets/opencv_and_Twin_yello_final.png', img)
    cv2.imwrite('/home/cvlab/Datasets/opencv_and_Twin_yello_yolo.png', annotated_frame)
