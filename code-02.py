from collections import deque
import cv2
import numpy as np
import math
import tensorflow.keras as keras
import tensorflow as tf
import serial
import time

# Serial = serial.Serial("COM4",9600,timeout=2)
# Serial.write(bytes("b","utf-8"))

# Quy định kích thước của ô vuông lấy mẫu
capture_box_dim = 10
# Tọa độ đặt vùng lấy mẫu
capture_pos_x = 500
capture_pos_y = 150
# Quy định khoảng cách theo chiều ngang và dọc giữa 2 ô
capture_box_sep_x = 8
capture_box_sep_y = 18

# Tọa độ của vùng đặt bàn tay cho việc xử lý
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width


# Lấy mẫu bàn tay
def hand_capture(frame_in, box_x, box_y):
    # Chuyển hình ảnh ban đầu thành mã màu HSV
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    # Biến ROI lưu tập hợp các ảnh từ các ô vuông lấy mẫu
    ROI = np.zeros([capture_box_dim*len(box_x),
                    capture_box_dim, 3], dtype=hsv.dtype)
    for i in range(len(box_x)):
        ROI[i*capture_box_dim:i*capture_box_dim+capture_box_dim, 0:capture_box_dim] = hsv[box_y[i]                                                                                          :box_y[i]+capture_box_dim, box_x[i]:box_x[i]+capture_box_dim]
    # Thu được ROI là một cái ảnh tổng hợp từ các ô vuông lấy mẫu
    # Tính histogram của ảnh ROI - cho biết được là màu da sẽ có biểu đồ histogram như thế nào
    hand_hist = cv2.calcHist([ROI], [0, 1], None, [20, 20], [0, 180, 0, 256])
    # Chuẩn hóa lại giá trị
    cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist


def hand_threshold(frame_in, hand_hist):
    #frame_in = cv2.medianBlur(frame_in, 3)
    # Chuyển màu sang mã màu hsv
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    # Lấy những tương tự với màu histogram lấy mẫu - thu được ảnh đen trắng
    back_projection = cv2.calcBackProject(
        [hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)
    # Lấy ngưỡng chuyển về ảnh nhị phân
    ret, thresh = cv2.threshold(
        back_projection, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = thresh
    # Lấp đầy các đốm trắng
    mask = cv2.dilate(mask, kernel,  iterations=2)
    return mask


def remove_bg(frame_roi):
    # Xử lí tách nền - mask là một ảnh đen trắng
    mask = bg_model.apply(frame_roi, learningRate=0)
    kernel = np.ones((4, 4), np.uint8)
    # Loại bỏ các đốm nhiễu nhỏ đen, trắng
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Ảnh nhị phân
    ret, mask = cv2.threshold(
        mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Sau đó lấy ảnh gốc and với ảnh nhị phân (mask) thì thu được ảnh đã loại bỏ nền
    frame_roi = cv2.bitwise_and(frame_roi, cv2.merge((mask, mask, mask)))
    return mask, frame_roi


def calculateAngle(far, start, end):
    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
    return angle


def distanceBetweenTwoPoints(start, end):
    return math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)


def resizeKeepRation(image, size, interpolation):
    h = image.shape[0]
    w = image.shape[1]
    diff = max(h, w)
    padding = int(3*diff / 8)
    mask = np.zeros((diff+2*padding, diff+2*padding), dtype=image.dtype)
    x_pos = int((mask.shape[1] - w) / 2.0)
    y_pos = int((mask.shape[0] - h) / 2.0)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = image[:, :]
    return cv2.resize(mask, (size, size), interpolation)


# ----------------- MAIN -------------------- #
if __name__ == "__main__":
    # Load model nhận diện chữ số có sẵn
    new_model = tf.keras.models.load_model('epic_num_reader.model')
    #
    pts = deque(maxlen=500)
    # Lưu tọa độ x, y tương ứng của các ô lấy mẫu
    box_pos_x = []
    box_pos_y = []

    # Biến n để cho biết sẽ có bao nhiêu ô ngang, bao nhiêu ô dọc
    n = 6
    # Khởi tạo vùng lấy mẫu màu da
    for i in range(n):
        box_pos_x.append(capture_pos_x + capture_box_dim *
                         i + capture_box_sep_x*i)
        temp = [capture_pos_y + capture_box_dim*i + capture_box_sep_y * i]*n
        box_pos_y += temp
    box_pos_x = box_pos_x*n

    # Tạo biến để đọc dữ liệu ảnh
    camera = cv2.VideoCapture(1)
    capture_done = False
    bg_captured = False
    hand_histogram = None
    listText = []
    black_board = np.zeros((384, 320))
    debug_frame = None
    debug = False
    cv2.namedWindow("Anh")
    cv2.moveWindow("Anh", 0, 650)
    cv2.namedWindow("ve contour")
    cv2.moveWindow("ve contour", 500, 650)
    cv2.namedWindow('frame')
    cv2.moveWindow('frame', 0, 0)
    cv2.namedWindow('calc back histogram')
    cv2.moveWindow('calc back histogram', 700, 0)
    guess = False
    guess_num = -1
    guess_esti = 0
    while(1):
        # Đọc ảnh lưu vào frame
        ret, frame = camera.read()
        # Lật ảnh
        frame = cv2.flip(frame, 1)
        # Bộ lọc - lọc nhiễu
        frame = cv2.bilateralFilter(frame, 5, 0, 0)
        frame_original = np.copy(frame)
        # Vẽ hình vùng xử lý
        cv2.rectangle(frame, (int(cap_region_x_begin*frame.shape[1]), 0), (frame.shape[1], int(
            cap_region_y_end*frame.shape[0])), (255, 0, 0), 1)

        # Lấy ảnh của vùng cần xử lý để tính toán
        frame_roi = frame_original[0:int(cap_region_y_end*frame.shape[0]), int(cap_region_x_begin *
                                                                               frame.shape[1]):frame.shape[1]].copy()
        frame_roi = cv2.medianBlur(frame_roi, 5)

        # Lấy ảnh của background
        if(bg_captured):
            # Loại background khỏi ảnh
            bkg_mask, bkg_frame = remove_bg(frame_roi)

        if (not (capture_done and bg_captured)):
            if (not bg_captured):
                # note chi dan
                cv2.putText(frame, "Remove hand from the frame and press 'b' to capture background", (int(
                    0.05*frame.shape[1]), int(0.97*frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 8)

            else:
                # note chi dan
                cv2.putText(frame, "Place hand inside boxes and press 'c' to capture hand histogram", (int(
                    0.08*frame.shape[1]), int(0.97*frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 8)

            for i in range(len(box_pos_x)):
                cv2.rectangle(frame, (box_pos_x[i], box_pos_y[i]), (
                    box_pos_x[i]+capture_box_dim, box_pos_y[i]+capture_box_dim), (255, 0, 0), 1)
            hand_hist_temp = hand_capture(frame_original, box_pos_x, box_pos_y)
            back_projection = hand_threshold(frame_roi, hand_hist_temp)
            # back_projection = cv2.medianBlur(back_projection, median_ksize)
            cv2.imshow("calc back histogram", back_projection)
        else:
            cv2.destroyWindow("calc back histogram")
            cv2.destroyWindow("ve contour")
            finger_count = 0
            # Thu được ảnh nhị phân mask, vùng bàn tay
            fgmask = hand_threshold(bkg_frame, hand_histogram)

            # -----
            # Lấy contours của ảnh nhị phân
            contours, hierarchy = cv2.findContours(
                fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # Lấy countours lớn nhất
            if (len(contours) != 0):
                hand_contour = max(contours, key=cv2.contourArea)
            # -----

            if(len(contours) != 0):
                temp = frame_roi
                # print("So contours tim duoc la: %d" % len(contours))
                M = cv2.moments(hand_contour)
                points1 = []
                # Lấy trọng tâm của bàn tay
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(temp, (cx, cy), 3, [0, 255, 255], -1)
                # Điều chỉnh cách lấy contours
                hand_contour = cv2.approxPolyDP(
                    hand_contour, 0.02*cv2.arcLength(hand_contour, True), True)
                # Lấy bao lồi của contours
                hull = cv2.convexHull(hand_contour, returnPoints=True)
                for point in hull:
                    if (cy > point[0][1]):
                        points1.append(tuple(point[0]))

                # ---- ve hand contour
                cv2.drawContours(temp, [hand_contour], 0, (255, 255, 0), 1)
                hull = cv2.convexHull(hand_contour, returnPoints=False)

                if (len(hull) > 3):
                    defects = cv2.convexityDefects(hand_contour, hull)
                    if type(defects) != type(None):
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i][0]
                            start = tuple(hand_contour[s, 0])
                            end = tuple(hand_contour[e, 0])
                            far = tuple(hand_contour[f, 0])
                            angle = calculateAngle(far, start, end)
                            if (d > 10000) and (angle <= math.pi/2) and (far[1] - cy) <= 0:
                                cv2.line(temp, start, end, [0, 255, 0], 5)
                                cv2.circle(temp, far, 5, [0, 0, 255], -1)
                                finger_count += 1
                cv2.putText(frame, str(finger_count+1), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 2)
                cv2.putText(frame, "So du doan: {} - Ti le: {}".format(guess_num, guess_esti), (20, frame.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 2)
                # -- Xu ly finger count
                if finger_count != 1:
                    extTop = tuple(
                        hand_contour[hand_contour[:, :, 1].argmin()][0])
                    cv2.circle(temp, extTop, 3, [100, 0, 255], -1)
                    pts.appendleft(extTop)
                    for i in range(1, len(pts)):
                        if len(pts) >= 12 and i == 1:
                            dX = pts[0][0] - pts[10][0]
                            dY = pts[0][1] - pts[10][1]
                        # cv2.line(temp, pts[i-1], pts[i], (0, 0, 255), 8)
                        t1 = list(pts[i-1])
                        t2 = list(pts[i])
                        t1[0] += (int(cap_region_x_begin*frame.shape[1]))
                        t2[0] += (int(cap_region_x_begin*frame.shape[1]))
                        cv2.line(frame, tuple(t1), tuple(t2), (0, 0, 255), 8)
                        cv2.line(black_board, pts[i-1],
                                 pts[i], (255, 255, 255), 8)
                    pass
                # Neu hai ngon
                elif finger_count == 1:
                    pts = deque(maxlen=500)
                    # ------
                    x1, x2, y1, y2 = -1, -1, -1, -1
                    for y in range(black_board.shape[0]):
                        for x in range(black_board.shape[1]):
                            if black_board[y, x] > 0:
                                if x1 == -1:
                                    x1 = x
                                if y1 == -1:
                                    y1 = y
                                if x2 == -1:
                                    x2 = x
                                if y2 == -1:
                                    y2 = y
                                x1, y1 = min(x1, x), min(y1, y)
                                x2, y2 = max(x2, x), max(y2, y)
                    if (x1 == -1 or x2 == -1 or y1 == -1 or y2 == -1):
                        black_board = np.zeros((384, 320))
                    else:
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        # ----
                        if (x > 0 and y > 0 and w > 0 and h > 0):
                            debug_frame = np.copy(black_board[y:y+h, x:x+w])
                            debug = True
                            anh_digit = resizeKeepRation(
                                (debug_frame), 28, interpolation=cv2.INTER_AREA)
                            anh_digit = tf.keras.utils.normalize(
                                anh_digit, axis=1)
                            anh_digit = np.reshape(anh_digit, (1, 28, 28))
                            # Dự đoán
                            predictions = new_model.predict(anh_digit)
                            guess_num = np.argmax(predictions[0])
                            guess_esti = predictions[0][guess_num]*100
                            print("-------------------")
                            print("So du doan", guess_num)
                            print("Ti le du doan", guess_esti)

                        # ------
                    black_board = np.zeros((384, 320))

                #black = np.ones((h,w))
                #black = black_board[y:h,x:w]
                if debug:
                    canvas = np.zeros((28, 28))
                    canvas[:] = anh_digit[0]
                    cv2.imshow("Anh", canvas)
                # cv2.imshow("ve contour", temp)
            else:
                frame = frame_original
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
        elif k == ord('c'):
            if (bg_captured):
                capture_done = True
                hand_histogram = hand_capture(
                    frame_original, box_pos_x, box_pos_y)
        elif k == ord('b'):
            # Đối tượng tách nền - dùng thuật toán MOG2 để tách nền
            bg_model = cv2.createBackgroundSubtractorMOG2()
            bg_captured = True
        elif k == ord('r'):
            capture_done = 0
            bg_captured = 0

    cv2.destroyAllWindows()
