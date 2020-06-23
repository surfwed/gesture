import cv2
import numpy as np
import math

# Tọa độ các điểm sẽ lấy mẫu
sample_X = []
sample_Y = []
# Độ dài mỗi cạnh hình vuông
sample_edge = 4


cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
# Lay histogram ban tay


# Hàm này để lấy mẫu màu sắc của bàn tay
# Người dùng sẽ đặt bàn tay sao cho vùng da đè lên các ô vuông màu xanh lá
# Các ô vuông màu xanh lá là các vị trí lấy mẫu
# Tham số truyền vào:
#   frame_in: là hình ảnh chứa bàn tay
#   sample_x: chứa tọa độ ô vuông theo trục x
#   sample_y: chứa tọa độ ô vuông theo trục y
def hand_capture(frame_in):
    # Chuyển hình ảnh ban đầu thành mã màu HSV
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)

    ROI = np.zeros([len(sample_Y)*sample_edge, len(sample_X)
                    * sample_edge, 3], dtype=hsv.dtype)
    for i in range(len(sample_Y)):
        for j in range(len(sample_X)):
            ROI[i*sample_edge:sample_edge*(i+1), j*sample_edge:sample_edge*(
                j+1)] = hsv[sample_Y[i]:sample_Y[i]+sample_edge, sample_X[j]:sample_X[j]+sample_edge]
    hand_hist = cv2.calcHist([ROI], [0, 1], None, [12, 12], [0, 180, 0, 256])
    cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist

# Phan vung ngon tay
# Biet duoc ban tay, bao dao ban tay


def mark_fingers(frame_in, hull, pt, radius):
    global first_iteration
    global finger_ct_history
    finger = [(hull[0][0][0], hull[0][0][1])]
    j = 0

    cx = pt[0]
    cy = pt[1]

    for i in range(len(hull)):
        dist = np.sqrt((hull[-i][0][0] - hull[-i+1][0][0])
                       ** 2 + (hull[-i][0][1] - hull[-i+1][0][1])**2)
        if (dist > 18):
            if(j == 0):
                finger = [(hull[-i][0][0], hull[-i][0][1])]
            else:
                finger.append((hull[-i][0][0], hull[-i][0][1]))
            j = j+1

    temp_len = len(finger)
    i = 0
    while(i < temp_len):
        dist = np.sqrt((finger[i][0] - cx)**2 + (finger[i][1] - cy)**2)
        if(dist < finger_thresh_l*radius or dist > finger_thresh_u*radius or finger[i][1] > cy+radius):
            finger.remove((finger[i][0], finger[i][1]))
            temp_len = temp_len-1
        else:
            i = i+1

    temp_len = len(finger)
    if(temp_len > 5):
        for i in range(1, temp_len+1-5):
            finger.remove((finger[temp_len-i][0], finger[temp_len-i][1]))

    palm = [(cx, cy), radius]

    if(first_iteration):
        finger_ct_history[0] = finger_ct_history[1] = len(finger)
        first_iteration = False
    else:
        finger_ct_history[0] = 0.34 * \
            (finger_ct_history[0]+finger_ct_history[1]+len(finger))

    if((finger_ct_history[0]-int(finger_ct_history[0])) > 0.8):
        finger_count = int(finger_ct_history[0])+1
    else:
        finger_count = int(finger_ct_history[0])

    finger_ct_history[1] = len(finger)

    count_text = "FINGERS:"+str(finger_count)
    cv2.putText(frame_in, count_text, (int(0.62*frame_in.shape[1]), int(
        0.88*frame_in.shape[0])), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)

    for k in range(len(finger)):
        cv2.circle(frame_in, finger[k], 10, 255, 2)
        cv2.line(frame_in, finger[k], (cx, cy), 255, 2)
    return frame_in, finger, palm

# Tim diem chinh giua cua ban tay


def mark_hand_center(frame_in, cont):
    max_d = 0
    pt = (0, 0)
    # Tra ve toa do, chieu dai rong cua hinh chu nhat bao contour
    x, y, w, h = cv2.boundingRect(cont)
    # around 0.25 to 0.6 region of height (Faster calculation with ok results)
    for ind_y in range(int(y+0.3*h), int(y+0.8*h)):
        # around 0.3 to 0.6 region of width (Faster calculation with ok results)
        for ind_x in range(int(x+0.3*w), int(x+0.6*w)):
            # Tim khoang cach den xa contour nhat
            dist = cv2.pointPolygonTest(cont, (ind_x, ind_y), True)
            if(dist > max_d):
                max_d = dist
                pt = (ind_x, ind_y)
    if(max_d > radius_thresh*frame_in.shape[1]):
        thresh_score = True
        cv2.circle(frame_in, pt, int(max_d), (255, 0, 0), 2)
    else:
        thresh_score = False
    return frame_in, pt, max_d, thresh_score


# filter va threshold


def hand_threshold(frame_in, hand_hist):
    # Lam mo anh
    frame_in = cv2.medianBlur(frame_in, 3)
    # cv2.imshow('frame_in', frame_in)

    # Chuyen sang mau hsv
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    # Tinh back project
    back_projection = cv2.calcBackProject(
        [hsv], [0, 1], hand_hist, [00, 180, 0, 256], 1)
    # back_projection = cv2.medianBlur(back_projection, median_ksize)
    cv2.imshow("cacl back histogram", back_projection)
    # ret, thresh = cv2.threshold(
    #     back_projection, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(
        back_projection, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold calc back histogram", thresh)
    kernel = np.ones((3, 3), np.uint8)
    # fgmask = cv2.morphologyEx(
    #     back_projection, cv2.MORPH_CLOSE, kernel, iterations=1)
    fgmask = thresh
    # fgmask = cv2.dilate(back_projection, kernel,  iterations=2)
    fgmask = cv2.dilate(fgmask, kernel,  iterations=2)
    # fgmask = cv2.erode(fgmask, kernel,  iterations=1)
    # fgmask = cv2.dilate(fgmask, kernel,  iterations=1)
    # edges = cv2.Canny(fgmask, 100, 100)
    # cv2.imshow("edges", edges)
    # cv2.imshow('fgmask dilate', fgmask)
    return fgmask
    cv2.imshow("fgmask backproject", fgmask)
    contours, hierarchy = cv2.findContours(
        fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    print("So contours tim duoc method 2 %d" % len(contours))
    temp = cv2.bitwise_and(frame_in, cv2.merge((fgmask, fgmask, fgmask)))
    cv2.drawContours(temp, [c], -1, (0, 255, 0), 3)
    cv2.imshow("temp", temp)
    # Tao kernel loc nhieu
    disc = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_elem_size, morph_elem_size))
    # Loc nhieu
    cv2.filter2D(back_projection, -1, disc, back_projection)
    # Loc nhieu GaussianBlur
    back_projection = cv2.GaussianBlur(
        back_projection, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    # Tiep tuc loc nhieu
    back_projection = cv2.medianBlur(back_projection, median_ksize)
    # Lay nguong
    ret, thresh = cv2.threshold(back_projection, hsv_thresh_lower, 255, 0)
    # cv2.imshow('thresh func', thresh)
    return thresh

# Ham remove background khoi mot anh


def remove_bg(frame_roi):
    # bg_model la mot doi tuong cua mog2
    # Ap dung remove background cho frame va tra ve mask
    fgmask = bg_model.apply(frame_roi, learningRate=0)
    # Tao kernel de loai bo mask nhieu
    kernel = np.ones((4, 4), np.uint8)
    # Tien hanh loai bo mask nhieu
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgmask = cv2.morphologyEx(
        fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Dem and mask voi frame -> co anh da remove background
    ret, fgmask = cv2.threshold(
        fgmask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Mask bkg subtraction', fgmask)

    contours, hierarchy = cv2.findContours(
        fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        temp = cv2.bitwise_and(frame_roi, cv2.merge((fgmask, fgmask, fgmask)))
        cv2.drawContours(temp, [c], -1, (0, 255, 0), 3)
        cv2.imshow("And mask bkg sub vs roi", temp)
    frame_roi = cv2.bitwise_and(frame_roi, cv2.merge((fgmask, fgmask, fgmask)))
    # cv2.imshow('frame_', frame)
    return frame_roi

# ----------------- MAIN -------------------- #


# camera
# Tao doi tuong de lay hinh anh tu camera
camera = cv2.VideoCapture(0)
# Lay mau histogram ban tay
capture_done = False
# Dieu kien da cap background chua
bg_captured = False
# ?
# GestureDictionary = DefineGestures()
# ?
# frame_gesture = Gesture("frame_gesture")

while(1):
    # Capture frame from camera
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame_original = np.copy(frame)
    # Phep loc nhieu tien xu ly
    frame = cv2.bilateralFilter(frame, 5, 0, 0)
    # Ve hinh chu nhat
    cv2.rectangle(frame, (int(cap_region_x_begin*frame.shape[1]), 0), (frame.shape[1], int(
        cap_region_y_end*frame.shape[0])), (255, 0, 0), 1)

    frame_roi = frame[0:int(cap_region_y_end*frame.shape[0]), int(cap_region_x_begin *
                                                                  frame.shape[1]):frame.shape[1]].copy()
    # cv2.imshow('frame_roi', frame_roi)
    # Neu da cap background roi thi:
    if(bg_captured):
        # Ham remove background
        fg_frame = remove_bg(frame_roi)
        # cv2.imshow("fg_frame", fg_frame)

    # Neu van chua lay mau background voi lay mau histogram ban tay thi
    if (not (capture_done and bg_captured)):
        # Neu chua lay mau background thi ghi note chu y lay mau
        if (not bg_captured):
            # note chi dan
            cv2.putText(frame, "Remove hand from the frame and press 'b' to capture background", (int(
                0.05*frame.shape[1]), int(0.97*frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 8)
        # Neu da lay mau background thi ghi note lay mau histogram ban tay
        else:
            # note chi dan
            cv2.putText(frame, "Place hand inside boxes and press 'c' to capture hand histogram", (int(
                0.08*frame.shape[1]), int(0.97*frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 8)
        # ?
        first_iteration = True
        # ?
        finger_ct_history = [0, 0]
        # Ve cac o vuong hinh chu nhat de laqqqy mau ban tay
        # Toa do cac o
        box_pos_x = np.array([capture_pos_x, capture_pos_x+capture_box_dim+capture_box_sep_x, capture_pos_x+2*capture_box_dim+2*capture_box_sep_x, capture_pos_x, capture_pos_x+capture_box_dim+capture_box_sep_x,
                              capture_pos_x+2*capture_box_dim+2*capture_box_sep_x, capture_pos_x, capture_pos_x+capture_box_dim+capture_box_sep_x, capture_pos_x+2*capture_box_dim+2*capture_box_sep_x], dtype=int)
        box_pos_y = np.array([capture_pos_y, capture_pos_y, capture_pos_y, capture_pos_y+capture_box_dim+capture_box_sep_y, capture_pos_y+capture_box_dim+capture_box_sep_y, capture_pos_y+capture_box_dim +
                              capture_box_sep_y, capture_pos_y+2*capture_box_dim+2*capture_box_sep_y, capture_pos_y+2*capture_box_dim+2*capture_box_sep_y, capture_pos_y+2*capture_box_dim+2*capture_box_sep_y], dtype=int)
        # Duyet for ve cac o
        for i in range(capture_box_count):
            cv2.rectangle(frame, (box_pos_x[i], box_pos_y[i]), (
                box_pos_x[i]+capture_box_dim, box_pos_y[i]+capture_box_dim), (255, 0, 0), 1)
    else:
        # Khi da lay lau duoc background va histogram hand
        # Phan loai ra vung chua hand
        fgmask = hand_threshold(fg_frame, hand_histogram)

        # -----
        cv2.imshow("bkg mask vs calBack histogram", fgmask)
        contours, hierarchy = cv2.findContours(
            fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if (len(contours) != 0):
            hand_contour = max(contours, key=cv2.contourArea)
        # -----
        # cv2.imshow('frame threshold', frame)

        if(len(contours) != 0):
            print("So contours tim duoc la: %d" % len(contours))
            # ---- ve hand contour
            temp = frame_roi
            # hull = cv2.convexHull(hand_contour, returnPoints=True)
            # cv2.drawContours(temp, [hand_contour], -1, (255, 0, 0), 3)
            hull = cv2.convexHull(hand_contour, returnPoints=False)
            defects = cv2.convexityDefects(hand_contour, hull)
            hull = cv2.convexHull(hand_contour, returnPoints=True)
            cv2.drawContours(temp, [hull], 0, (0, 255, 0), 3)
            count = 0
            # ----
            finger = [(hull[0][0][0], hull[0][0][1])]
            flag = True
            for i in range(len(hull)):
                dist = np.sqrt((hull[-i][0][0] - hull[-i+1][0][0]) **
                               2 + (hull[-i][0][1] - hull[-i+1][0][1])**2)
                if dist > 18:
                    if flag:
                        finger = [(hull[-i][0][0], hull[-i][0][1])]
                    else:
                        flag = False
                        finger.append((hull[-i][0][0], hull[-i][0][1]))

            print('Len finger: ', len(finger))
            for i in range(len(finger)):
                (x, y) = finger[i]
                cv2.circle(temp, (x, y), 5, [0, 0, 255], -1)
            # count is keeping track of number of defect points
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                # If normal distance between farthest point(defect) and contour is > 14000 and < 28000, it is the desired defect point.

                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])
                # draw a circle/ dot at the defect point.
                # cv2.circle(temp, start, 5, [255, 255, 0], -1)
                # cv2.circle(temp, end, 5, [0, 255, 0], -1)
                # cv2.circle(temp, far, 5, [0, 255, 255], -1)
                # cv2.line(temp, start, end, [100, 100, 100], 3)
                # cv2.line(temp, start, far, [100, 100, 100], 3)
                # cv2.line(temp, far, end, [100, 100, 100], 3)
                # count is keeping track of number of defect points
                count += 1

            cv2.imshow("ve contour", temp)
            cv2.putText(frame, str(count+1), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 8)
            # -----

            # Tim vi tri trung tam cua hand
            # frame, hand_center, hand_radius, hand_size_score = mark_hand_center(
            #     frame_original, hand_contour)
            # Tinh moment cua contour --------------
            # M = cv2.moments(hand_contour)
            # ind_x, ind_y = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            # ind_x = ind_x + int(cap_region_x_begin*frame.shape[1])
            # dist = cv2.pointPolygonTest(hand_contour, (ind_x, ind_y), True)
            # print(ind_x, ind_y)
            # cv2.circle(frame, (ind_x, ind_y), 10, (255, 0, 0), 2)
            # ----------
            x, y, w, h = cv2.boundingRect(hand_contour)
            x = x + int(cap_region_x_begin*frame.shape[1])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            center = (int(x + w/2), int(y + h/2))
            # Neu tim duoc thi xac dinh ngon tya
            # if(hand_size_score):
            #     frame, finger, palm = mark_fingers(
            #         frame, hand_convex_hull, hand_center, hand_radius)
            #     frame, gesture_found = find_gesture(frame, finger, palm)
        else:
            frame = frame_original
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    elif k == ord('c'):
        if (bg_captured):
            capture_done = True
            hand_histogram = hand_capture(frame_original, box_pos_x, box_pos_y)
    elif k == ord('b'):
        bg_model = cv2.createBackgroundSubtractorMOG2()

        bg_captured = True
    elif k == ord('r'):
        capture_done = 0
        bg_captured = 0

cv2.destroyAllWindows()
