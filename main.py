from collections import deque
import cv2
import numpy as np
import math
import tensorflow.keras as keras
import tensorflow as tf



capture_box_count = 9
capture_box_dim = 10
capture_pos_x = 500
capture_pos_y = 150
capture_box_sep_x = 8
capture_box_sep_y = 18

cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width


def hand_capture(frame_in, box_x, box_y):
    # Chuyển hình ảnh ban đầu thành mã màu HSV
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    ROI = np.zeros([capture_box_dim*len(box_x),
                    capture_box_dim, 3], dtype=hsv.dtype)
    for i in range(len(box_x)):
        ROI[i*capture_box_dim:i*capture_box_dim+capture_box_dim, 0:capture_box_dim] = hsv[box_y[i]
            :box_y[i]+capture_box_dim, box_x[i]:box_x[i]+capture_box_dim]
    hand_hist = cv2.calcHist([ROI], [0, 1], None, [20, 20], [0, 180, 0, 256])
    cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist


def hand_threshold(frame_in, hand_hist):
    #frame_in = cv2.medianBlur(frame_in, 3)
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    # Tinh back project
    back_projection = cv2.calcBackProject(
        [hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)
    ret, thresh = cv2.threshold(
        back_projection, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    # fgmask = cv2.morphologyEx(
    #     back_projection, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = thresh
    # fgmask = cv2.dilate(back_projection, kernel,  iterations=2)
    mask = cv2.dilate(mask, kernel,  iterations=2)
    return mask


def remove_bg(frame_roi):
    mask = bg_model.apply(frame_roi, learningRate=0)
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    ret, mask = cv2.threshold(
        mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
def determine_direction(diff):
    diffx, diffy = diff[0], diff[1]
    if abs(diffx) <= 10 and abs(diffy) <= 10:
        return "St"
    elif diffx > 10 and abs(diffy) <= 10:
        return "E"
    elif diffx < -10 and abs(diffy) <= 10:
        return "W"
    elif abs(diffx) <= 10 and diffy < -10:
        return "N"
    elif abs(diffx) <= 10 and diffy > 10:
        return "S"
    elif diffx > 15 and diffy > 15:
        return "SE"
    elif diffx < -15 and diffy > 15:
        return "SW"
    elif diffx > 15 and diffy < -15:
        return "NE"
    elif diffx < -15 and diffy < -15:
        return "NW"


def process_created_gesture(created_gesture):
    if created_gesture != []:
        for i in range(created_gesture.count(None)):
            created_gesture.remove(None)
        for i in range(created_gesture.count('St')):
            created_gesture.remove('St')

        if len(created_gesture) < 2:
            return created_gesture
        copy = [created_gesture[0], ]
        for i in range(1, len(created_gesture)):
            if created_gesture[i] != copy[len(copy)-1]:
                copy.append(created_gesture[i])

        created_gesture = copy
    return created_gesture

def resizeKeepRation(image, size, interpolation):
    h = image.shape[0]
    w = image.shape[1]    
    diff = max(h, w)
    padding = int(3*diff / 8)
    mask = np.zeros((diff+2*padding, diff+2*padding), dtype=image.dtype)
    x_pos = int((mask.shape[1] - w) / 2.0)
    y_pos = int((mask.shape[0] - h) / 2.0)
    mask[y_pos:y_pos+h,x_pos:x_pos+w] = image[:,:]
    return cv2.resize(mask, (size, size), interpolation)
# ----------------- MAIN -------------------- #
if __name__ == "__main__":
    
    new_model = tf.keras.models.load_model('epic_num_reader.model')
    pts = deque(maxlen=500)
    box_pos_x = []
    box_pos_y = []
    n = 6
    for i in range(n):
        box_pos_x.append(capture_pos_x + capture_box_dim*i + capture_box_sep_x*i)
        temp = [capture_pos_y + capture_box_dim*i + capture_box_sep_y * i]*n
        box_pos_y += temp
    box_pos_x = box_pos_x*n
    # print(box_pos_x)
    # print(box_pos_y)
    camera = cv2.VideoCapture(0)
    capture_done = False
    bg_captured = False
    hand_histogram = None
    listText = []
    black_board = np.zeros((384, 320))
    debug_frame = None
    debug = False
    DuDoan = ""
    cv2.namedWindow("Anh")                  
    cv2.moveWindow("Anh",0,550)
    cv2.namedWindow("ve contour")
    cv2.moveWindow("ve contour",500, 500)
    cv2.namedWindow('frame')
    cv2.moveWindow('frame',0,0)
    while(1):

        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.bilateralFilter(frame, 5, 0, 0)
        frame_original = np.copy(frame)
        # Ve hinh chu nhat
        cv2.rectangle(frame, (int(cap_region_x_begin*frame.shape[1]), 0), (frame.shape[1], int(
            cap_region_y_end*frame.shape[0])), (255, 0, 0), 1)

        frame_roi = frame_original[0:int(cap_region_y_end*frame.shape[0]), int(cap_region_x_begin *
                                                                            frame.shape[1]):frame.shape[1]].copy()
        frame_roi = cv2.medianBlur(frame_roi, 5)
        # print(frame_roi.shape)
        # Neu da cap background roi thi:
        if(bg_captured):
            # Ham remove background
            bkg_mask, bkg_frame = remove_bg(frame_roi)
            #cv2.imshow("Hinh nen",bkg_frame)

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

            # Ve cac o vuong hinh chu nhat de laqqqy mau ban tay
            # Toa do cac o
            

            # Duyet for ve cac o
            for i in range(len(box_pos_x)):
                cv2.rectangle(frame, (box_pos_x[i], box_pos_y[i]), (
                    box_pos_x[i]+capture_box_dim, box_pos_y[i]+capture_box_dim), (255, 0, 0), 1)
            hand_hist_temp = hand_capture(frame_original, box_pos_x, box_pos_y)
            back_projection = hand_threshold(frame_roi, hand_hist_temp)
            # back_projection = cv2.medianBlur(back_projection, median_ksize)
            cv2.imshow("cacl back histogram", back_projection)
        else:
            finger_count = 0
            fgmask = hand_threshold(bkg_frame, hand_histogram)
            cv2.imshow("bkg mask vs calBack histogram", fgmask)
            #cv2.imshow("bkg mask", bkg_mask)

            # -----
            contours, hierarchy = cv2.findContours(
                fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if (len(contours) != 0):
                hand_contour = max(contours, key=cv2.contourArea)
            # -----
            # cv2.imshow('frame threshold', frame)

            if(len(contours) != 0):
                temp = frame_roi
                #print("So contours tim duoc la: %d" % len(contours))
                M = cv2.moments(hand_contour)
                points1 = []
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(temp, (cx, cy), 3, [0, 255, 255], -1)
                hand_contour = cv2.approxPolyDP(
                    hand_contour, 0.02*cv2.arcLength(hand_contour, True), True)
                hull = cv2.convexHull(hand_contour, returnPoints=True)
                for point in hull:
                    if (cy > point[0][1]):
                        points1.append(tuple(point[0]))

                # ---- ve hand contour
                cv2.drawContours(temp, [hand_contour], 0, (255, 255, 0), 1)
                #cv2.drawContours(temp, [hull], 0, (255, 0, 255), 1)
                # hull = cv2.convexHull(hand_contour, returnPoints=True)
                # cv2.drawContours(temp, [hand_contour], -1, (255, 0, 0), 3)
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
                cv2.putText(frame, str(finger_count+1), (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 1, 8)
                direction = ""
                dX = 0
                dY = 0
                # -- Xu ly finger count
                if finger_count != 1:
                    extTop = tuple(hand_contour[hand_contour[:, :, 1].argmin()][0])
                    cv2.circle(temp, extTop, 3, [100, 0, 255], -1)
                    pts.appendleft(extTop)
                    for i in range(1, len(pts)):
                        if len(pts) >= 12 and i == 1:
                            dX = pts[0][0] - pts[10][0]
                            dY = pts[0][1] - pts[10][1]
                        cv2.line(temp, pts[i-1], pts[i], (0, 0, 255), 8)
                        cv2.line(black_board, pts[i-1], pts[i], (255, 255, 255), 8)
                        
                        
                    text = determine_direction((dX, dY))
                    cv2.putText(temp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 0, 255), 1, 2)
                    cv2.putText(temp, "dx: {}, dy: {}".format(dX, dY),
                                (10, temp.shape[1] -
                                10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, (0, 0, 255), 1)
                    listText.append(text)
                    pass
                # Neu hai ngon
                elif finger_count == 1:
                    if (len(listText) > 0):
                        temp_list = process_created_gesture(listText)
                        print(temp_list)
                    listText = []
                    pts = deque(maxlen=500)
                    # ------
                    x1, x2, y1, y2 = -1, -1, -1, -1
                    for y in range(black_board.shape[0]):
                        for x in range(black_board.shape[1]):
                            if black_board[y,x] >0:
                                if x1 == -1 : 
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
                        x, y, w, h = x1, y1, x2 - x1, y2- y1
                        # ----                        
                        if (x >0 and y>0 and w>0 and h>0):
                            print(x,y,w,h)                        
                            debug_frame = np.copy(black_board[y:y+h, x:x+w])                            
                            # cv2.namedWindow("black_board")                
                            # cv2.moveWindow("black_board", 700, 0)
                            # cv2.rectangle(black_board, (x,y),(x+w,y+h), [200,100,0], 1)
                            # cv2.imshow("black_board", black_board)                            
                        debug = True
                        anh_digit = resizeKeepRation((debug_frame), 28, interpolation = cv2.INTER_AREA)
                        anh_digit = tf.keras.utils.normalize(anh_digit, axis = 1)                        
                        anh_digit = np.reshape(anh_digit, (1,28,28))                        
                        predictions = new_model.predict(anh_digit)
                        nu = np.argmax(predictions[0])
                        print("So du doan",nu)    
                        print("Ti le du doan",predictions[0][nu]*100)    
                        # ------
                    black_board = np.zeros((384, 320))
                
                #black = np.ones((h,w))
                #black = black_board[y:h,x:w]
                if debug:      
                    canvas = np.zeros((28,28))
                    # ret, anh = cv2.threshold(debug_frame, 200, 255,cv2.THRESH_BINARY )  
                    # anh = resizeKeepRation(anh, 28, interpolation = cv2.INTER_AREA)
                    canvas[:] = anh_digit[0]                    
                    cv2.imshow("Anh",canvas)
                #     print("debug_frame",debug_frame.shape)
                #     cv2.namedWindow("anh")
                #     cv2.moveWindow("anh", 0, 500)
                #     cv2.imshow("anh", debug_frame)               
                
                cv2.imshow("ve contour", temp)
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

    cv2.destroyAllWindows()
