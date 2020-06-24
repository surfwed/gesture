import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import time


class Hand:
    def __init__(self):
        pass

    def nothing(self):
        pass

    def trackbar(self):
        pass

    def click(self, event, x, y, flags, param):
        global gx, gy
        if event == cv2.EVENT_LBUTTONDOWN:
            gx = x
            gy = y
            return 0
        if event == cv2.EVENT_RBUTTONDOWN:
            return 1

    def check(self, radius, fx, fy, only_mask_hand):
        dem = 1000
        for i in range(0, 1000, 1):
            px = fx + int(math.sin(2*math.pi * i/1000) * radius)
            py = fy + int(math.cos(2*math.pi * i/1000) * radius)
            if (px < 0 or py < 0 or px >= only_mask_hand.shape[1] or py >= only_mask_hand.shape[0]):
                continue
            if only_mask_hand[py, px] == 0:
                # print(py, px, only_mask_hand[py, px])
                dem -= 1
        return dem/1000 > 0.8

    def check_border(self, fx, fy, radius, only_mask_hand):
        for i in range(0, 360, 1):
            px = fx + int(math.sin(2*math.pi * i/360) * radius)
            py = fy + int(math.cos(2*math.pi * i/360) * radius)
            if (px < 0 or py < 0 or px >= only_mask_hand.shape[1] or py >= only_mask_hand.shape[0]):
                continue
            if only_mask_hand[py, px] == 0:
                return False
        return True

    def is_border(self, fx, fy, only_mask_hand):
        okw = False
        okb = False
        mx = [-1, 0, 1]
        my = [-1, 0, 1]
        if (fx < 0 or fy < 0 or fx >= only_mask_hand.shape[1] or fy >= only_mask_hand.shape[0]):
            return False
        for x in mx:
            for y in my:
                px = fx + x
                py = fy + y
                if (mx == 0 and my == 0) or px < 0 or py < 0 or px >= only_mask_hand.shape[1] or py >= only_mask_hand.shape[0]:
                    continue
                if only_mask_hand[py, px] > 0:
                    okw = True
                else:
                    okb = True
        return okw and okb

    def calculateAngle(self, far, start, end):
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
        return angle

    def calculateDistance(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def main(self):
        global gx, gy

        pre_time = 0
        now_time = 0
        size_resize_width = 280
        size_resize_height = 280
        gx = int(size_resize_width / 2)
        gy = int(size_resize_height / 2)
        gx = 102
        gy = 116
        frame = cv2.imread('figure//hand16.jpg', 1)
        original_frame = frame.copy()
        frame = cv2.resize(frame, (size_resize_width, size_resize_height))
        # smooth image
        kernel_3x3 = np.ones((3, 3), np.float32)
        kernel_5x5 = np.ones((5, 5), np.float32)
        frame = cv2.bilateralFilter(frame, 9, 75, 75)

        fr = frame.copy()
        hsv_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        ycrcb_img = cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)
        cv2.namedWindow("After Resize Frame")
        cv2.setMouseCallback("After Resize Frame", self.click)
        index_image = 1

        # Khoi tao cac tham so lay mau
        number_rows = 5
        number_columns = 5
        box_sample_width_length = 8
        box_sample_height_length = 8
        padding_x = 3
        padding_y = 3

        while (1):
            now_time = time.time()
            elapsed_time = now_time - pre_time
            pre_time = now_time

            frame = fr.copy()

            k = cv2.waitKey(1) & 0xFF

            px = gx
            py = gy

            # -----------
            canvas_ycrcb = np.ones(
                (box_sample_height_length*number_rows, box_sample_width_length*number_columns, 3), dtype=np.uint8)

            # Lay Mau
            for y in range(0, number_rows):
                for x in range(0, number_columns):
                    x0 = px + (padding_x + box_sample_width_length) * x
                    y0 = py + (padding_y + box_sample_height_length) * y
                    cv2.rectangle(frame, (x0, y0), (x0 + box_sample_width_length,
                                                    y0 + box_sample_height_length), (255, 0, 0), 1)
                    canvas_ycrcb[y*box_sample_height_length:(y+1)*box_sample_height_length, x*box_sample_width_length:(
                        x+1)*box_sample_width_length] = ycrcb_img[y0:y0+box_sample_height_length, x0:x0+box_sample_width_length]

            # Tim vung anh co mau tuong tu
            hand_hist_ycrcb = cv2.calcHist([canvas_ycrcb], [1, 2], None, [
                10, 10], [0, 256, 0, 256])
            cv2.normalize(hand_hist_ycrcb, hand_hist_ycrcb,
                          0, 255, cv2.NORM_MINMAX)
            dst_ycrcb = cv2.calcBackProject(
                [ycrcb_img], [1, 2], hand_hist_ycrcb, [0, 256, 0, 256], 1)

            _, mask_ycrcb = cv2.threshold(
                dst_ycrcb, 10, 255, cv2.THRESH_BINARY)

            mask_ycrcb = cv2.erode(mask_ycrcb, kernel_3x3, iterations=3)
            mask_ycrcb = cv2.dilate(mask_ycrcb, kernel_5x5, iterations=2)

            contours, _ = cv2.findContours(
                mask_ycrcb, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # Lấy countours lớn nhất
            if (len(contours) != 0):
                hand_contour = max(contours, key=cv2.contourArea)

            # cv2.drawContours(frame, [hand_contour], 0, (255, 255, 0), 1)

            # Tao anh mask chi co ban tay
            only_mask_hand = np.zeros_like(mask_ycrcb)
            # Ve contour ban tay vao anh mask
            cv2.drawContours(only_mask_hand, [hand_contour], 0, 255, -1)

            leftmost = tuple(hand_contour[hand_contour[:, :, 0].argmin()][0])
            rightmost = tuple(hand_contour[hand_contour[:, :, 0].argmax()][0])
            topmost = tuple(hand_contour[hand_contour[:, :, 1].argmin()][0])
            bottommost = tuple(hand_contour[hand_contour[:, :, 1].argmax()][0])
            leftTop = (leftmost[0], topmost[1])
            rightBottom = (rightmost[0], bottommost[1])

            cv2.imshow("CalcBackProject YCrCb", dst_ycrcb)
            cv2.imshow("Threshold YCrCb", mask_ycrcb)
            cv2.imshow("Only Mask Hand", only_mask_hand)

            dist_ycrcb = cv2.distanceTransform(only_mask_hand, cv2.DIST_L2, 3)

            cv2.normalize(dist_ycrcb, dist_ycrcb, 0, 1.0, cv2.NORM_MINMAX)
            cv2.imshow('Distance Transform Image YCrCb', dist_ycrcb)

            _, dist_ycrcb = cv2.threshold(
                dist_ycrcb, 0.8, 1.0, cv2.THRESH_BINARY)
            # cv2.imshow('Peaks', dist_ycrcb)
            dist_ycrcb = dist_ycrcb.astype('uint8')

            contours, _ = cv2.findContours(
                dist_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if (len(contours) == 0):
                continue
            center_inner_circle_x, center_inner_circle_y = 0, 0
            for cnt in contours:
                extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
                if center_inner_circle_x == 0 and center_inner_circle_y == 0:
                    center_inner_circle_x, center_inner_circle_y = extTop[0], extTop[1]
                    continue
                if center_inner_circle_y > extTop[1]:
                    center_inner_circle_x, center_inner_circle_y = extTop[0], extTop[1]
                # print(extTop[0], extTop[1])
                # print(cnt.shape)

            cv2.circle(frame, (center_inner_circle_x,
                               center_inner_circle_y), 3, [0, 255, 255], -1)

            l_radius = 0
            r_radius = min(dist_ycrcb.shape)
            # print(l_radius, r_radius)

            inner_radius = 0
            while (l_radius <= r_radius):
                mid = l_radius + (r_radius - l_radius) / 2
                if self.check(mid, center_inner_circle_x, center_inner_circle_y, only_mask_hand):
                    l_radius = mid + 1
                    inner_radius = mid
                else:
                    r_radius = mid - 1

            clip_x = rightBottom[0]
            clip_y = center_inner_circle_y + int(inner_radius * 1.2)
            mask_hand_palm_rectangle = np.full_like(only_mask_hand, 0)
            cv2.rectangle(mask_hand_palm_rectangle, leftTop,
                          (clip_x, clip_y), 255, -1)

            # mask_hand_palm_remove = only_mask_hand.copy()
            mask_hand_palm_remove = cv2.bitwise_and(
                mask_hand_palm_rectangle, only_mask_hand)
            # cv2.circle(mask_hand_palm_remove, (center_inner_circle_x, center_inner_circle_y), int(
            #     inner_radius*1.2), 0, -1)
            # Anh ban tay sau khi bo bot cac phan thua
            cv2.imshow("Mask Hand Palm Remove", mask_hand_palm_remove)
            # Tiep tuc loai bo cac phan thua
            contour_mask_hand_palm_remove, _ = cv2.findContours(
                mask_hand_palm_remove, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if (len(contour_mask_hand_palm_remove) != 0):
                contour_mask_hand_palm_remove = max(
                    contour_mask_hand_palm_remove, key=cv2.contourArea)
            # contour_mask_hand_palm_remove = cv2.approxPolyDP(
            #     contour_mask_hand_palm_remove, 0.001*cv2.arcLength(contour_mask_hand_palm_remove, True), True)
            # print(contour_mask_hand_palm_remove.shape)
            # print(contour_mask_hand_palm_remove)
            # print(contour_mask_hand_palm_remove[1, 0])
            # so = 0
            # for x in contour_mask_hand_palm_remove:
            #     so += 1
            #     if so % 50 == 0:
            #         cv2.putText(frame, str(so), (x[0][0] - 10, x[0][1] - 10),
            #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255), 1)
            convex_hull_mask_hand = cv2.convexHull(
                contour_mask_hand_palm_remove, returnPoints=True)
            # print(convex_hull_mask_hand.shape)
            # break
            cv2.drawContours(
                frame, [contour_mask_hand_palm_remove], 0, (255, 255, 0), 1)
            cv2.drawContours(
                frame, [convex_hull_mask_hand], 0, (109, 107, 74), 1)
            convex_hull_mask_hand = cv2.convexHull(
                contour_mask_hand_palm_remove, returnPoints=False)
            thresh_finger_near = 15
            able_finger = []
            index_able_finger = []
            if (len(convex_hull_mask_hand) > 3):
                defects = cv2.convexityDefects(
                    contour_mask_hand_palm_remove, convex_hull_mask_hand)
                if type(defects) != type(None):
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i][0]
                        start = tuple(contour_mask_hand_palm_remove[s, 0])
                        end = tuple(contour_mask_hand_palm_remove[e, 0])
                        far = tuple(contour_mask_hand_palm_remove[f, 0])
                        # Tim cac diem co kha nang la dau ngon tay
                        # -----------------------------------------------------------------
                        if len(able_finger) == 0:
                            able_finger.append(start)
                            index_able_finger.append(s)
                        is_added = True
                        for i in range(len(able_finger)-1, len(able_finger) - 4 - 1, -1):
                            if i < 0:
                                break
                            if self.calculateDistance(able_finger[i], start) <= thresh_finger_near or self.calculateDistance(able_finger[0], start) <= thresh_finger_near:
                                is_added = False
                        if is_added:
                            able_finger.append(start)
                            index_able_finger.append(s)
                        is_added = True
                        for i in range(len(able_finger)-1, len(able_finger) - 4 - 1, -1):
                            if i < 0:
                                break
                            if self.calculateDistance(able_finger[i], end) <= thresh_finger_near or self.calculateDistance(able_finger[0], end) <= thresh_finger_near:
                                is_added = False
                        if is_added:
                            able_finger.append(end)
                            index_able_finger.append(e)
                        # -----------------------------------------------------------------

                        # cv2.circle(frame, start, 4, [119, 78, 47], -1)
                        # cv2.putText(frame, str(so), (start[0] - 10, start[1] - 10),
                        #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (100), 1)
                        # cv2.circle(frame, end, 4, [0, 54, 135], -1)
                        # cv2.putText(frame, str(so), (end[0] - 10, end[1] - 10),
                        #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255), 1)
                        # cv2.circle(frame, far, 4, [131, 52, 108], -1)
                        # end = tuple(hand_contour[e, 0])
                        # far = tuple(hand_contour[f, 0])
                        # angle = self.calculateAngle(far, start, end)
                        # if (d > 10000) and (angle <= math.pi/2) and (far[1] - cy) <= 0:
                        #     cv2.line(temp, start, end, [0, 255, 0], 5)
                        #     cv2.circle(temp, far, 5, [0, 0, 255], -1)
                        #     finger_count += 1
            so = 0
            for index in index_able_finger:
                so += 1
                index_padding = 50
                index_r = index + index_padding
                index_l = index - index_padding
                start = tuple(contour_mask_hand_palm_remove[index_l, 0])
                if index_r >= len(contour_mask_hand_palm_remove):
                    index_r = index_r - len(contour_mask_hand_palm_remove)
                end = tuple(contour_mask_hand_palm_remove[index_r, 0])
                far = tuple(contour_mask_hand_palm_remove[index, 0])
                angle = self.calculateAngle(far, start, end) * 180 / math.pi
                if angle < 60:
                    cv2.circle(frame, far, 4, [53, 67, 203], -1)
                    # print(original_frame.shape)
                    # print(frame.shape)
                    original_far_x = int(float(original_frame.shape[1] /
                                               frame.shape[1]) * far[0])
                    original_far_y = int(float(original_frame.shape[0] /
                                               frame.shape[0]) * far[1])
                    cv2.circle(original_frame, (original_far_x,
                                                original_far_y), 4, [53, 67, 203], -1)
                    cv2.circle(frame, start, 4, [163, 113, 36], -1)
                    cv2.circle(frame, end, 4, [0, 0, 128], -1)
                    cv2.putText(frame, str(round(angle, 2)), (far[0] - 10, far[1] - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (100), 1)
            # print(palm_mask)
            cv2.circle(frame, (center_inner_circle_x, center_inner_circle_y), int(
                inner_radius), [0, 0, 255], 2)
            cv2.circle(frame, (center_inner_circle_x, center_inner_circle_y), int(
                inner_radius*1.2), [255, 0, 0], 2)

            # print(dist_ycrcb.shape)
            # print(dist_ycrcb)
            # break
            flag = 0
            if k == ord('p'):
                number_columns += 1
                number_rows += 1
            elif k == ord('o'):
                number_columns -= 1
                number_rows -= 1
            elif k == ord('k'):
                index_image += 1
                flag = 1
            elif k == ord('j'):
                index_image -= 1
                if index_image <= 0:
                    index_image = 1
                flag = 1
            elif k == ord('q'):
                break
            if flag:
                frame = cv2.imread('figure//hand'+str(index_image)+'.jpg', 1)
                original_frame = frame.copy()
                if frame is None:
                    index_image -= 1
                    continue
                frame = cv2.resize(
                    frame, (size_resize_width, size_resize_height))
                # smooth image
                frame = cv2.bilateralFilter(frame, 9, 75, 75)

                # frame = cv2.filter2D(frame, -1, kernel)
                fr = frame.copy()
                ycrcb_img = cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)

            cv2.putText(frame, "FPS={0}".format(round(1.0/elapsed_time, 2)), (10, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (209, 80, 0, 255), 1)
            cv2.putText(frame, "hand{0}".format(str(index_image)), (150, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (209, 80, 0, 255), 1)

            cv2.imshow("After Resize Frame", frame)
            cv2.imshow("Original Frame", original_frame)

        # cv2.waitKey()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    hand = Hand()
    hand.main()
