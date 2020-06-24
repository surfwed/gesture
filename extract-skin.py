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
        frame = cv2.imread('figure//hand3.jpg', 1)
        frame = cv2.resize(frame, (size_resize_width, size_resize_height))
        # smooth image
        kernel_3x3 = np.ones((3, 3), np.float32)
        kernel_5x5 = np.ones((5, 5), np.float32)
        frame = cv2.bilateralFilter(frame, 9, 75, 75)

        fr = frame.copy()
        hsv_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        ycrcb_img = cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)
        cv2.namedWindow("Original Frame")
        cv2.setMouseCallback("Original Frame", self.click)
        so = 1

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

            cv2.drawContours(frame, [hand_contour], 0, (255, 255, 0), 1)

            # Create mask where white is what we want, black otherwise
            only_mask_hand = np.zeros_like(mask_ycrcb)
            # Draw filled contour in mask
            cv2.drawContours(only_mask_hand, [hand_contour], 0, 255, -1)

            cv2.imshow("CalcBackProject YCrCb", dst_ycrcb)
            cv2.imshow("Threshold YCrCb", mask_ycrcb)
            cv2.imshow("Only Mask Hand", only_mask_hand)

            dist_ycrcb = cv2.distanceTransform(only_mask_hand, cv2.DIST_L2, 3)

            cv2.normalize(dist_ycrcb, dist_ycrcb, 0, 1.0, cv2.NORM_MINMAX)
            cv2.imshow('Distance Transform Image YCrCb', dist_ycrcb)

            _, dist_ycrcb = cv2.threshold(
                dist_ycrcb, 0.8, 1.0, cv2.THRESH_BINARY)
            cv2.imshow('Peaks', dist_ycrcb)
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

            # min_contour = min(contours, key=cv2.contourArea)
            # min_contour = np.reshape(min_contour, (-1, 2))
            # data = sorted(min_contour, key=lambda x: x[1])
            # center_inner_circle_x, center_inner_circle_y = data[0]
            # print(min_contour)
            # print(data[0])
            # print(fx, fy)
            # break
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
            palm_mask = []
            for i in range(0, 100, 1):
                point_x = center_inner_circle_x + \
                    int(math.sin(2*math.pi * i/100) * inner_radius * 1.2)
                point_y = center_inner_circle_y + \
                    int(math.cos(2*math.pi * i/100) * inner_radius * 1.2)

                if self.is_border(point_x, point_y, only_mask_hand):
                    palm_mask.append((point_x, point_y))
                    continue
                if point_y < 0 or point_x < 0 or point_y >= only_mask_hand.shape[0] or point_x >= only_mask_hand.shape[1]:
                    continue
                if only_mask_hand[point_y, point_x] == 0:
                    continue
                l_radius = 0
                r_radius = inner_radius*1.2
                radius_return = 0
                while (l_radius <= r_radius):
                    mid = l_radius + (r_radius - l_radius) / 2
                    if self.check_border(point_x, point_y, mid, only_mask_hand):
                        l_radius = mid + 1
                        radius_return = mid
                    else:
                        r_radius = mid - 1
                for i in range(0, 360, 1):
                    dot_x = point_x + \
                        int(math.sin(2*math.pi * i/360) * radius_return)
                    dot_y = point_y + \
                        int(math.cos(2*math.pi * i/360) * radius_return)
                    if self.is_border(dot_x, dot_y, only_mask_hand):
                        palm_mask.append((dot_x, dot_y))
                        continue
            palm_mask = np.array(palm_mask)
            # break
            # print("palm mask shape", palm_mask.shape)
            for point in palm_mask:
                lx, ly = point
                # print(lx, ly)
                cv2.circle(frame, (lx, ly), 2, [51, 214, 255], -1)

            # print(palm_mask)
            # cv2.circle(frame, (center_inner_circle_x, center_inner_circle_y), int(
            #     inner_radius), [0, 0, 255], 2)
            # cv2.circle(frame, (center_inner_circle_x, center_inner_circle_y), int(
            #     inner_radius*1.2), [255, 0, 0], 2)

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
                so = so + 1
                flag = 1
            elif k == ord('j'):
                so = so - 1
                if so < 0:
                    so = 1
                flag = 1
            elif k == ord('q'):
                break
            if flag:
                frame = cv2.imread('figure//hand'+str(so)+'.jpg', 1)
                if frame is None:
                    so = so - 1
                    continue
                frame = cv2.resize(
                    frame, (size_resize_width, size_resize_height))
                # smooth image
                kernel = np.ones((5, 5), np.float32)/25
                frame = cv2.bilateralFilter(frame, 9, 75, 75)

                # frame = cv2.filter2D(frame, -1, kernel)
                fr = frame.copy()
                hsv_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
                ycrcb_img = cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)

            cv2.putText(frame, "FPS={0}".format(round(1.0/elapsed_time, 2)), (10, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (209, 80, 0, 255), 1)
            cv2.imshow("Original Frame", frame)

        # cv2.waitKey()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    hand = Hand()
    hand.main()
