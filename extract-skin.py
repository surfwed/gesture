import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


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
            gx, gy = x, y

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
        for x in mx:
            for y in my:
                px = fx + x
                py = fy + y
                if (mx == 0 and my == 0) or (px < 0 or py < 0 or px >= only_mask_hand.shape[1] or py >= only_mask_hand.shape[0]):
                    continue
                if only_mask_hand[py, px] == 0:
                    okb = True
                if only_mask_hand[py, px] != 0:
                    okw = True
        return okw and okw

    def main(self):
        global gx, gy
        gx = 249
        gy = 312

        frame = cv2.imread('figure//hand3.jpg', 1)
        frame = cv2.resize(frame, (450, 600))
        # smooth image

        frame = cv2.bilateralFilter(frame, 9, 75, 75)

        fr = frame.copy()
        hsv_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        ycrcb_img = cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)
        cv2.namedWindow("Original Frame")
        cv2.setMouseCallback("Original Frame", self.click)
        so = 1
        while (1):
            frame = fr.copy()
            # cv2.imshow("HSV Frame", hsv_frame)
            # get in range
            # lh = cv2.getTrackbarPos('LH', 'track-bar')
            # hh = cv2.getTrackbarPos('HH', 'track-bar')
            # ls = cv2.getTrackbarPos('LS', 'track-bar')
            # hs = cv2.getTrackbarPos('HS', 'track-bar')
            # lv = cv2.getTrackbarPos('LV', 'track-bar')
            # hv = cv2.getTrackbarPos('HV', 'track-bar')
            # lower = np.array([lh, ls, lv], dtype=np.uint8)
            # upper = np.array([hh, hs, hv], dtype=np.uint8)
            # skinMask = cv2.inRange(hsv_frame, lower, upper)
            # cv2.imshow("skin-mask", skinMask)

            k = cv2.waitKey(1) & 0xFF

            px = gx
            py = gy
            m = 8
            n = 8
            width_length = 8
            height_length = 8
            padding_x = 3
            padding_y = 3
            # ve cac hinh vuong nao
            canvas_hsv = np.ones(
                (height_length*m, width_length*n, 3), dtype=np.uint8)
            canvas_ycrcb = np.ones(
                (height_length*m, width_length*n, 3), dtype=np.uint8)
            for y in range(0, m):
                for x in range(0, n):
                    x0 = px + (padding_x + width_length) * x
                    y0 = py + (padding_y + height_length) * y
                    cv2.rectangle(frame, (x0, y0), (x0 + width_length,
                                                    y0 + height_length), (255, 0, 0), 1)
                    canvas_hsv[y*height_length:(y+1)*height_length, x*width_length:(
                        x+1)*width_length] = hsv_frame[y0:y0+height_length, x0:x0+width_length]
                    canvas_ycrcb[y*height_length:(y+1)*height_length, x*width_length:(
                        x+1)*width_length] = ycrcb_img[y0:y0+height_length, x0:x0+width_length]
                    # canvas[y*height_length:(y+1)*height_length, x*width_length:(
                    #     x+1)*width_length] = fr[y0:y0+height_length, x0:x0+width_length]
            hand_hist_hsv = cv2.calcHist([canvas_hsv], [0, 1], None, [
                180, 256], [0, 180, 0, 256])
            hand_hist_ycrcb = cv2.calcHist([canvas_ycrcb], [1, 2], None, [
                10, 10], [0, 256, 0, 256])
            cv2.normalize(hand_hist_hsv, hand_hist_hsv,
                          0, 255, cv2.NORM_MINMAX)
            cv2.normalize(hand_hist_ycrcb, hand_hist_ycrcb,
                          0, 255, cv2.NORM_MINMAX)
            dst_hsv = cv2.calcBackProject(
                [hsv_frame], [0, 1], hand_hist_hsv, [0, 180, 0, 256], 1)
            dst_ycrcb = cv2.calcBackProject(
                [ycrcb_img], [1, 2], hand_hist_ycrcb, [0, 256, 0, 256], 1)

            ret, mask_hsv = cv2.threshold(
                dst_hsv, 10, 255, cv2.THRESH_BINARY)
            ret, mask_ycrcb = cv2.threshold(
                dst_ycrcb, 10, 255, cv2.THRESH_BINARY)

            kn = np.ones((3, 3), np.float32)
            # mask_hsv = cv2.erode(mask_hsv, kn, iterations=0)
            # mask_hsv = cv2.dilate(mask_hsv, kernel, iterations=0)
            # cv2.imshow("CalcBackProject HSV", dst_hsv)
            # cv2.imshow("Threshold HSV", mask_hsv)
            # --
            mask_ycrcb = cv2.erode(mask_ycrcb, kn, iterations=3)
            mask_ycrcb = cv2.dilate(mask_ycrcb, kernel, iterations=2)

            contours, hierarchy = cv2.findContours(
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
            # ---
            # cv2.imshow("Bitwise and", cv2.bitwise_and(mask_hsv, mask_ycrcb))
            # ---

            # --
            # dist_hsv = cv2.distanceTransform(mask_hsv, cv2.DIST_L2, 3)
            # Normalize the distance image for range = {0.0, 1.0}
            # so we can visualize and threshold it
            # cv2.normalize(dist_hsv, dist_hsv, 0, 1.0, cv2.NORM_MINMAX)
            # cv2.imshow('Distance Transform Image HSV', dist_hsv)
            # --
            dist_ycrcb = cv2.distanceTransform(only_mask_hand, cv2.DIST_L1, 3)
            # Normalize the distance image for range = {0.0, 1.0}
            # so we can visualize and threshold it
            cv2.normalize(dist_ycrcb, dist_ycrcb, 0, 1.0, cv2.NORM_MINMAX)
            cv2.imshow('Distance Transform Image YCrCb', dist_ycrcb)

            _, dist_ycrcb = cv2.threshold(
                dist_ycrcb, 0.9, 1.0, cv2.THRESH_BINARY)
            cv2.imshow('Peaks', dist_ycrcb)
            dist_ycrcb = dist_ycrcb.astype('uint8')

            contours, _ = cv2.findContours(
                dist_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_contour = min(contours, key=cv2.contourArea)
            min_contour = np.reshape(min_contour, (-1, 2))
            data = sorted(min_contour, key=lambda x: x[1])
            # print(min_contour)
            # print(data[0])
            center_inner_circle_x, center_inner_circle_y = data[0]
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
            # for i in range(0, 360, 1):
            #     point_x = center_inner_circle_x + \
            #         int(math.sin(2*math.pi * i/360) * inner_radius)
            #     point_y = center_inner_circle_y + \
            #         int(math.cos(2*math.pi * i/360) * inner_radius)
            #     if self.is_border(point_x, point_y, only_mask_hand):
            #         palm_mask.append((point_x, point_y))
            #         continue
            #     if point_y < 0 or point_y < 0 or point_y >= only_mask_hand.shape[0] or point_x >= only_mask_hand.shape[1]:
            #         continue
            #     if only_mask_hand[point_y, point_x] == 0:
            #         continue
            #     l_radius = 0
            #     r_radius = inner_radius
            #     radius_return = 0
            #     while (l_radius <= r_radius):
            #         mid = l_radius + (r_radius - l_radius) / 2
            #         if self.check_border(point_x, point_y, mid, only_mask_hand):
            #             l_radius = mid + 1
            #             radius_return = mid
            #         else:
            #             r_radius = mid - 1
            #     for i in range(0, 360, 1):
            #         dot_x = point_x + \
            #             int(math.sin(2*math.pi * i/360) * radius_return)
            #         dot_y = point_y + \
            #             int(math.cos(2*math.pi * i/360) * radius_return)
            #         if self.is_border(dot_x, dot_y, only_mask_hand):
            #             palm_mask.append((point_x, point_y))
            #             continue
            palm_mask = np.array(palm_mask)
            print(palm_mask.shape)
            # print(palm_mask)
            cv2.circle(frame, (center_inner_circle_x, center_inner_circle_y), int(
                inner_radius), [0, 0, 255], 2)
            cv2.circle(frame, (center_inner_circle_x, center_inner_circle_y), int(
                inner_radius*1.2), [255, 0, 0], 2)
            cv2.imshow("Original Frame", frame)
            # print(dist_ycrcb.shape)
            # print(dist_ycrcb)
            # break
            flag = 0
            if k == ord('k'):
                so = so + 1
                flag = 1
            if k == ord('j'):
                so = so - 1
                if so < 0:
                    so = 1
                flag = 1
            if flag:
                frame = cv2.imread('figure//hand'+str(so)+'.jpg', 1)
                frame = cv2.resize(frame, (450, 600))
                # smooth image
                kernel = np.ones((5, 5), np.float32)/25
                frame = cv2.bilateralFilter(frame, 9, 75, 75)

                # frame = cv2.filter2D(frame, -1, kernel)
                fr = frame.copy()
                hsv_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
                ycrcb_img = cv2.cvtColor(fr, cv2.COLOR_BGR2YCR_CB)
            if k == ord('q'):
                break

        # cv2.waitKey()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    hand = Hand()
    hand.main()
