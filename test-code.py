import cv2
import numpy as np


def nothing(x):
    pass

def cal_sigma(x):    
        return 1.0/(2.0*pow(x, 2))
if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.moveWindow('image',0,350)

    cv2.namedWindow('hsv')
    cv2.moveWindow('hsv',0,0)

    cv2.namedWindow('h image')
    cv2.moveWindow('h image',400,0)

    cv2.namedWindow('s image')
    cv2.moveWindow('s image',800,0)

    cv2.namedWindow('v image')
    cv2.moveWindow('v image',1200,0)

    cv2.namedWindow('b image')
    cv2.moveWindow('b image',400,350)

    cv2.namedWindow('g image')
    cv2.moveWindow('g image',800,350)

    cv2.namedWindow('r image')
    cv2.moveWindow('r image',1200,350)

    cv2.namedWindow('ht image')
    cv2.moveWindow('ht image',400,700)

    cv2.namedWindow('st image')
    cv2.moveWindow('st image',800,700)

    cv2.createTrackbar('H', 'image', 43, 180, nothing)
    cv2.createTrackbar('S', 'image', 52, 255, nothing)
    cv2.createTrackbar('Sigma', 'image', 8, 20, nothing)
    while (1):
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        frame_roi = frame[0:300, frame.shape[1] - 300:frame.shape[1]].copy()
        img = frame_roi   

        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('hsv',hsv_image)
        h, s, v = cv2.split(hsv_image)
        # h, s, v = hsv_image.copy(), hsv_image.copy(), hsv_image.copy()
        ret, ht = cv2.threshold(
                h, 90.0, 180.0, cv2.THRESH_BINARY)
        ret, st = cv2.threshold(
                s, 128.0, 256.0, cv2.THRESH_BINARY)

        # h[:,:,1] = 0
        # h[:,:,2] = 0
        # s[:,:,0] = 0
        # s[:,:,2] = 0
        # v[:,:,0] = 0
        # v[:,:,1] = 0
        # h, s, v = np.float64(h), np.float64(s), np.float64(v)
        b, g, r = cv2.split(img)
        # b, g, r = img.copy(), img.copy(), img.copy()
        # b[:,:,1] = 0
        # b[:,:,2] = 0
        # g[:,:,0] = 0
        # g[:,:,2] = 0
        # r[:,:,0] = 0
        # r[:,:,1] = 0
        # b, g, r = np.float64(b), np.float64(g), np.float64(r)
        cv2.imshow('image', img)
        cv2.imshow('h image',h)
        cv2.imshow('s image',s)
        cv2.imshow('v image',v)
        cv2.imshow('r image',r)
        cv2.imshow('ht image',ht)
        cv2.imshow('st image',st)   
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        hv = cv2.getTrackbarPos('H', 'image')
        sv = cv2.getTrackbarPos('S', 'image') 
        sigma = cv2.getTrackbarPos('Sigma', 'image')
        hv = hv - 90
        temp0 = h - ht
        temp1 = 0.5*(temp0 - hv)
        temp2 = 0.1*(s - sv)
        cv2.imshow('b image', temp1)
        cv2.imshow('g image', temp2)
        temp0 = cv2.pow(temp1, 2)
        temp1 = cv2.pow(temp2, 2)
        temp2 = temp0 + temp1
        temp0 = -(temp2 *cal_sigma(sigma) )
        temp1 = cv2.exp(temp0)
        temp2 = temp1*255.0   
        cv2.imshow('r image',temp2) 

    cv2.destroyAllWindows()
