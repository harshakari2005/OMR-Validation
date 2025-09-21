import cv2
import numpy as np

def is_blurry(gray, threshold=100.0):
    # Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    return var < threshold, var

def brightness_score(gray):
    # mean brightness 0-255
    return float(np.mean(gray))

def estimate_skew_angle(gray):
    # Use edges and Hough lines to estimate dominant angle
    edges = cv2.Canny(gray,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for x1,y1,x2,y2 in lines[:,0]:
        angle = np.degrees(np.arctan2(y2-y1,x2-x1))
        angles.append(angle)
    if len(angles)==0:
        return 0.0
    median_angle = np.median(angles)
    return median_angle
