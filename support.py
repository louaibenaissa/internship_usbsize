import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import os
import seaborn as sns


def load_images(path) :
    """ 
    load_images function takes in path, loads the image and converts its color to grayscale.

    @path : path to the image.
    """
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (1000,1000))
    return img


lower_blue = np.array([80, 5, 5])
upper_blue = np.array([150, 255, 255])

def cntr_find(imgs):
    """
    cntr_find function utilises a mask to make the countour detection easier, and finds the usb box's edge points coordinates.

    @imgs : image where we find the contours.
    """

    img0 = imgs.copy()

    hsv = cv.cvtColor(imgs, cv.COLOR_RGB2HSV)      
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    contours ,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    contouredimg=imgs.copy()
    cv.drawContours(contouredimg, [largest_contour], -1, (191,213,229),thickness=cv.FILLED)
    blured_img = cv.cvtColor(cv.medianBlur(contouredimg, 3), cv.COLOR_RGB2GRAY)
    _,th = cv.threshold(blured_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv.contourArea)
    contoured_img2= cv.drawContours(img0, [max_contour], -1, (0, 255, 0), 2)

    hull = cv.convexHull(max_contour)
    epsilon = 0.02 * cv.arcLength(hull, True)
    approx = cv.approxPolyDP(hull, epsilon, True)
    return approx


def box_img(pts, img):
    """
    box_img : function that takes in the points of the edges and warp transforms the image to a perspective-corrected square.

    @pts : edge points.
    @img : image to be boxed.
    """
    output = np.float32([[0,0],[519,0],[519,519],[0,519]])
    approx = np.float32(pts)
    approx = np.reshape(approx,(4,2))
    M = cv.getPerspectiveTransform(approx, output)
    out = cv.warpPerspective(img,M,(520, 520),flags=cv.INTER_LINEAR)
    return out


def equalizer(img):
    """
    equalizer function applies Contrast Limited Adaptive Histogram Equalization to the image.

    @img : the image where clahe should be applied.
    """
    b, g, r = cv.split(img)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    r_clahe = clahe.apply(r)

    image_clahe = cv.merge((b_clahe, g_clahe, r_clahe))
    return image_clahe

# Red color range to detect the ASA logo.
lower_red_orange = np.array([0, 120, 50])
upper_red_orange = np.array([5, 255, 255])

def mask_red_orange(image):
    """
    mas_red_orange function takes an image, and applies a red/orange masks keeping only that color, to detect ASA's logo.

    @image : image to be masked.
    """
    image = image.copy()
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    mask = cv.inRange(hsv, lower_red_orange, upper_red_orange)

    result = cv.bitwise_and(image, image, mask=mask)
    return result

def rotate(img):
    """ 
    rotate function uses the logo's placement, to correct the orientation of the image (the logo should be on the top right of the usb).

    @img : image to be rotated.
    """

    immg = mask_red_orange(img)
    r, g, b = cv.split(immg)
    corner = list()
    corner.append(np.float32(r[0:120,400:519]).mean())
    corner.append(np.float32(r[0:120,0:150]).mean())
    corner.append(np.float32(r[400:519,0:120]).mean())
    corner.append(np.float32(r[400:519,400:519]).mean())
    hh = corner.index(max(corner))

    h, w = immg.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv.getRotationMatrix2D(center, hh*-90, 1.0)
    new_image = np.zeros((h, w, immg.shape[2]), dtype=np.uint8)

    new_img = cv.warpAffine(img, rotation_matrix, (w, h), dst=new_image, flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE) 
    return new_img


def get_num(img):
    """ 
    get_num function crops the image to keep the USB size only.
    
    @img : Image to be cropped.
    """
    num = img[200:300,350:500]
    return num


