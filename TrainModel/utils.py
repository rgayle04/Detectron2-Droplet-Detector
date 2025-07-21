import cv2 
import detectron2
import os
import numpy as np
import matplotlib.pyplot as plt
import math


g_pixelsPerUnit = 6.12


def getModelInput(image, sidelength):
    mi = cv2.resize(image, dsize=(sidelength, sidelength))
    mi = np.asarray(mi)
    mi = mi/255.0
    mi = np.expand_dims(mi, axis=0)
    return mi


# Output headers in order:
# Time Stamp, Adjusted Time, 
# Droplet 1 Radius, Droplet 1 Volume, Droplet 2 Radius, Droplet 2 Volume,
# Total Volume, DIB Radius, Contact Angle, Radial Distance
# Math from:
# Droplet Shape Analysis and Permeability Studies in Droplet Lipid Bilayers by Sanhita S. Dixit et. al.
def processframe(c1, c2):
    rDistance = math.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)

    # Two unconnected droplets; do not record
    if rDistance ** 2 >= (c1[2] + c2[2]) ** 2:
        return
    
    #print([c1, c2])
    
    # convert from pixels to microns and compensate for top-down 
    # perspective of microscope
    r1 = float(c1[2]) / g_pixelsPerUnit
    r2 = float(c2[2]) / g_pixelsPerUnit
    lf = rDistance / g_pixelsPerUnit
    lr = math.sqrt((r2-r1)**2. + lf**2.)

    if lr == 0:
        return None
    
    cos_theta=(lr**2 - (r1**2 + r2**2))/ (2 * r1 * r2)

    cos_theta= max(-1.0,min(1.0, cos_theta))
    thetab = math.acos(cos_theta)

    rdib = (r1 * r2 * math.sin(thetab)) / lr
    
    if rdib == 0:
        return None

    # 1/2 of DIB angle is what chemists use
    theta_degrees = (180. * thetab) / math.pi
    theta_degrees /= 2.

    if theta_degrees == 0:
        return None


    # dome heights for volume of sphere - dome
    a = ((r1**2. - r2**2.) + lr**2.) / (2. * lr)
    b = lr - a
    c1h = r1 - a
    c2h = r2 - b

    v1 = (4. * math.pi * r1**3.) / 3.
    v1 -= (math.pi * c1h * (3. * rdib**2. + c1h**2.)) / 6.

    v2 = (4. * math.pi * r2**3.) / 3.
    v2 -= (math.pi * c2h * (3. * rdib**2. + c2h**2.)) / 6.

    tv = v1 + v2

    #print([r1, v1, r2, v2, tv, rdib, theta_degrees, lr])
    # input()
    return r1, v1, r2, v2, tv, rdib, theta_degrees, lr
    
def getBoundingSquare(bbox, h, w):
    (bbx1, bby1, bbx2, bby2) = bbox 
    (bcx, bcy) = ((bbx1+bbx2)/2, (bby1+bby2)/2)
    halfside = max([bbx2-bbx1, bby2-bby1])/2
    (bsx1,bsy1,bsx2,bsy2) = (int(bcx-halfside), int(bcy-halfside),int(bcx+halfside), int(bcy+halfside))

    if bsx1 < 0:
        bsx2 = int(2*halfside)
        bsx1 = 0 
    if bsy1 < 0:
        bsy2 = int(2* halfside)
        bsy1 = 0 
    if bsx2 >= w:
        bsx1-=(bsx2-(w-1))
        bsx2=w-1
    if bsy2 >= h:
        bsy1-=(bsy2-(h-1))
        bsy2 = h-1 

    return bsx1, bsy1, bsx2, bsy2

def getcb2cImg(image, box, debugmode=False):
    (h,w) = map(lambda x: int(x/2), image.shape[:2])
    cbimage=np.zeros((h,w,3), np.uint8)
    cbimage[:] = (255, 255, 255)
    crop_h = box[3]-box[1]
    crop_w = box[2]-box[0]
    y_offset = (h - crop_h)//2
    x_offset = (w - crop_w)//2

    cbimage[y_offset: y_offset + crop_h, x_offset: x_offset+crop_w] = image[box[1]:box[3], box[0]:box[2]].copy()

    if debugmode is True:
        cv2.imshow('cb2c image', cbimage)
        key = cv2.waitKey(0)
        if key == 27:
            exit()
    return cbimage