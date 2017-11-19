# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/aligned000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/aligned001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    original_image = cv2.resize(cv2.imread("images/aligned001.jpg"), (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))
    #cv2.imshow("", np.array(proj_mask * 255, dtype=np.uint8))
    #cv2.waitKey()  

    scan_bits = np.zeros((h,w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt = cv2.resize(cv2.imread("images/aligned%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        scan_bits[on_mask] += bit_code
    

        # TODO: populate scan_bits by putting the bit_code according to on_mask

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    global camera_points
    global projector_points
    global colors
    colors = []
    camera_points = []
    projector_points = []

    correspending_image = np.zeros((scan_bits.shape[0], scan_bits.shape[1], 3))
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code
            x_p, y_p = binary_codes_ids_codebook[scan_bits[y,x]]
            if x_p >= 1279 or y_p >= 799: # filter
                continue
            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
            camera_points.append([x/2.,y/2.])
            projector_points.append([x_p, y_p])

            correspending_image[y, x] = np.array([0, projector_points[-1][1], projector_points[-1][0]])
            colors.append(original_image[y,x][::-1])
    colors = np.array(colors)
    #cv2.normalize(correspending_image,  correspending_image, 0, 255, cv2.NORM_MINMAX)
    #cv2.imshow("", np.array(correspending_image, dtype=np.uint8))
    #cv2.waitKey()  
 
    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']


    #import pdb
    #pdb.set_trace()

    camera_points = np.array(camera_points,dtype = np.float32)
    projector_points = np.array(projector_points, dtype = np.float32) 

    camera_points = cv2.undistortPoints(camera_points.reshape((camera_points.shape[0], 1, 2)), camera_K, camera_d)
    projector_points = cv2.undistortPoints(projector_points.reshape((projector_points.shape[0], 1, 2)), projector_K, projector_d)

    camera_points = camera_points.reshape(camera_points.shape[0],2)
    projector_points = projector_points.reshape(projector_points.shape[0],2)

    
    camera_P = np.eye(4)[:3]
    projector_P = np.hstack((projector_R,projector_t))
    
    
    points_3d = cv2.triangulatePoints(camera_P, projector_P, camera_points.transpose(), projector_points.transpose())
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.transpose())
    mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)
    mask = mask.reshape((mask.shape[0],))

    camera_points = camera_points[mask]
    projector_points = projector_points[mask]
    points_3d = points_3d[mask]
    colors = colors[mask]
    
    return points_3d,colors
    
def write_3d_points(points_3d, colors):
    
    # ===== DO NOT CHANGE THIS FUNCTION =====
    
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))
    
    #output_name = sys.argv[1] + "output.obj"
    #with open(output_name,"w") as f:
    #    for p,c in zip(points_3d, colors):
    #        f.write("v %d %d %d %f %f %f\n"%(p[0,0],p[0,1],p[0,2], c[0], c[1], c[2]))
            #f.write("v %d %d %d\n"%(p[0,0],p[0,1],p[0,2]))

    return points_3d
    
if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====
    
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d,colors = reconstruct_from_binary_patterns()
    write_3d_points(points_3d, colors)
    
