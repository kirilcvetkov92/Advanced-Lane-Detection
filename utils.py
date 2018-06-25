import cv2
import numpy as np

def cal_undistort(img, objpoints, imgpoints):
    # Image Calibration
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst, dist, mtx

def get_img_obj_points(img, nx, ny):
    imgpoints = []
    objpoints = []

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    #if corners are found
    if ret == True:
        imgpoints.append(corners)
        # Draw and display the corners
        objpoints.append(objp)

    return imgpoints, objpoints


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    gradient_direction = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(gradient_direction)
    binary_output[(gradient_direction >= thresh[0]) & (gradient_direction <= thresh[1])] = 1

    return binary_output



def sobel_filter(image, ksize=3):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, 'x', 10,200)
    grady = abs_sobel_thresh(gray, 'y', 10,200)
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7,1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined_condition = ((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))
    return combined_condition

def hls_filter(image):
        # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]

    # Threshold color channel
    s_thresh_min = 120
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary_condition = (s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)
    return s_binary_condition

def rgb_filter(image):
    # Extract RG colors for better yellow line isolation
    color_threshold = 155
    R = image[:, :, 0]
    G = image[:, :, 1]
    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_threshold) & (G > color_threshold)
    return r_g_condition

    
def filter_image(image):
    sobel_condition = sobel_filter(image.copy())
    hls_condition = hls_filter(image.copy())
    rgb_condition = rgb_filter(image.copy())
    height, width  = image.shape[0], image.shape[1]
    # apply the region of interest mask
    combined_binary = np.zeros((height, width), dtype=np.uint8)
    combined_binary[(rgb_condition) & (hls_condition | sobel_condition)] = 1

    mask = np.zeros_like(combined_binary)
    region_of_intersect = np.array([[0,height], [width/2, int(0.5*height)], [width, height]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_intersect], 1)
    thresholded = cv2.bitwise_and(combined_binary, mask)
    return thresholded


def get_curvature_radius(fit, ploty):
	x = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
	y_eval = np.max(ploty)
	curverad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	fit_cr = np.polyfit(ploty*ym_per_pix, x*xm_per_pix, 2)
	# Calculate the new radii of curvature
	curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
	# Now our radius of curvature is in meters
	return curverad