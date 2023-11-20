import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import io, feature
import pywt
def home_page():
    st.title("Image Upload App")

    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        st.header("Upload Any Kind of Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "tiff", "tif"])

        if uploaded_file is not None:
            fingerprint_image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

            st.session_state.fingerprint_image = fingerprint_image

            st.success("Image Uploaded Successfully!")

def display_uploaded_image_page():
    st.title("Display Uploaded Image")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    st.image(fingerprint_image, caption="Fingerprint Image", channels="BGR", use_column_width=True)

    st.write("Saving the uploaded image...")
    cv2.imwrite("uploaded_image.jpg", cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2RGB))
    st.success("Image Saved Successfully!")

def lbp_page():
    st.title("LBP Feature Extraction")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image
    lbp_image = perform_lbp_feature_extraction(fingerprint_image)
     # Perform histogram equalization on the LBP image
    lbp_equalized = cv2.equalizeHist(lbp_image)
 # Calculate the histogram of the original LBP image
    hist_lbp, _ = np.histogram(lbp_image.flatten(), 256, [0, 256])

    
    plt.figure(figsize=(8, 8))
    plt.imshow(lbp_image, cmap='gray')
    plt.axis('off')
    # Display the original LBP image
    plt.subplot(2, 2, 1)
    plt.imshow(lbp_image, cmap='gray')
    plt.axis('off')
    

    # Display the histogram of the original LBP image
    plt.subplot(2, 2, 3)
    plt.plot(hist_lbp, color='b')
    plt.title("Histogram of  LBP Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    st.pyplot()

    st.write("Saving the LBP image...")
    cv2.imwrite("LBP_image.jpg", lbp_image)
    st.success("LBP Image Saved Successfully!")

def perform_lbp_feature_extraction(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform LBP feature extraction
    lbp_image = np.zeros_like(gray)
    height, width = gray.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center = gray[i, j]
            val_ar = [
                gray[i - 1, j - 1] >= center,
                gray[i - 1, j] >= center,
                gray[i - 1, j + 1] >= center,
                gray[i, j + 1] >= center,
                gray[i + 1, j + 1] >= center,
                gray[i + 1, j] >= center,
                gray[i + 1, j - 1] >= center,
                gray[i, j - 1] >= center
            ]
            val = sum([val_ar[p] * (2 ** p) for p in range(8)])
            lbp_image[i, j] = val
            
    return lbp_image
      

def ltp_upper_page():
    st.title("LTP_Upper Feature Extraction")
    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return
    fingerprint_image = st.session_state.fingerprint_image
    ltp_upper_image = perform_ltp_upper(fingerprint_image)
    # Perform histogram equalization on the LTP_Upper image
    ltp_upper_equalized = cv2.equalizeHist(ltp_upper_image)
    
    # Calculate the histogram of the original LTP_Upper image
    hist_ltp_upper, _ = np.histogram(ltp_upper_image.flatten(), 256, [0, 256])
    plt.figure(figsize=(8, 8))
    
    # Display the original LTP_Upper image
    plt.subplot(2, 2, 1)
    plt.imshow(ltp_upper_image, cmap='gray')
    plt.axis('off')
    # Display the histogram of the original LTP_Upper image
    plt.subplot(2, 2, 3)
    plt.plot(hist_ltp_upper, color='b')
    plt.title("Histogram of  LTP_Upper Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.savefig("LTP_Upper.png")
    st.pyplot()
    st.success("LTP_Upper Image Saved Successfully!")

def perform_ltp_upper(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ltp_upper = np.zeros_like(gray_img, dtype=np.uint8)
    for i in range(1, gray_img.shape[0] - 1):
        for j in range(1, gray_img.shape[1] - 1):
            center_pixel = gray_img[i, j]
            neighbors = [gray_img[i-1, j], gray_img[i, j+1], gray_img[i+1, j], gray_img[i, j-1]]
            ltp_upper[i, j] = int(np.all(center_pixel >= neighbors))
    return ltp_upper

def ltp_lower_page():
    st.title("LTP_Lower Feature Extraction")
    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return
    fingerprint_image = st.session_state.fingerprint_image
    ltp_lower_image = perform_ltp_lower(fingerprint_image)
     # Perform histogram equalization on the LTP_Lower image
    ltp_lower_equalized = cv2.equalizeHist(ltp_lower_image)
    
    # Calculate the histogram of the original LTP_Lower image
    hist_ltp_lower, _ = np.histogram(ltp_lower_image.flatten(), 256, [0, 256])
    
    plt.figure(figsize=(8, 8))
    
    # Display the original LTP_Lower image
    plt.subplot(2, 2, 1)
    plt.imshow(ltp_lower_image, cmap='gray')
    plt.axis('off')
    # Display the histogram of the original LTP_Lower image
    plt.subplot(2, 2, 3)
    plt.plot(hist_ltp_lower, color='b')
    plt.title("Histogram of  LTP_Lower Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.savefig("LTP_Lower.png")
    st.pyplot()
    st.success("LTP_Lower Image Saved Successfully!")

def perform_ltp_lower(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ltp_lower = np.zeros_like(gray_img, dtype=np.uint8)
    for i in range(1, gray_img.shape[0] - 1):
        for j in range(1, gray_img.shape[1] - 1):
            center_pixel = gray_img[i, j]
            neighbors = [gray_img[i-1, j], gray_img[i, j+1], gray_img[i+1, j], gray_img[i, j-1]]
            ltp_lower[i, j] = int(np.all(center_pixel > neighbors))
    return ltp_lower


import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ... (Existing code)
def rgb_splitter_page():
    st.title("RGB Splitter")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Split the image into RGB channels
    red_channel = fingerprint_image[:, :, 2]
    green_channel = fingerprint_image[:, :, 1]
    blue_channel = fingerprint_image[:, :, 0]

    # Display the RGB split images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(red_channel, cmap='Reds')
    ax[0].set_title('Red Channel', fontsize=15)
    ax[0].axis('off')
    ax[1].imshow(green_channel, cmap='Greens')
    ax[1].set_title('Green Channel', fontsize=15)
    ax[1].axis('off')
    ax[2].imshow(blue_channel, cmap='Blues')
    ax[2].set_title('Blue Channel', fontsize=15)
    ax[2].axis('off')

    st.pyplot(fig)

 # Compute histograms for each RGB channel
    hist_red, _ = np.histogram(red_channel.ravel(), bins=256, range=(0, 256))
    hist_green, _ = np.histogram(green_channel.ravel(), bins=256, range=(0, 256))
    hist_blue, _ = np.histogram(blue_channel.ravel(), bins=256, range=(0, 256))

    # Display histograms
    plt.figure(figsize=(15, 4))

    plt.plot(hist_red, color='r')
    plt.xlim([0, 256])
    plt.title('Red Channel Histogram')
    st.pyplot()

    plt.figure(figsize=(15, 4))
    plt.plot(hist_green, color='g')
    plt.xlim([0, 256])
    plt.title('Green Channel Histogram')
    st.pyplot()

    plt.figure(figsize=(15, 4))
    plt.plot(hist_blue, color='b')
    plt.xlim([0, 256])
    plt.title('Blue Channel Histogram')
    st.pyplot()


def histogram_equalization_page():
    st.title("Histogram Equalization")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to grayscale for histogram equalization
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # Calculate the histogram of the original and equalized image
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # Plot the original histogram and CDF
    plt.plot(cdf_normalized, color='b')
    plt.hist(gray_image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histogram'), loc='upper left')
    plt.title("Histogram Equalization")
    plt.savefig("histogram_equalization.png")
    st.pyplot()
    st.success("Histogram_Equalization Image Saved Successfully!")
    

    # Display the equalized image
    #st.image(equalized_image, caption="Equalized Image", channels="GRAY", use_column_width=True)


def grayscale_page():
    st.title("Grayscale Transformation")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)
    # Calculate the histogram of the grayscale image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
 # Display the equalized grayscale image
    st.image(gray_image, use_column_width=True, caption='Grayscale Image')


    # Display the histogram
    st.write('Histogram:')
    st.bar_chart(hist)
    plt.savefig("grayscale.png")
    st.pyplot()
    st.success("Grayscale Image Saved Successfully!")
    

def resizing_page():
    st.title("Image Resizing")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Resize the image using different interpolation techniques
    half = cv2.resize(fingerprint_image, (0, 0), fx=0.5, fy=0.5)
    bigger = cv2.resize(fingerprint_image, (1050, 1610))
    stretch_near = cv2.resize(fingerprint_image, (780, 540), interpolation=cv2.INTER_NEAREST)
    # Perform histogram equalization on the original image
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
 # Calculate histogram of the equalized image
    hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
   # Display the resized images and the histogram for the equalized grayscale image
    Titles = ["Original", "Half", "Bigger", "Interpolation Nearest"]
    images = [fingerprint_image, half, bigger, stretch_near,]
    count = 4

    for i in range(count):
        plt.subplot(2, 2, i + 1)
        plt.title(Titles[i])
        plt.imshow(images[i])
        plt.axis('off')

    st.pyplot()
     # Display the histogram
    st.subheader("Resizing Histogram")
    plt.figure(figsize=(6, 3))
    plt.title("Resizing Histogram")
    plt.plot(hist)
    plt.xlim([0, 256])
    st.pyplot()

    

    
def kmean_clustering_page():
    st.title("K-Mean Clustering")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to RGB
    image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2RGB)
    

    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))

    # Convert to float
    pixel_values = np.float32(pixel_values)

    # Define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Number of clusters (K)
    k = 3

    # Perform K-Means clustering
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Flatten the labels array
    labels = labels.flatten()

    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels]

    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    
    # Show the segmented image
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.savefig("kmean_clustering.png")
    st.pyplot()
# Calculate and plot the histogram of the segmented image
    hist_segmented = cv2.calcHist([segmented_image], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histogram of kmean clustering  Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist_segmented, color='b')
    st.pyplot()
    st.success(" Kmean_Clustering Image Saved Successfully!")




import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ... (Existing code)

def kmean_clustering_k2_page():
    st.title("K-Mean Clustering (K=2)")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to RGB
    image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))

    # Convert to float
    pixel_values = np.float32(pixel_values)

    # Define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Number of clusters (K)
    k = 2

    # Perform K-Means clustering
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Flatten the labels array
    labels = labels.flatten()

    # Disable cluster number 2 and turn its pixels into black
    masked_image = np.copy(image)
    masked_image = masked_image.reshape((-1, 3))
    cluster_to_disable = 2
    masked_image[labels == cluster_to_disable] = [0, 0, 0]
    masked_image = masked_image.reshape(image.shape)
  # Compute the histogram of the clustered image
    hist = cv2.calcHist([masked_image], [0], None, [256], [0, 256])
    # Show the segmented image with cluster number 2 disabled
    plt.imshow(masked_image)
    plt.axis('off')
    st.pyplot()
    plt.savefig("kmean_clustering_k2.png")
     # Display the histogram
    plt.figure()
    plt.plot(hist)
    plt.title('Histogram of kmean_clustering_k2')
    plt.savefig("histogram_kmean_clustering_k2.png")
    st.pyplot()
    st.success("Kmean_Clustering_K2 Image Saved Successfully!")


def feature_extraction_page():
    st.title("Feature Extraction")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Perform Canny edge detection
    edges = feature.canny(gray_image, sigma=1)

    # Display the Canny edges image
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edges")
    plt.axis('off')
    st.pyplot()
    # Compute and display the histogram of the grayscale image
    plt.subplot(122)
    hist, _ = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))
    plt.plot(hist, color='b')
    plt.xlim([0, 256])
    plt.title("Histogram of Feature Extraction")

    plt.savefig("feature_extration.png")
    st.pyplot()
    st.success("Feature_Extraction Image Saved Successfully!")


def image_transformation_page():
    st.title("Image Transformation")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to RGB (assuming it's already in RGB format)
    pic = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(pic)
    plt.axis('off')
    st.pyplot()
    # Compute and display the histogram of the image
    plt.figure(figsize=(6, 2))
    hist, _ = np.histogram(pic.ravel(), bins=256, range=(0, 256))
    plt.plot(hist, color='b')
    plt.xlim([0, 256])
    plt.title('Histogram of Image Transformation')

    plt.savefig("histogram_image_transformation.png")
    st.pyplot()
    plt.savefig("image_transformation.png")
    st.success("Image_Transformation Image Saved Successfully!")


def color_negation_page():
    st.title("Color Negation")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to RGB (assuming it's already in RGB format)
    pic = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2RGB)

    # Perform color negation
    negative = 255 - pic

    # Display the negated image
    plt.figure(figsize=(6, 6))
    plt.imshow(negative)
    plt.axis('off')
    plt.savefig("color _negation.png")
    st.pyplot()
     # Convert the negated image to grayscale
    gray_negative = cv2.cvtColor(negative, cv2.COLOR_RGB2GRAY)

    # Compute and display the histogram of the grayscale negated image
    plt.figure(figsize=(6, 2))
    hist, _ = np.histogram(gray_negative.ravel(), bins=256, range=(0, 256))
    plt.plot(hist, color='b')
    plt.xlim([0, 256])
    plt.title('Histogram of Color Negation ')
    plt.savefig("histogram_color_negation.png")
    st.pyplot()
    st.success("Color_Negation Image Saved Successfully!")


def log_transform_page():
    st.title("Log Transformation")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Calculate the maximum pixel value
    max_pixel_value = np.max(gray_image)

    # Perform log transformation
    c = 255 / np.log(1 + max_pixel_value)
    transformed_image = c * np.log(1 + gray_image)

    # Display the transformed image
    plt.figure(figsize=(5, 5))
    plt.imshow(transformed_image, cmap='gray')
    plt.axis('off')
    plt.savefig("log_transfrom.png")
    st.pyplot()
     # Compute and display the histogram of the transformed image
    plt.figure(figsize=(5, 2))
    hist, _ = np.histogram(transformed_image.ravel(), bins=256, range=(0, 256))
    plt.plot(hist, color='b')
    plt.xlim([0, 256])
    plt.title('Histogram of Log_Transformation')
    plt.savefig("histogram_log_transform.png")
    st.pyplot()
    st.success("Log_Transform Image Saved Successfully!")

def gamma_correction_page():
    st.title("Gamma Correction")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Apply gamma correction
    gamma = 2.2  # Set the value of gamma (Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright)
    gamma_correction = ((fingerprint_image / 255) ** (1 / gamma))

     # Display the gamma-corrected image
    plt.figure(figsize=(5, 5))
    plt.imshow(gamma_correction)
    plt.axis('off')
    st.pyplot()
     # Calculate and display the histogram of the corrected image
    corrected_gray = cv2.cvtColor((gamma_correction * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([corrected_gray], [0], None, [256], [0, 256])
    if hist is not None:
        plt.figure(figsize=(5, 3))
        plt.plot(hist, color='blue')
        plt.title("Histogram of Gamma Correction")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        st.pyplot()
        plt.savefig("gamma_correction.png")
        st.success(" Gamma_correction Image Saved Successfully!")
    

def hsv_page():
    st.title("HSV Color Space")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2HSV)

    # Display the HSV image
    plt.imshow(hsv_image)
    plt.axis('off')
    st.pyplot()
     # Calculate and display the histogram of the HSV image
    hist_hue = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    if hist_hue is not None:
        plt.figure(figsize=(5, 3))
        plt.plot(hist_hue, color='b')
        plt.title("HSV Histogram")
        plt.xlabel("Hue Value")
        plt.ylabel("Frequency")
        plt.savefig("hsv.png")
        st.pyplot()
        st.success(" HSV Image Saved Successfully!")

def lab_page():
    st.title("LAB Color Space")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2LAB)

    # Display the LAB image
    plt.imshow(lab_image)
    plt.axis('off')
    st.pyplot()
    # Extract the LAB channels
    l_channel = lab_image[:, :, 0]
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    # Compute histograms for each LAB channel
    hist_l, _ = np.histogram(l_channel.ravel(), bins=256, range=(0, 256))
    hist_a, _ = np.histogram(a_channel.ravel(), bins=256, range=(0, 256))
    hist_b, _ = np.histogram(b_channel.ravel(), bins=256, range=(0, 256))

    # Display histograms for each LAB channel
    plt.figure(figsize=(15, 4))

    plt.plot(hist_l, color='r')
    plt.xlim([0, 256])
    plt.title('L Channel Histogram')
    st.pyplot()

    plt.figure(figsize=(15, 4))

    plt.plot(hist_a, color='g')
    plt.xlim([0, 256])
    plt.title('A Channel Histogram')
    st.pyplot()

    plt.figure(figsize=(15, 4))

    plt.plot(hist_b, color='b')
    plt.xlim([0, 256])
    plt.title('B Channel Histogram')
    st.pyplot()
    plt.savefig("lab.png")
    st.success(" LAB Image Saved Successfully!")

def ycrcb_page():
    st.title("YCrCb Color Space")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to YCrCb color space
    ycrcb_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2YCrCb)

    # Display the YCrCb image
    plt.imshow(ycrcb_image)
    plt.axis('off')
    st.pyplot()
    # Extract the YCrCb channels
    y_channel = ycrcb_image[:, :, 0]
    cr_channel = ycrcb_image[:, :, 1]
    cb_channel = ycrcb_image[:, :, 2]

    # Compute histograms for each YCrCb channel
    hist_y, _ = np.histogram(y_channel.ravel(), bins=256, range=(0, 256))
    hist_cr, _ = np.histogram(cr_channel.ravel(), bins=256, range=(0, 256))
    hist_cb, _ = np.histogram(cb_channel.ravel(), bins=256, range=(0, 256))

    # Display histograms
    plt.figure(figsize=(15, 4))

    plt.plot(hist_y, color='r')
    plt.xlim([0, 256])
    plt.title('Y Channel Histogram')
    st.pyplot()

    plt.figure(figsize=(15, 4))

    plt.plot(hist_cr, color='g')
    plt.xlim([0, 256])
    plt.title('Cr Channel Histogram')
    st.pyplot()

    plt.figure(figsize=(15, 4))

    plt.plot(hist_cb, color='b')
    plt.xlim([0, 256])
    plt.title('Cb Channel Histogram')
    st.pyplot()
    plt.savefig("ycrcb.png")
    
    st.success(" Ycrcb Image Saved Successfully!")


def shape_analysis_page():
    st.title("Shape Analysis")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)
     # Apply histogram equalization to enhance contrast
    equalized_image = cv2.equalizeHist(gray_image)


    # Apply binary thresholding
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     # Create a list to store shape areas
    shape_areas = []

    # Display the number of detected shapes
    st.write(f"Number of Shapes Detected: {len(contours)}")

    # Display information about each detected shape
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        moments = cv2.moments(contour)

        # Add the shape area to the list
        shape_areas.append(area)

        st.write(f"Shape {i + 1}")
        st.write(f"Area: {area:.2f}")
        st.write(f"Perimeter: {perimeter:.2f}")
        st.write(f"Moments: {moments}")

        # Draw the contour on a copy of the original image
        shape_image = fingerprint_image.copy()
        cv2.drawContours(shape_image, [contour], -1, (0, 255, 0), 3)

        # Display the image with the contour
        plt.figure(figsize=(6, 6))
        plt.imshow(shape_image)
        plt.axis('off')
        plt.title(f"Shape {i + 1}")
        plt.savefig(f"shape_{i + 1}.png")
        st.pyplot()
          # Calculate and display the histogram of shape areas
    plt.figure(figsize=(8, 4))
    plt.hist(shape_areas, bins=20, color='blue', alpha=0.7)
    plt.xlabel("Shape Area")
    plt.ylabel("Frequency")
    plt.title("Histogram of Shape Areas")
    st.pyplot()
    
    st.success(" Shape_Analysis Images and Information Saved Successfully!")
       
    
    
def glcm_page():
    st.title("Gray-Level Co-occurrence Matrix (GLCM)")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization to enhance contrast
    equalized_image = cv2.equalizeHist(gray_image)


    # Calculate GLCM with specified parameters
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Compute GLCM properties
    contrast = graycoprops(glcm, prop='contrast')
    dissimilarity = graycoprops(glcm, prop='dissimilarity')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    energy = graycoprops(glcm, prop='energy')
    correlation = graycoprops(glcm, prop='correlation')

    # Display computed GLCM properties
    st.write("GLCM Properties:")
    st.write(f"Contrast: {contrast}")
    st.write(f"Dissimilarity: {dissimilarity}")
    st.write(f"Homogeneity: {homogeneity}")
    st.write(f"Energy: {energy}")
    st.write(f"Correlation: {correlation}")
   

    # Display the original image
    plt.figure(figsize=(6, 6))
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.title("Original Gray Image")
    plt.savefig("original_gray_image.png")
    st.pyplot()
    
    st.success("GLCM Properties and Image Saved Successfully!")
   

def prewitt_sharpening_page():
    st.title("Prewitt Sharpening")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Apply Prewitt sharpening
    kernel_x = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])
    
    kernel_y = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])

    sharpened_image_x = cv2.filter2D(gray_image, -1, kernel_x)
    sharpened_image_y = cv2.filter2D(gray_image, -1, kernel_y)
    sharpened_image = cv2.addWeighted(sharpened_image_x, 0.5, sharpened_image_y, 0.5, 0)

    # Display the sharpened image
    plt.imshow(sharpened_image, cmap='gray')
    plt.axis('off')
    plt.savefig("prewitt_sharpening.png")
    st.pyplot()
    st.success("Prewitt_Sharpening Image Saved Successfully!")
     # Calculate and display a histogram of pixel values in the sharpened image
    hist_sharpened = cv2.calcHist([sharpened_image], [0], None, [256], [0, 256])
    hist_sharpened = hist_sharpened.squeeze()
    plt.figure()
    plt.hist(sharpened_image.ravel(), bins=256, range=[0, 256])
    plt.title("Histogram of Prewitt Sharpened Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    st.pyplot()
    

import cv2
import numpy as np
import matplotlib.pyplot as plt

def scharr_sharpening_page():
    st.title("Scharr Sharpening")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Apply Scharr sharpening
    gradient_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)

    # Convert gradients to uint8 data type
    gradient_x = cv2.convertScaleAbs(gradient_x)
    gradient_y = cv2.convertScaleAbs(gradient_y)

    sharpened_image = cv2.addWeighted(gray_image, 1, gradient_x, 0.5, 0)
    sharpened_image = cv2.addWeighted(sharpened_image, 1, gradient_y, 0.5, 0)

    # Display the sharpened image
    st.image(sharpened_image, caption="Scharr Sharpening", use_column_width=True)
    st.success("Scharr Sharpening Applied Successfully!")


 # Plot histogram for the sharpened image
    hist_data = sharpened_image.flatten()
    plt.hist(hist_data, bins=256, range=(0, 256), density=True, color='b', alpha=0.7)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of  Scharr Sharpened Image')
    st.pyplot()


def sobel_sharpening_page():
    st.title("Sobel Sharpening")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel sharpening
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Convert gradients to uint8 data type
    gradient_x = cv2.convertScaleAbs(gradient_x)
    gradient_y = cv2.convertScaleAbs(gradient_y)

    sharpened_image = cv2.addWeighted(gray_image, 1, gradient_x, 0.5, 0)
    sharpened_image = cv2.addWeighted(sharpened_image, 1, gradient_y, 0.5, 0)

    # Display the sharpened image
    st.image(sharpened_image, caption="Sobel Sharpening", use_column_width=True)
    st.success("Sobel Sharpening Applied Successfully!")
     # Plot histogram for the sharpened image
    hist_data = sharpened_image.flatten()
    plt.hist(hist_data, bins=256, range=(0, 256), density=True, color='b', alpha=0.7)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of sobel Sharpened Image')
    st.pyplot()

def gaussian_high_pass_filter_page():
    st.title("Gaussian High Pass Filter")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Calculate the high pass filter by subtracting the blurred image from the original image
    high_pass_image = gray_image - blurred_image

    # Display the high pass filtered image
    st.image(high_pass_image, caption="Gaussian High Pass Filter", use_column_width=True)
    st.success("Gaussian High Pass Filter Applied Successfully!")
     # Calculate and display the histogram of the high-pass filtered image
    hist = cv2.calcHist([high_pass_image], [0], None, [256], [0, 256])
    hist = hist.squeeze()
    
    plt.figure(figsize=(8, 5))
    plt.title("Histogram of High Pass Filtered Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    st.pyplot(plt)

def gaussian_low_pass_filter_page():
    st.title("Gaussian Low Pass Filter")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    ksize = st.slider("Kernel Size", min_value=1, max_value=15, step=2, value=3)
    blurred_image = cv2.GaussianBlur(gray_image, (ksize, ksize), 0)

    # Display the original and blurred images side by side
    col1, col2 = st.columns(2)
    col1.header("Original Image")
    col1.image(gray_image, use_column_width=True)

    col2.header("Blurred Image")
    col2.image(blurred_image, use_column_width=True)
     # Compute and display a histogram for the blurred image
    hist_blurred, _ = np.histogram(blurred_image.flatten(), 256, [0, 256])
    plt.figure()
    plt.plot(hist_blurred, color='b')
    plt.xlim([0, 256])
    plt.title('Histogram of Blurred Image')
    st.pyplot()


    st.success("Gaussian Low Pass Filter Applied Successfully!")
    

def sift_page():
    st.title("SIFT Feature Extraction")

    if 'fingerprint_image' not in st.session_state:
        st.warning("Please upload a fingerprint image on the 'Home' page.")
        return

    fingerprint_image = st.session_state.fingerprint_image
    gray_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # Draw keypoints on the image
    sift_image = cv2.drawKeypoints(gray_image, keypoints, None)

    # Display the image with SIFT keypoints
    plt.imshow(sift_image, cmap='gray')
    plt.axis('off')
    plt.title("SIFT Features")
    plt.savefig("sift_features.png")
    st.pyplot()
     # Compute and display a histogram of the descriptor values
    plt.figure()
    hist, bins = np.histogram(descriptors.ravel(), bins=256, range=(0, 256))
    plt.plot(hist, color='b')
    plt.xlim([0, 256])
    plt.title('Descriptor Histogram')
    plt.savefig("descriptor_histogram.png")
    st.pyplot()

    st.success("SIFT Features Image Saved Successfully!")

    


    # Disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)
    


def main():
    pages = ["Home", "Display Uploaded Image", "LBP Page", "LTP_Upper Page", "LTP_Lower Page",
             "RGB Splitter", "Histogram Equalization", "Grayscale page", "Resizing",
             "K-Mean Clustering", "K-Mean Clustering (K=2)", "Feature Extraction",
             "Image Transformation", "Color Negation", "Log Transformation",
             "Gamma Correction", "HSV Color Space", "LAB Color Space", "YCrCb Color Space",
             "Shape Analysis", "GLCM", "Prewitt Sharpening",
             "Scharr Sharpening", "Sobel Sharpening", "Gaussian High Pass Filter","Gaussian Low Pass Filter", "SIFT Feature Extraction"]

    selected_page = st.sidebar.radio("", pages)  # Initialize selected_page here

    if selected_page == "Home":
        home_page()
    elif selected_page == "Display Uploaded Image":
        display_uploaded_image_page()
    elif selected_page == "LBP Page":
        lbp_page()
    elif selected_page == "LTP_Upper Page":
        ltp_upper_page()
    elif selected_page == "LTP_Lower Page":
        ltp_lower_page()
    elif selected_page == "RGB Splitter":
        rgb_splitter_page()
    elif selected_page == "Histogram Equalization":
        histogram_equalization_page()
    elif selected_page == "Grayscale page":
        grayscale_page()
    elif selected_page == "Resizing":
        resizing_page()
    elif selected_page == "K-Mean Clustering":
        kmean_clustering_page()
    elif selected_page == "K-Mean Clustering (K=2)":
        kmean_clustering_k2_page()
    elif selected_page == "Feature Extraction":
        feature_extraction_page()
    elif selected_page == "Image Transformation":
        image_transformation_page()
    elif selected_page == "Color Negation":
        color_negation_page()
    elif selected_page == "Log Transformation":
        log_transform_page()
    elif selected_page == "Gamma Correction":
        gamma_correction_page()
    elif selected_page == "HSV Color Space":
        hsv_page()
    elif selected_page == "LAB Color Space":
        lab_page()
    elif selected_page == "YCrCb Color Space":
        ycrcb_page()
    elif selected_page == "Shape Analysis":
        shape_analysis_page()
    elif selected_page == "GLCM":
        glcm_page()
    elif selected_page == "Prewitt Sharpening":
        prewitt_sharpening_page()
    elif selected_page == "Scharr Sharpening":
        scharr_sharpening_page()
    elif selected_page == "Sobel Sharpening":
        sobel_sharpening_page()
    elif selected_page == "Gaussian High Pass Filter":
         gaussian_high_pass_filter_page()
    elif selected_page == "Gaussian Low Pass Filter":
         gaussian_low_pass_filter_page()
    elif selected_page == "SIFT Feature Extraction":
        sift_page()
    
if __name__ == "__main__":
    main()





