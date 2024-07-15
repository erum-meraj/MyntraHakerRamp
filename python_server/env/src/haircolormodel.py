import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_face(image):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No face detected")
        return None
    
    return faces[0]  # Return the first detected face

def mask_hair(image, face_rect):
    (x, y, w, h) = face_rect
    
    # Estimate the hair region by taking the upper part of the face rectangle
    hair_rect = (x, y - int(h / 2), w, int(h / 2))
    (hx, hy, hw, hh) = hair_rect
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[max(hy, 0):hy+hh, hx:hx+hw] = 255
    
    return mask

def get_dominant_color(image, mask):
    # Mask the image to get only the hair region
    hair_region = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert the hair region to LAB color space for better clustering
    lab_image = cv2.cvtColor(hair_region, cv2.COLOR_BGR2LAB)
    
    # Reshape the image to a 2D array of pixels
    pixel_values = lab_image.reshape((-1, 3))
    
    # Remove black pixels (which were outside the mask)
    pixel_values = pixel_values[pixel_values[:, 0] != 0]
    
    # Apply k-means clustering to find the dominant color
    k = 3
    _, labels, centers = cv2.kmeans(np.float32(pixel_values), k, None, 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Find the most dominant cluster
    dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
    
    # Convert the LAB color back to RGB
    dominant_color_rgb = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_LAB2RGB)[0][0]
    
    # Convert the dominant color from RGB to HEX
    hex_color = '#%02x%02x%02x' % (dominant_color_rgb[0], dominant_color_rgb[1], dominant_color_rgb[2])
    
    return hex_color

def main(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Image not found or unable to read")
        return
    
    # Detect the face
    face_rect = detect_face(image)
    if face_rect is None:
        return
    
    # Mask the hair region
    hair_mask = mask_hair(image, face_rect)
    
    # Get the hair color
    hair_color_hex = get_dominant_color(image, hair_mask)
    
    print("Detected Hair Color (HEX):", hair_color_hex)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    
    # Original image
    #plt.subplot(1, 2, 1)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.title("Original Image")
    
    # Hair mask
    #hair_region = cv2.bitwise_and(image, image, mask=hair_mask)
    #plt.subplot(1, 2, 2)
    #plt.imshow(cv2.cvtColor(hair_region, cv2.COLOR_BGR2RGB))
    #plt.title("Detected Hair Region")
    
    #plt.show()

if __name__ == "__main__":
    image_path = r'F:\intern prep\myntra hack\face dataset\model\image4.jpg'

    main(image_path)
