import cv2
import mediapipe as mp
import numpy as np

# Path to the uploaded image
def recog(image_path):
    image_path = r'F:\intern prep\myntra hack\face dataset\model\image4.jpg'

    # Load the image
    image = cv2.imread(image_path)

    # Check if the image is loaded properly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        exit()

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to find face landmarks
    results = face_mesh.process(rgb_image)

    def get_average_color(image, landmarks, indices):
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), 255)
        mean = cv2.mean(image, mask=mask)[:3]
        return '#{:02x}{:02x}{:02x}'.format(int(mean[2]), int(mean[1]), int(mean[0]))

    # Define regions of interest for skin and eyes
    regions = {
        "left_cheek": [234, 93, 132, 58, 172, 136, 150],
        "right_cheek": [454, 323, 361, 288, 412, 377, 391],
        "forehead": [9, 107, 336, 297],
        "left_eye": [33, 246, 161, 160, 159, 158, 157, 173],
        "right_eye": [263, 466, 388, 387, 386, 385, 384, 398]
    }

    colors = {}
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            for region_name, region_points in regions.items():
                color = get_average_color(image, landmarks, region_points)
                colors[region_name] = color

            return (colors)

    # Display the image with landmarks for visualization (optional)
    #if results.multi_face_landmarks:
    #    for face_landmarks in results.multi_face_landmarks:
    #        mp.solutions.drawing_utils.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    #cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
