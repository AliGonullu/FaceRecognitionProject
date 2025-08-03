import cv2
import DetectFaces
from deepface import DeepFace
import ImageGrid
from PyQt5.QtWidgets import QApplication
import sys

CONFIDENCE_THRESHOLD = 0.5

def display_images_in_grid(images, window_name="Face Recognition Results"):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = ImageGrid.ScrollableImageGrid(images=images, window_name=window_name)
    window.show()
    app.exec_()


def CompareImage(selected_image_idx):
    Image_Paths = DetectFaces.get_all_image_paths()

    selected_image_path = Image_Paths[selected_image_idx]
    result_images = []
    distances = []
    match_checks = []
    max_dist = 1

    for i in range(len(Image_Paths)):
        test_image_path = Image_Paths[i]
        result = DeepFace.verify(img1_path=selected_image_path, img2_path=test_image_path, model_name="ArcFace")
        distances.append(result["distance"])
        match_checks.append(result["verified"])
    
    max_dist = max(distances)
    
    for i in range(len(Image_Paths)):
        test_image_path = Image_Paths[i]
        test_image = cv2.imread(test_image_path)
        text = ""
        if selected_image_idx != i:
            try:
                similarity_percentage = max(0, abs(max_dist - distances[i]) * 100)
                text = f"Similarity: {similarity_percentage:.2f}%"
                text_color = (0, 255, 0) if match_checks[i] and i != selected_image_idx else (0, 0, 255)
            except ValueError as e:
                print(e)
                similarity_percentage = 0
        else:
            text = f"| SELECTED IMAGE |"
            text_color = (255, 255, 255)

        DetectFaces.detectFaceOnImage(test_image_path, test_image, CONFIDENCE_THRESHOLD)
        DetectFaces.putText_with_outline(test_image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, text_color, thickness=2, outline_thickness=6)
        result_images.append(test_image)

    display_images_in_grid(images=result_images)