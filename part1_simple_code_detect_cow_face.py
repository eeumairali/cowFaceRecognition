import cv2
import os
from ultralytics import YOLO

class YOLOProcessor:
    def __init__(self, model_path):
        """
        Initialize the YOLO model.
        :param model_path: Path to the trained YOLO model.
        """
        self.model = YOLO(model_path)

    def process_image(self, image_path, output_folder,file_out_name):
        """
        Process an image with YOLO and save the result.
        """
        os.makedirs(output_folder, exist_ok=True)
        results = self.model(image_path)  # Run YOLO detection
          # Define the output file name
        for r in results:
            r.save(filename=os.path.join(output_folder, "{file_out_name}.png"))  # Save the result
        # Optionally, you can also display the image using OpenCV
       
        print(f"Processed image saved at {output_folder}/{file_out_name}.png")
   
# Example Usage
if __name__ == "__main__":
    model_path = "/mnt/c/Users/eeuma/Desktop/CowFacialIdentification/fined_tuned_detection_model/yolov8_fine_tuned_cow_face.pt"
    processor = YOLOProcessor(model_path)

    # Process an image
    output_path = "/mnt/c/Users/eeuma/Desktop/CowFacialIdentification/demo_images/outputs"
    file_out_name = 'part2_demo_out'
    image_path = "/mnt/c/Users/eeuma/Desktop/CowFacialIdentification/demo_images/inputs/two_cows_brown.png"
    processor.process_image(image_path, output_path)
