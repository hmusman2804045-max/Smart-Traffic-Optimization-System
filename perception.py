from ultralytics import YOLO
import cv2
import os
import random
import glob

class TrafficPerception:
    def __init__(self, model_variant='yolov8n.pt'):
        print(f" Initializing YOLOv8 ({model_variant})...")
        self.model = YOLO(model_variant)
        self.traffic_classes = [0, 2, 3, 5, 7]

    def process_frame(self, frame_path):
        if not os.path.exists(frame_path):
            return None, "Error: File not found."
        results = self.model(frame_path, classes=self.traffic_classes, verbose=False)
        result = results[0]
        counts = {
            "cars": 0,
            "trucks": 0,
            "buses": 0,
            "motorcycles": 0,
            "pedestrians": 0
        }
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 2: counts["cars"] += 1
            elif cls_id == 7: counts["trucks"] += 1
            elif cls_id == 5: counts["buses"] += 1
            elif cls_id == 3: counts["motorcycles"] += 1
            elif cls_id == 0: counts["pedestrians"] += 1
        density_score = counts["cars"] + (counts["trucks"] * 1.5) + (counts["buses"] * 1.5) + counts["pedestrians"]
        return result, counts, density_score

    def save_annotated_frame(self, result, output_path):
        annotated_frame = result.plot()
        cv2.imwrite(output_path, annotated_frame)
        print(f" Annotated frame saved to {output_path}")

if __name__ == "__main__":
    perception = TrafficPerception()
    test_images_dir = "bdd100k/bdd100k/images/100k/test"
    image_paths = glob.glob(os.path.join(test_images_dir, "*.jpg"))
    if image_paths:
        sample_img = random.choice(image_paths)
        print(f" Selected random test image: {sample_img}")
        res, counts, score = perception.process_frame(sample_img)
        print(f"\n --- Detection Results for {os.path.basename(sample_img)} ---")
        for key, value in counts.items():
            print(f" {key.capitalize()}: {value}")
        print(f" Total Density Score: {score}")
        perception.save_annotated_frame(res, "output_perception_test.jpg")
    else:
        print(f" No .jpg images found in {test_images_dir}. Please ensure extraction worked.")
