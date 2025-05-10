import os
import cv2
import numpy as np
import time
from datetime import datetime
import glob
import urllib.request
import shutil


class NumberPlatePreprocessor:
    def __init__(self, source_dir="captured_vehicles", target_dir="captured_plates"):
        """
        Initialize the number plate preprocessor.
        """
        # Get the project root directory (one level up from datapreprocessing)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set absolute paths for source and target directories
        self.source_dir = os.path.join(base_dir, source_dir)
        self.target_dir = os.path.join(base_dir, "datapreprocessing", target_dir)
        
        print(f"Source directory: {self.source_dir}")
        print(f"Target directory: {self.target_dir}")
        
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        self.last_processed_image = None

        # Fix cascade path to be in the datapreprocessing directory
        cascade_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(cascade_dir, "haarcascade_russian_plate_number.xml")
        if not os.path.exists(cascade_path):
            self.download_cascade(cascade_path)

        self.plate_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Process all existing images in the source directory
        self.process_all_images()

    def download_cascade(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"

        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"Downloaded cascade file to {save_path}")
        except Exception as e:
            print(f"Failed to download cascade file: {e}")
            default_cascades = cv2.data.haarcascades
            fallback = os.path.join(default_cascades, "haarcascade_russian_plate_number.xml")
            if os.path.exists(fallback):
                shutil.copy(fallback, save_path)
                print(f"Copied default cascade to {save_path}")
            else:
                print("No default cascade available. License plate detection may not work.")

    def get_latest_image(self):
        try:
            files = glob.glob(os.path.join(self.source_dir, "*.jpg"))
            if not files:
                return None
            latest_file = max(files, key=os.path.getmtime)
            if latest_file == self.last_processed_image:
                return None
            self.last_processed_image = latest_file
            return latest_file
        except Exception as e:
            print(f"Error getting latest image: {e}")
            return None
    
    def get_all_images(self):
        """Get all images in the source directory"""
        try:
            return glob.glob(os.path.join(self.source_dir, "*.jpg"))
        except Exception as e:
            print(f"Error getting images: {e}")
            return []

    def detect_and_crop_plate(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                return False, None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Try with different parameters for better detection
            detection_params = [
                {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (25, 10)},
                {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (20, 8)},
                {"scaleFactor": 1.03, "minNeighbors": 2, "minSize": (15, 6)},
            ]
            
            for params in detection_params:
                plates = self.plate_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=params["scaleFactor"], 
                    minNeighbors=params["minNeighbors"], 
                    minSize=params["minSize"], 
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(plates) > 0:
                    break
            
            if len(plates) == 0:
                return self.detect_plate_by_contours(img)

            if len(plates) > 1:
                plates = [max(plates, key=lambda x: x[2] * x[3])]

            for (x, y, w, h) in plates:
                pad_x = int(w * 0.05)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img.shape[1], x + w + pad_x)
                y2 = min(img.shape[0], y + h + pad_y)
                plate_img = img[y1:y2, x1:x2]
                return True, plate_img

            return False, None
        except Exception as e:
            print(f"Error detecting plate: {e}")
            return False, None

    def detect_plate_by_contours(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(gray, 30, 200)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    if 1.5 <= aspect_ratio <= 5.0:
                        pad_x = int(w * 0.05)
                        pad_y = int(h * 0.1)
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(img.shape[1], x + w + pad_x)
                        y2 = min(img.shape[0], y + h + pad_y)
                        plate_img = img[y1:y2, x1:x2]
                        return True, plate_img

            return False, None
        except Exception as e:
            print(f"Error in contour detection: {e}")
            return False, None

    def save_plate_image(self, plate_img, original_image_path):
        try:
            original_filename = os.path.basename(original_image_path)
            plate_filename = f"plate_{original_filename}"
            plate_path = os.path.join(self.target_dir, plate_filename)
            cv2.imwrite(plate_path, plate_img)
            return plate_path
        except Exception as e:
            print(f"Error saving plate image: {e}")
            return None

    def process_latest_image(self):
        latest_image = self.get_latest_image()
        if not latest_image:
            return False
        success, plate_img = self.detect_and_crop_plate(latest_image)
        if not success or plate_img is None:
            print(f"No license plate detected in {latest_image}")
            return False
        plate_path = self.save_plate_image(plate_img, latest_image)
        if not plate_path:
            return False
        print(f"Saved license plate from {latest_image} to {plate_path}")
        return True
    
    def process_all_images(self):
        """Process all existing images in the source directory"""
        print(f"Processing all existing images in {self.source_dir}...")
        images = self.get_all_images()
        if not images:
            print("No images found in source directory.")
            return
            
        print(f"Found {len(images)} images to process.")
        for image_path in images:
            print(f"Processing {image_path}...")
            success, plate_img = self.detect_and_crop_plate(image_path)
            if not success or plate_img is None:
                print(f"No license plate detected in {image_path}")
                continue
            plate_path = self.save_plate_image(plate_img, image_path)
            if plate_path:
                print(f"Saved license plate from {image_path} to {plate_path}")
        
        print("Finished processing all existing images.")

    def monitor_directory(self, interval=1.0):
        print(f"Monitoring {self.source_dir} for new images...")
        try:
            while True:
                self.process_latest_image()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Monitoring stopped.")


if __name__ == "__main__":
    preprocessor = NumberPlatePreprocessor(
        source_dir="captured_vehicles",
        target_dir="captured_plates"
    )
    preprocessor.monitor_directory(interval=1.0)
