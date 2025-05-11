import os
import cv2
import pytesseract
import glob
import re
from datetime import datetime
import sys
import numpy as np

def extract_text_from_plates():
    """
    Extract text from license plate images by converting them to grayscale.
    The function processes images from the captured_plates directory and
    sends the extracted text to data.py.
    
    Returns:
        list: List of dictionaries containing plate information
    """
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plates_dir = r"E:\My Files\Elytra Solutions\MadeshPradesh\nmp\datapreprocessing\captured_plates"
    # Check if directory exists
    if not os.path.exists(plates_dir):
        print(f"Error: Directory not found: {plates_dir}")
        return []
    
    # Get all jpg files in the plates directory
    plate_images = glob.glob(os.path.join(plates_dir, "*.jpg"))
    
    if not plate_images:
        print(f"No license plate images found in {plates_dir}")
        return []
    
    print(f"Found {len(plate_images)} license plate images to process")
    
    # Configure pytesseract path (for Windows)
    # Set the path to Tesseract executable
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Regular expression pattern for license plate numbers (adjust as needed)
    # This pattern is for Indian license plates
    plate_pattern = re.compile(r'[A-Z]{2}\s*[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{4}')
    
    results = []
    
    for i, image_path in enumerate(plate_images):
        filename = os.path.basename(image_path)
        print(f"Processing {filename}...")
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                continue
            
            # Resize image for better OCR (if too small)
            height, width = img.shape[:2]
            if width < 300:
                scale_factor = 300 / width
                img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Try multiple preprocessing techniques and combine results
            plate_text = ""
            confidence = 0
            
            # Method 1: Basic preprocessing
            # Apply bilateral filter to reduce noise while keeping edges sharp
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply adaptive thresholding
            thresh1 = cv2.adaptiveThreshold(
                bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 19, 9
            )
            
            # Method 2: Alternative preprocessing
            # Apply Gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Otsu's thresholding
            _, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 3: Edge enhancement
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Dilate edges to connect broken characters
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Try different OCR configurations on each preprocessed image
            ocr_configs = [
                '--oem 1 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # License plate mode
                '--oem 1 --psm 8 -l eng',  # Single word mode
                '--oem 1 --psm 6 -l eng'   # Sparse text mode
            ]
            
            # Process each preprocessing method with each OCR config
            all_texts = []
            
            for thresh in [thresh1, thresh2, dilated_edges]:
                for config in ocr_configs:
                    text = pytesseract.image_to_string(thresh, config=config)
                    text = text.strip().replace('\n', ' ').replace('\r', '')
                    
                    # Remove non-alphanumeric characters except spaces
                    text = re.sub(r'[^A-Z0-9 ]', '', text.upper())
                    
                    if text:
                        all_texts.append(text)
            
            # Try to match the license plate pattern in all extracted texts
            for text in all_texts:
                match = plate_pattern.search(text)
                if match:
                    plate_text = match.group(0).replace(' ', '')
                    break
            
            # If no pattern match found, use the longest text
            if not plate_text and all_texts:
                plate_text = max(all_texts, key=len)
            
            # Extract timestamp and vehicle type from filename
            # Example filename: plate_car_0.95_20250511_015317_648163.jpg
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vehicle_type = "unknown"
            
            parts = filename.split('_')
            if len(parts) >= 3 and parts[0] == "plate":
                vehicle_type = parts[1]  # car, bus, truck, etc.
                
                if len(parts) >= 6:
                    date_part = parts[3]  # 20250511
                    time_part = parts[4]  # 015317
                    
                    # Parse date and time
                    year = date_part[0:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    
                    hour = time_part[0:2]
                    minute = time_part[2:4]
                    second = time_part[4:6]
                    
                    # Format as readable timestamp
                    timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
            
            # Add to results
            plate_info = {
                "sn": i + 1,
                "vehicle_proprietor": vehicle_type,
                "exacttime": timestamp,
                "number_plate": plate_text,
                "image_path": image_path
            }
            
            results.append(plate_info)
            print(f"Extracted: {plate_text}")
            
            # Save debug images (optional)
            debug_dir = os.path.join(os.path.dirname(plates_dir), "debug")
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
                
            cv2.imwrite(os.path.join(debug_dir, f"thresh1_{filename}"), thresh1)
            cv2.imwrite(os.path.join(debug_dir, f"thresh2_{filename}"), thresh2)
            cv2.imwrite(os.path.join(debug_dir, f"edges_{filename}"), dilated_edges)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Send results to data.py
    send_to_data_module(results)
    
    return results

def send_to_data_module(plate_data):
    """
    Send the extracted plate data to the data.py module.
    
    Args:
        plate_data (list): List of dictionaries containing plate information
    """
    try:
        # Import the data module
        detector_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(detector_dir, "data.py")
        
        # Check if data.py exists
        if not os.path.exists(data_path):
            print(f"Warning: data.py not found at {data_path}")
            print("Creating a basic data.py file...")
            
            # Create a basic data.py file
            with open(data_path, 'w') as f:
                f.write("""
# License plate data module
import os
import csv
from datetime import datetime

plate_data = []

def store_plate_data(data):
    \"\"\"
    Store license plate data and save to CSV.
    
    Args:
        data (list): List of dictionaries containing plate information
    \"\"\"
    global plate_data
    plate_data = data
    print(f"Stored {len(data)} license plate records")
    
    # Save to CSV file
    save_to_csv(data)
    
    return True

def get_plate_data():
    \"\"\"
    Get the stored license plate data.
    
    Returns:
        list: List of dictionaries containing plate information
    \"\"\"
    return plate_data

def save_to_csv(data):
    \"\"\"
    Save the plate data to a CSV file.
    
    Args:
        data (list): List of dictionaries containing plate information
    \"\"\"
    if not data:
        print("No data to save")
        return None
    
    try:
        # Get the database directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        database_dir = os.path.join(base_dir, "database")
        
        # Create directory if it doesn't exist
        if not os.path.exists(database_dir):
            os.makedirs(database_dir)
        
        # Generate filename with today's date
        today = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"license_plates_{today}.csv"
        csv_path = os.path.join(database_dir, csv_filename)
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ["SN", "vehicle_proprietor", "exacttime", "number_plate"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in data:
                # Convert keys to match the fieldnames
                row = {
                    "SN": item["sn"],
                    "vehicle_proprietor": item["vehicle_proprietor"],
                    "exacttime": item["exacttime"],
                    "number_plate": item["number_plate"]
                }
                writer.writerow(row)
        
        print(f"Saved {len(data)} records to {csv_path}")
        return csv_path
    
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return None
""")
        
        # Add the detector directory to the Python path
        if detector_dir not in sys.path:
            sys.path.append(detector_dir)
        
        # Import the data module
        import importlib
        data_module = importlib.import_module("data")
        
        # Reload the module in case it was modified
        importlib.reload(data_module)
        
        # Store the data
        if hasattr(data_module, 'store_plate_data'):
            data_module.store_plate_data(plate_data)
        else:
            print("Warning: store_plate_data function not found in data.py")
            
    except Exception as e:
        print(f"Error sending data to data.py: {e}")

if __name__ == "__main__":
    print("License Plate Text Extractor")
    plate_data = extract_text_from_plates()
    print(f"Processed {len(plate_data)} license plates")
    print(plate_data)