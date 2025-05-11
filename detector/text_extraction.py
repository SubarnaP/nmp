import os
import cv2
import pytesseract
import glob
import re
from datetime import datetime
import sys
import numpy as np
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def extract_text_from_plates(debug_mode=True):
    """
    Extract text from license plate images by converting them to grayscale.
    The function processes images from the captured_plates directory and
    sends the extracted text to data.py.
    
    Args:
        debug_mode (bool): Enable additional debugging output
    
    Returns:
        list: List of dictionaries containing plate information
    """
    # Define paths
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
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Verify Tesseract installation
    if not os.path.exists(tesseract_path):
        print(f"ERROR: Tesseract not found at {tesseract_path}")
        print("Please install Tesseract OCR or update the path in the code.")
        return []
    
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Test Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except Exception as e:
        print(f"Error accessing Tesseract: {e}")
        print("Please make sure Tesseract is properly installed.")
        return []
    
    # Regular expression pattern for license plate numbers (adjust as needed)
    # This pattern is for Indian license plates - made more flexible
    plate_pattern = re.compile(r'[A-Z]{1,2}\s*[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{3,4}')
    
    results = []
    
    # Create debug directory
    debug_dir = os.path.join(os.path.dirname(plates_dir), "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    for i, image_path in enumerate(plate_images):
        filename = os.path.basename(image_path)
        print(f"\nProcessing {filename}...")
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                continue
            
            # Save original image for debugging
            if debug_mode:
                cv2.imwrite(os.path.join(debug_dir, f"original_{filename}"), img)
            
            # Resize image for better OCR (if too small)
            height, width = img.shape[:2]
            if width < 300:
                scale_factor = 300 / width
                img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                print(f"Resized image from {width}x{height} to {int(width*scale_factor)}x{int(height*scale_factor)}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Save grayscale image for debugging
            if debug_mode:
                cv2.imwrite(os.path.join(debug_dir, f"gray_{filename}"), gray)
            
            # Try multiple preprocessing techniques and combine results
            plate_text = ""
            all_texts = []
            
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
            
            # Method 4: Simple binary threshold
            _, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Method 5: Contrast enhancement
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            _, thresh4 = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save all preprocessed images for debugging
            cv2.imwrite(os.path.join(debug_dir, f"thresh1_{filename}"), thresh1)
            cv2.imwrite(os.path.join(debug_dir, f"thresh2_{filename}"), thresh2)
            cv2.imwrite(os.path.join(debug_dir, f"edges_{filename}"), dilated_edges)
            cv2.imwrite(os.path.join(debug_dir, f"thresh3_{filename}"), thresh3)
            cv2.imwrite(os.path.join(debug_dir, f"thresh4_{filename}"), thresh4)
            
            # Try different OCR configurations on each preprocessed image
            ocr_configs = [
                '--oem 1 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # License plate mode
                '--oem 1 --psm 8 -l eng',  # Single word mode
                '--oem 1 --psm 6 -l eng',  # Sparse text mode
                '--oem 3 --psm 11 -l eng'  # Full page mode with neural net
            ]
            
            # Process each preprocessing method with each OCR config
            all_results = {}
            
            for idx, thresh in enumerate([thresh1, thresh2, dilated_edges, thresh3, thresh4]):
                method_name = f"Method {idx+1}"
                all_results[method_name] = {}
                
                for config_idx, config in enumerate(ocr_configs):
                    config_name = f"Config {config_idx+1}"
                    
                    try:
                        text = pytesseract.image_to_string(thresh, config=config)
                        text = text.strip().replace('\n', ' ').replace('\r', '')
                        
                        # Remove non-alphanumeric characters except spaces
                        text = re.sub(r'[^A-Z0-9 ]', '', text.upper())
                        
                        all_results[method_name][config_name] = text
                        
                        if text:
                            all_texts.append(text)
                            
                            # Check if it matches the license plate pattern
                            match = plate_pattern.search(text)
                            if match:
                                matched_text = match.group(0).replace(' ', '')
                                print(f"Found match in {method_name}, {config_name}: {matched_text}")
                                
                                # If we haven't found a plate text yet, or this one is better
                                if not plate_text or len(matched_text) > len(plate_text):
                                    plate_text = matched_text
                    
                    except Exception as e:
                        print(f"Error in OCR {method_name}, {config_name}: {e}")
            
            # Print all OCR results for debugging
            if debug_mode:
                print("\nAll OCR Results:")
                for method, configs in all_results.items():
                    print(f"\n{method}:")
                    for config, text in configs.items():
                        print(f"  {config}: '{text}'")
            
            # If no pattern match found, use the longest text
            if not plate_text and all_texts:
                plate_text = max(all_texts, key=len)
                print(f"No pattern match found. Using longest text: {plate_text}")
            
            # Extract timestamp and vehicle type from filename
            # Example filename: plate_car_0.95_20250511_015317_648163.jpg
            
            # Use current system time instead of parsing from filename
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Set vehicle_proprietor to "unknown" regardless of filename
            vehicle_type = "unknown"
            
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
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Send results to data.py
    if results:
        send_to_data_module(results)
    else:
        print("No license plate text was successfully extracted.")
    
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

# Global variable to store plate data
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
        
        # Check if file already exists for today
        if os.path.exists(csv_path):
            print(f"CSV file for today already exists at {csv_path}")
            # Append to existing file instead of creating a new one
            existing_data = []
            try:
                with open(csv_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        existing_data.append(row)
                
                # Get the highest SN value
                max_sn = 0
                for item in existing_data:
                    try:
                        sn = int(item["SN"])
                        if sn > max_sn:
                            max_sn = sn
                    except (ValueError, KeyError):
                        pass
                
                # Update SN values for new data
                for i, item in enumerate(data):
                    item["sn"] = max_sn + i + 1
                
                # Append to existing file
                with open(csv_path, 'a', newline='') as csvfile:
                    fieldnames = ["SN", "vehicle_proprietor", "exacttime", "number_plate"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    for item in data:
                        # Convert keys to match the fieldnames
                        row = {
                            "SN": item["sn"],
                            "vehicle_proprietor": item["vehicle_proprietor"],
                            "exacttime": item["exacttime"],
                            "number_plate": item["number_plate"]
                        }
                        writer.writerow(row)
                
                print(f"Appended {len(data)} records to existing file {csv_path}")
                return csv_path
                
            except Exception as e:
                print(f"Error reading existing CSV file: {e}")
                # If there's an error reading the existing file, we'll create a new one
        
        # Create a new file if it doesn't exist or couldn't be read
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
        
        print(f"Saved {len(data)} records to new file {csv_path}")
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

class PlateImageHandler(FileSystemEventHandler):
    """
    Watchdog handler for monitoring the captured_plates directory.
    Automatically processes new images as they arrive.
    """
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.last_processed_time = time.time()
        self.processing_lock = False
        
    def on_created(self, event):
        # Only process if it's a file and it's a jpg
        if not event.is_directory and event.src_path.lower().endswith('.jpg'):
            # Add a small delay to ensure file is completely written
            time.sleep(1)
            
            # Don't process too frequently (at least 2 seconds between processing)
            current_time = time.time()
            if current_time - self.last_processed_time < 2:
                return
                
            self.last_processed_time = current_time
            
            # Avoid processing while another process is running
            if self.processing_lock:
                print(f"Already processing images, skipping: {event.src_path}")
                return
                
            try:
                self.processing_lock = True
                print(f"New image detected: {event.src_path}")
                plate_data = extract_text_from_plates(debug_mode=self.debug_mode)
                print(f"Processed {len(plate_data)} license plates")
                
                if plate_data:
                    print("\nExtracted license plates:")
                    for item in plate_data:
                        print(f"  {item['number_plate']} ({item['vehicle_proprietor']})")
            finally:
                self.processing_lock = False

def start_watchdog(debug_mode=True):
    """
    Start the watchdog observer to monitor the captured_plates directory.
    
    Args:
        debug_mode (bool): Enable additional debugging output
    """
    plates_dir = r"E:\My Files\Elytra Solutions\MadeshPradesh\nmp\datapreprocessing\captured_plates"
    
    # Check if directory exists
    if not os.path.exists(plates_dir):
        print(f"Error: Directory not found: {plates_dir}")
        return
    
    # Create the event handler and observer
    event_handler = PlateImageHandler(debug_mode=debug_mode)
    observer = Observer()
    
    # Schedule the observer to watch the directory
    observer.schedule(event_handler, plates_dir, recursive=False)
    
    # Start the observer
    observer.start()
    print(f"Watching for new images in {plates_dir}")
    
    try:
        # Process any existing images first
        plate_data = extract_text_from_plates(debug_mode=debug_mode)
        print(f"Initially processed {len(plate_data)} license plates")
        
        # Keep the thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Watchdog stopped")
    
    observer.join()

if __name__ == "__main__":
    print("License Plate Text Extractor")
    print("============================")
    
    # Check if watchdog mode is enabled
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        print("Starting in watchdog mode...")
        start_watchdog(debug_mode=True)
    else:
        # Run in one-time processing mode
        plate_data = extract_text_from_plates(debug_mode=True)
        print(f"\nProcessed {len(plate_data)} license plates")
        
        if plate_data:
            print("\nExtracted license plates:")
            for item in plate_data:
                print(f"  {item['number_plate']} ({item['vehicle_proprietor']})")
        else:
            print("\nNo license plates were successfully extracted.")
            print("Please check the debug images and Tesseract installation.")
  
