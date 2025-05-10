import os
import cv2
import pytesseract
import glob
import re
from datetime import datetime
import sys

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
    plates_dir = os.path.join(base_dir, "datapreprocessing", "captured_plates")
    
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
    # Uncomment and set this if pytesseract is not in PATH
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Regular expression pattern for license plate numbers (adjust as needed)
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
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing to improve OCR accuracy
            # Bilateral filter to reduce noise while keeping edges sharp
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 19, 9
            )
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(
                thresh, 
                config='--oem 1 --psm 7 -l eng'
            )
            
            # Clean the extracted text
            text = text.strip().replace('\n', ' ').replace('\r', '')
            
            # Try to match the license plate pattern
            match = plate_pattern.search(text)
            if match:
                plate_text = match.group(0).replace(' ', '')
            else:
                plate_text = text
            
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
plate_data = []

def store_plate_data(data):
    \"\"\"
    Store license plate data.
    
    Args:
        data (list): List of dictionaries containing plate information
    \"\"\"
    global plate_data
    plate_data = data
    print(f"Stored {len(data)} license plate records")
    
    # You can add additional processing here, such as:
    # - Saving to a database
    # - Generating reports
    # - Triggering notifications
    
    return True

def get_plate_data():
    \"\"\"
    Get the stored license plate data.
    
    Returns:
        list: List of dictionaries containing plate information
    \"\"\"
    return plate_data
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