# License plate data module
import os
import csv
from datetime import datetime

# Global variable to store plate data
plate_data = []

def store_plate_data(data):
    """
    Store license plate data and save to CSV.
    
    Args:
        data (list): List of dictionaries containing plate information
    """
    global plate_data
    plate_data = data
    print(f"Stored {len(data)} license plate records")
    
    # Save to CSV file
    save_to_csv(data)
    
    return True

def get_plate_data():
    """
    Get the stored license plate data.
    
    Returns:
        list: List of dictionaries containing plate information
    """
    return plate_data

def save_to_csv(data):
    """
    Save the plate data to a CSV file.
    
    Args:
        data (list): List of dictionaries containing plate information
    """
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

def analyze_data():
    """
    Analyze the plate data to generate statistics.
    
    Returns:
        dict: Statistics about the plate data
    """
    if not plate_data:
        return {"error": "No data available"}
    
    # Count vehicles by type
    vehicle_counts = {}
    for item in plate_data:
        vehicle_type = item["vehicle_proprietor"]
        if vehicle_type in vehicle_counts:
            vehicle_counts[vehicle_type] += 1
        else:
            vehicle_counts[vehicle_type] = 1
    
    return {
        "total_plates": len(plate_data),
        "vehicle_types": vehicle_counts
    }