from vehicle_detect import VehicleDetector

def main():
    # Create vehicle detector instance
    detector = VehicleDetector(model_path="yolov8n.pt", camera_id=1)
    
    # Run the detection loop
    detector.run_detection()

if __name__ == "__main__":
    main()