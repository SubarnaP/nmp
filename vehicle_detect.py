import cv2
from ultralytics import YOLO
from capture import VehicleCapture
import threading
# Fix the import to get the NumberPlatePreprocessor class
from detector.plate_detector import NumberPlatePreprocessor

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", camera_id=1, capture_dir="captured_vehicles", plate_dir="captured_plates"):
        """
        Initialize the vehicle detector with a YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model file
            camera_id (int): Camera ID to use (default is 1)
            capture_dir (str): Directory to save captured vehicle images
            plate_dir (str): Directory to save cropped license plate images
        """
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.camera_id = camera_id
        self.cap = None
        
        # List of vehicle classes to detect
        self.vehicle_classes = ["car", "bus", "motorbike", "truck", "bicycle"]
        
        # Initialize vehicle capture system
        self.vehicle_capture = VehicleCapture(save_dir=capture_dir)
        
        # Initialize license plate preprocessor
        self.plate_preprocessor = NumberPlatePreprocessor(
            source_dir=capture_dir,
            target_dir=plate_dir
        )
        
        # Start plate processing in a separate thread
        self.start_plate_processing()
    
    def start_plate_processing(self):
        """Start license plate processing in a separate thread"""
        self.plate_thread = threading.Thread(
            target=self.plate_preprocessor.monitor_directory,
            args=(1.0,),  # Check every 1 second
            daemon=True   # Thread will exit when main program exits
        )
        self.plate_thread.start()
    
    def start_capture(self):
        """Start video capture from the specified camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        return self.cap.isOpened()
    
    def process_frame(self, frame):
        """
        Process a single frame to detect vehicles.
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            numpy.ndarray: Processed frame with vehicle detections
        """
        # Run YOLOv8 detection on the frame
        results = self.model(frame, stream=True)
        
        # Loop over results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]
                
                # Filter for vehicles only
                if label in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Try to capture the vehicle
                    captured = self.vehicle_capture.capture_vehicle(
                        frame, (x1, y1, x2, y2), label, conf
                    )
                    
                    # Add capture status to the label
                    status = "Captured" if captured else "Tracked"
                    cv2.putText(frame, f"{label} {conf:.2f} - {status}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2)
        
        return frame
    
    def run_detection(self):
        """
        Run the vehicle detection loop on video stream.
        
        Returns:
            bool: True if completed successfully, False otherwise
        """
        if not self.start_capture():
            print("Failed to open camera")
            return False
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process the frame
            processed_frame = self.process_frame(frame)
                
            # Show frame
            cv2.imshow("YOLOv8 Vehicle Detection", processed_frame)
                
            # Break with 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        # Clean up
        self.release()
        return True
    
    def release(self):
        """Release video capture and close windows"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()