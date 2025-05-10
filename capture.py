import cv2
import os
import time
import numpy as np
from datetime import datetime

class VehicleCapture:
    def __init__(self, save_dir="captured_vehicles"):
        """
        Initialize the vehicle capture system.
        
        Args:
            save_dir (str): Directory to save captured vehicle images
        """
        self.save_dir = save_dir
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Dictionary to track captured vehicles
        # Key: vehicle signature, Value: timestamp of last capture
        self.captured_vehicles = {}
        
        # Minimum time between captures of the same vehicle (seconds)
        self.recapture_delay = 5
        
        # Similarity threshold for considering vehicles the same
        # Increasing this value to make matching more strict
        self.similarity_threshold = 0.92
        
        # Keep track of recent positions to avoid capturing stationary vehicles
        self.recent_positions = []
        self.position_history_size = 10
        self.position_similarity_threshold = 20  # pixels
        
    def _get_vehicle_signature(self, vehicle_img):
        """
        Generate a signature for the vehicle to identify it.
        
        Args:
            vehicle_img (numpy.ndarray): Cropped image of the vehicle
            
        Returns:
            numpy.ndarray: Resized and normalized image as signature
        """
        # Resize to a standard size for comparison
        resized = cv2.resize(vehicle_img, (64, 64))
        # Convert to grayscale for more robust comparison
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        # Apply histogram equalization for better feature matching
        equalized = cv2.equalizeHist(gray)
        # Normalize
        normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
        return normalized
    
    def _is_similar_position(self, bbox):
        """
        Check if the current bounding box is similar to recently seen positions.
        
        Args:
            bbox (tuple): Current bounding box (x1, y1, x2, y2)
            
        Returns:
            bool: True if similar position was recently seen
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        for pos in self.recent_positions:
            pos_x, pos_y = pos
            # Calculate Euclidean distance between centers
            distance = np.sqrt((center_x - pos_x)**2 + (center_y - pos_y)**2)
            if distance < self.position_similarity_threshold:
                return True
                
        # Add current position to history
        self.recent_positions.append((center_x, center_y))
        # Keep history at fixed size
        if len(self.recent_positions) > self.position_history_size:
            self.recent_positions.pop(0)
            
        return False
    
    def _is_same_vehicle(self, signature, bbox, existing_signatures):
        """
        Check if the vehicle has been captured recently.
        
        Args:
            signature (numpy.ndarray): Signature of current vehicle
            bbox (tuple): Bounding box of current vehicle
            existing_signatures (dict): Dictionary of existing vehicle signatures
            
        Returns:
            bool: True if the vehicle was recently captured, False otherwise
        """
        current_time = time.time()
        
        # First check if the vehicle is in a similar position as recently seen vehicles
        if self._is_similar_position(bbox):
            return True
            
        for existing_sig, timestamp in list(existing_signatures.items()):
            # Skip if enough time has passed since last capture
            if current_time - timestamp > self.recapture_delay:
                continue
                
            # Convert string key back to numpy array
            existing_sig_array = np.frombuffer(existing_sig, dtype=np.uint8).reshape(64, 64)
            
            # Compare signatures using multiple methods for better accuracy
            
            # 1. Normalized cross-correlation
            result = cv2.matchTemplate(signature, existing_sig_array, cv2.TM_CCORR_NORMED)
            similarity1 = result[0][0]
            
            # 2. Mean squared error (lower is more similar)
            mse = np.mean((signature.astype("float") - existing_sig_array.astype("float")) ** 2)
            mse_similarity = 1 - (mse / 255.0)  # Convert to similarity score (0-1)
            
            # 3. Structural similarity (higher is more similar)
            try:
                ssim = cv2.matchTemplate(signature, existing_sig_array, cv2.TM_CCOEFF_NORMED)[0][0]
            except:
                ssim = 0
                
            # Combine similarity scores (weighted average)
            combined_similarity = (0.5 * similarity1) + (0.3 * mse_similarity) + (0.2 * ssim)
            
            if combined_similarity > self.similarity_threshold:
                return True
                
        return False
    
    def capture_vehicle(self, frame, bbox, vehicle_class, confidence):
        """
        Capture an image of the detected vehicle if it hasn't been captured recently.
        
        Args:
            frame (numpy.ndarray): Full frame from the video
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
            vehicle_class (str): Class of the detected vehicle
            confidence (float): Detection confidence
            
        Returns:
            bool: True if a new vehicle was captured, False otherwise
        """
        x1, y1, x2, y2 = bbox
        
        # Skip small detections (likely false positives)
        width = x2 - x1
        height = y2 - y1
        if width < 50 or height < 50:
            return False
            
        # Add some padding around the vehicle (10% on each side)
        height_frame, width_frame = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        # Ensure coordinates are within frame boundaries
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(width_frame, x2 + pad_x)
        y2_pad = min(height_frame, y2 + pad_y)
        
        # Crop the vehicle from the frame
        vehicle_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Skip if the cropped image is empty
        if vehicle_img.size == 0:
            return False
            
        # Get vehicle signature
        signature = self._get_vehicle_signature(vehicle_img)
        
        # Convert signature to bytes for dictionary key
        signature_bytes = signature.tobytes()
        
        # Check if this is a new vehicle or enough time has passed
        current_time = time.time()
        
        if signature_bytes in self.captured_vehicles:
            # Vehicle was captured before, check if enough time has passed
            last_capture_time = self.captured_vehicles[signature_bytes]
            if current_time - last_capture_time < self.recapture_delay:
                return False
        elif self._is_same_vehicle(signature, bbox, self.captured_vehicles):
            # Similar vehicle was captured recently
            return False
            
        # Update capture time
        self.captured_vehicles[signature_bytes] = current_time
        
        # Generate filename with timestamp and vehicle class
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{vehicle_class}_{confidence:.2f}_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save the image
        cv2.imwrite(filepath, vehicle_img)
        
        return True