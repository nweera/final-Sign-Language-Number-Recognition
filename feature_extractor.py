import numpy as np
from HandTrackingModule import handDetector

class HandFeatureExtractor:
    def __init__(self):
        """Initialize the hand feature extractor with hand detector"""
        self.detector = handDetector(detectionCon=0.8, maxHands=1)
        
    def extract_features(self, img):
        """Extract hand landmarks and calculate features
        
        Args:
            img: Input image containing hand
            
        Returns:
            numpy array of features for hand recognition
        """
        img = self.detector.findHands(img, draw=False)
        landmarks = self.detector.findPosition(img, draw=False)
        
        if len(landmarks) < 21:  # Not enough landmarks for a complete hand
            return np.zeros(84)  # Return zeros for consistent shape (42 raw + 42 engineered features)
            
        # Convert to numpy array for easier manipulation
        lm_array = np.array(landmarks)[:, 1:]  # Extract x,y coordinates
        
        # Basic features - normalized coordinates
        # Find bounding box of the hand to normalize coordinates
        x_min, y_min = np.min(lm_array, axis=0)
        x_max, y_max = np.max(lm_array, axis=0)
        
        # Avoid division by zero
        x_range = max(1, x_max - x_min)
        y_range = max(1, y_max - y_min)
        
        # Normalize coordinates to [0,1] range
        normalized_lm = (lm_array - [x_min, y_min]) / [x_range, y_range]
        
        # Flatten landmarks
        raw_features = normalized_lm.flatten()
        
        # Add engineered features - distances between key points
        engineered_features = []
        
        # Reference point (palm center - landmark 0)
        palm = normalized_lm[0]
        
        # Calculate distances from palm to each fingertip (landmarks 4, 8, 12, 16, 20)
        fingertips = [4, 8, 12, 16, 20]
        for tip in fingertips:
            dist = np.linalg.norm(normalized_lm[tip] - palm)
            engineered_features.append(dist)
        
        # Distances between adjacent fingertips
        for i in range(len(fingertips)-1):
            dist = np.linalg.norm(normalized_lm[fingertips[i]] - normalized_lm[fingertips[i+1]])
            engineered_features.append(dist)
        
        # Angles at each finger joint
        for finger in range(5):  # 5 fingers
            base = finger * 4 + 1  # Base of each finger
            for joint in range(3):  # 3 joints per finger
                p1 = normalized_lm[base + joint]
                p2 = normalized_lm[base + joint + 1]
                # Use the palm as a reference point
                angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                engineered_features.append(angle)
        
        # Combine raw and engineered features
        all_features = np.concatenate([raw_features, np.array(engineered_features)])
        return all_features