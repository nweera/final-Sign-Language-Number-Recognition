import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from feature_extractor import HandFeatureExtractor

class NumberSignRecognizer:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.model = None
        self.feature_extractor = HandFeatureExtractor()
        self.classes = list(range(10))  # 0-9 digits
        
    def load_dataset(self, respect_splits=False):
        """
        Load images from dataset directories and extract features
        
        Args:
            respect_splits: If True, return separate train/val/test datasets
                           If False, combine all data into one dataset
        """
        if respect_splits and os.path.exists(os.path.join(self.dataset_path, "train")):
            # Return separate datasets for each split
            splits = {}
            
            # Process each split separately
            for split in ["train", "validation", "test"]:
                X_split = []
                y_split = []
                
                split_path = os.path.join(self.dataset_path, split)
                if not os.path.exists(split_path):
                    print(f"Warning: '{split}' folder not found")
                    continue
                    
                print(f"Processing {split} split...")
                for number in self.classes:
                    folder_path = os.path.join(split_path, str(number))
                    if not os.path.exists(folder_path):
                        print(f"Warning: {split}/{number} folder not found")
                        continue
                    
                    self._process_folder(folder_path, number, X_split, y_split)
                
                if X_split:
                    splits[split] = (np.array(X_split), np.array(y_split))
                    print(f"  {split}: {len(X_split)} samples")
                    # Check class distribution
                    counter = Counter(y_split)
                    print(f"  {split} class distribution:", dict(sorted(counter.items())))
            
            if not splits:
                raise ValueError("No valid data found in any split. Check your dataset path.")
                
            return splits
        else:
            # Original behavior - combine all data
            X = []  # Features
            y = []  # Labels
            
            if os.path.exists(os.path.join(self.dataset_path, "train")):
                # Process organized dataset but combine all splits
                for split in ["train", "validation", "test"]:
                    split_path = os.path.join(self.dataset_path, split)
                    if not os.path.exists(split_path):
                        continue
                        
                    print(f"Processing {split} split...")
                    for number in self.classes:
                        folder_path = os.path.join(split_path, str(number))
                        if not os.path.exists(folder_path):
                            print(f"Warning: {split}/{number} folder not found")
                            continue
                        
                        self._process_folder(folder_path, number, X, y)
            else:
                # Process flat dataset structure
                for number in self.classes:
                    folder_path = os.path.join(self.dataset_path, str(number))
                    if not os.path.exists(folder_path):
                        print(f"Warning: Folder for number {number} not found")
                        continue
                    
                    self._process_folder(folder_path, number, X, y)
            
            if not X:
                raise ValueError("No valid data found. Check your dataset path.")
                
            print(f"Dataset loaded: {len(X)} samples, {len(X[0])} features per sample")
            
            # Check class distribution
            counter = Counter(y)
            print("Class distribution:", dict(sorted(counter.items())))
            
            return np.array(X), np.array(y)
    
    def _process_folder(self, folder_path, label, X, y):
        """Process all images in a folder"""
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(img)
                    
                    # Only add if valid features were extracted (non-zero)
                    if not np.all(features == 0):
                        X.append(features)
                        y.append(label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    def train_model(self, optimize=False):
        """Train a machine learning model on the dataset"""
        print("Loading dataset...")
        
        # Check if dataset has predefined splits
        has_splits = os.path.exists(os.path.join(self.dataset_path, "train"))
        
        if has_splits:
            print("Using predefined train/validation/test splits...")
            splits = self.load_dataset(respect_splits=True)
            
            # Extract train, validation and test data
            if 'train' not in splits:
                print("Error: Training data not found in splits!")
                return False
                
            X_train, y_train = splits['train']
            
            # Use validation set if available, otherwise use a portion of train
            if 'validation' in splits:
                X_val, y_val = splits['validation']
            else:
                print("No validation set found, using 20% of training data")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
            
            # Use test set if available
            if 'test' in splits:
                X_test, y_test = splits['test']
            else:
                X_test, y_test = X_val, y_val
                print("No test set found, using validation set for testing")
        else:
            # No predefined splits, load all data and split manually
            X, y = self.load_dataset()
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Further split test into validation and test
            X_val, X_test, y_val, y_test = train_test_split(
                X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
            )
        
        if optimize:
            # Use grid search to find optimal hyperparameters
            print("Optimizing model hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1  # Use all available cores
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_
        else:
            # Train with default parameters
            print("Training model...")
            self.model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            self.model.fit(X_train, y_train)
        
        # Evaluate the model
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_preds)
        test_accuracy = accuracy_score(y_test, test_preds)
        
        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")
        
        # Feature importance analysis
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 10 most important features:")
        for i in range(min(10, len(importances))):
            print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
        
        # Save the trained model
        with open('number_sign_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved to number_sign_model.pkl")
        
        return True
    
    def load_model(self, model_path='number_sign_model.pkl'):
        """Load a previously trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Train a model first.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_from_image(self, img):
        """Predict the number from an image"""
        if self.model is None:
            print("No model loaded. Train or load a model first.")
            return None
        
        features = self.feature_extractor.extract_features(img)
        
        # If no valid hand features were detected
        if np.all(features == 0):
            return None
            
        # Make prediction
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        confidence = np.max(probabilities)
        
        # Get top 3 predictions
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3_predictions = [(self.classes[idx], probabilities[idx]) for idx in top3_indices]
        
        return prediction, confidence, top3_predictions
    
    def run_webcam_detection(self):
        """Run real-time detection using webcam"""
        if self.model is None:
            if not self.load_model():
                print("Training new model...")
                if not self.train_model():
                    print("Failed to train model. Check your dataset.")
                    return

        # Try to open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam. Check your camera connection.")
            return

        # Adjust camera settings for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        pTime = 0

        # For tracking stability of predictions
        prediction_history = []
        max_history = 15  # Increased for better stability

        # Minimum confidence threshold to accept a prediction
        min_confidence = 0.4  # Slightly increased threshold

        print("\nWebcam detection started.")
        print("Controls:")
        print("- S: Take a snapshot and save")
        print("- D: Toggle debug visualization")
        print("- Q: Quit")

        debug_mode = False
        
        try:
            while True:
                success, img = cap.read()
                if not success:
                    print("Failed to get frame from webcam")
                    break

                # Mirror the image horizontally
                img = cv2.flip(img, 1)
                img_height, img_width = img.shape[:2]

                # Create a copy for debugging
                debug_img = img.copy() if debug_mode else None

                # Process the image
                img = self.feature_extractor.detector.findHands(img)
                landmarks = self.feature_extractor.detector.findPosition(img, draw=True)

                # Calculate FPS
                cTime = time.time()
                fps = 1 / (cTime - pTime) if cTime != pTime else 0
                pTime = cTime

                # Show FPS
                cv2.putText(img, f"FPS: {int(fps)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                # Make prediction if hand is detected
                if landmarks and len(landmarks) >= 15:  # Ensure enough landmarks are detected
                    result = self.predict_from_image(img)
                    if result is not None:
                        number, confidence, top3 = result

                        # Only consider predictions with reasonable confidence
                        if confidence >= min_confidence:
                            # Add to history for stability
                            prediction_history.append(number)
                            if len(prediction_history) > max_history:
                                prediction_history.pop(0)

                        # Get most common prediction from history
                        if prediction_history:
                            counter = Counter(prediction_history)
                            stable_prediction = counter.most_common(1)[0][0]
                            stability = counter[stable_prediction] / len(prediction_history)

                            # Enhanced visualization with color coding based on confidence
                            color = (0, 255, 0) if confidence > 0.7 else \
                                   (0, 255, 255) if confidence > 0.5 else \
                                   (0, 165, 255)  # Green > Yellow > Orange

                            # Display prediction and confidence
                            cv2.putText(img, f"Number: {stable_prediction}", (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                            cv2.putText(img, f"Conf: {confidence:.2f}", (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            cv2.putText(img, f"Stability: {stability:.2f}", (10, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                            # Show top 3 predictions in debug mode
                            if debug_mode:
                                y_offset = 190
                                cv2.putText(img, "Top 3 predictions:", 
                                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                                for i, (pred, prob) in enumerate(top3):
                                    y_offset += 25
                                    cv2.putText(img, f"{i+1}. Number {pred}: {prob:.2f}", 
                                                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                                0.6, (255, 255, 255), 1)
                else:
                    # Clear history when no hand is detected
                    if prediction_history:
                        # Gradually reduce history to prevent flickering
                        if len(prediction_history) > 2:
                            prediction_history.pop(0)

                # Add colored rectangle for visual feedback
                if prediction_history:
                    counter = Counter(prediction_history)
                    if counter:
                        stable_prediction = counter.most_common(1)[0][0]
                        stability = counter[stable_prediction] / len(prediction_history)
                        
                        # More dynamic color based on stability and history length
                        r = int(255 * (1 - stability))
                        g = int(255 * stability)
                        cv2.rectangle(img, (img_width-60, 10), (img_width-10, 60), 
                                      (0, g, r), -1)
                        
                        # Show number in the indicator
                        cv2.putText(img, str(stable_prediction), 
                                    (img_width-45, 45), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

                # Display instructions
                cv2.putText(img, "S: Snapshot | D: Debug | Q: Quit",
                            (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 1)

                cv2.imshow("Number Sign Detection", img)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Take a snapshot and save
                    timestamp = int(time.time() * 1000)
                    snapshot_filename = f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(snapshot_filename, img)
                    print(f"Snapshot saved to {snapshot_filename}")
                elif key == ord('d'):
                    # Toggle debug mode
                    debug_mode = not debug_mode
                    print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

        except Exception as e:
            print(f"Error during webcam detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def evaluate_model_with_visualization(self):
        """Evaluate model performance with confusion matrix visualization"""
        if self.model is None:
            if not self.load_model():
                print("No model available. Train a model first.")
                return
        
        # Check if we have predefined splits
        has_splits = os.path.exists(os.path.join(self.dataset_path, "train"))
        
        if has_splits:
            print("Using predefined test split for evaluation...")
            splits = self.load_dataset(respect_splits=True)
            
            # Use test set for evaluation if available
            if 'test' in splits:
                X_test, y_test = splits['test']
                print(f"Evaluating on dedicated test set ({len(X_test)} samples)")
                
                # Make predictions on test set
                y_pred = self.model.predict(X_test)
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Print classification report
                print("Classification Report (Test Set):")
                print(classification_report(y_test, y_pred))
            else:
                print("No test set found. Loading all data for evaluation.")
                X, y = self.load_dataset()
                y_pred = self.model.predict(X)
                cm = confusion_matrix(y, y_pred)
                print("Classification Report (All Data):")
                print(classification_report(y, y_pred))
        else:
            # No predefined splits, evaluate on all data
            print("No predefined splits found. Loading all data for evaluation.")
            X, y = self.load_dataset()
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            # Print classification report
            print("Classification Report:")
            print(classification_report(y, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=16)
        plt.colorbar()
        
        # Set class labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, fontsize=12)
        plt.yticks(tick_marks, self.classes, fontsize=12)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12)
        
        plt.ylabel('True Number', fontsize=14)
        plt.xlabel('Predicted Number', fontsize=14)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved to confusion_matrix.png")
        
        # Show the plot
        plt.show()
        
        # Calculate per-class accuracy
        print("\nPer-class accuracy:")
        for cls in self.classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) > 0:
                cls_acc = np.mean(y_pred[cls_indices] == cls)
                print(f"Class {cls}: {cls_acc:.2f}")