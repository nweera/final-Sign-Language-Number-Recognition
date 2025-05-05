import os
import cv2
import time
import pickle
from collections import Counter
from recognizer import NumberSignRecognizer

def delete_model():
    """Deletes the saved model file if it exists."""
    model_file = 'number_sign_model.pkl'
    if os.path.exists(model_file):
        try:
            os.remove(model_file)
            print(f"Model '{model_file}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting model file: {e}")
    else:
        print("No model file found to delete.")

def main():
    print("Enhanced Number Sign Language Recognition System")
    print("=============================================")
    
    # Get dataset path from user or use default
    default_path = "dataset"
    user_path = input(f"Enter dataset path (press Enter for default '{default_path}'): ")
    dataset_path = user_path if user_path.strip() else default_path
    
    recognizer = NumberSignRecognizer(dataset_path)
    
    try:
        while True:
            print("\nMenu:")
            print("1. Train new model")
            print("2. Run webcam detection")
            print("3. Evaluate model")
            print("4. Advanced training (with hyperparameter optimization)")
            print("5. Exit")
            
            choice = input("Enter your choice (1-5): ")
            
            if choice == '1':
                recognizer.train_model()
            elif choice == '2':
                recognizer.run_webcam_detection()
            elif choice == '3':
                recognizer.evaluate_model_with_visualization()
            elif choice == '4':
                recognizer.train_model(optimize=True)
            elif choice == '5':
                print("Exiting program...")
                break
            else:
                print("Invalid choice. Please try again.")
    finally:
        # Ensure the model is deleted when exiting
        delete_model()

if __name__ == "__main__":
    main()
