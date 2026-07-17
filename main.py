import os
import cv2
import tensorflow as tf
from FaceID import set_face_id, resize_frame_to_square, verify
from layer import L1Dist

while True:
    # Display options to the user
    print("\nOptions:")
    print("1. Set up Face ID (Capture images and train model)")
    print("2. Real Time Test (Verify identity via webcam)")
    print("3. Quit")

    # Get user input
    choice = input("Enter your choice: ")

    if choice == "1":
        # Logic for setting up ID
        set_face_id()
    elif choice == "2":
        # Load tensorflow/keras model
        model_path = 'siamesemodel.h5'
        if not os.path.exists(model_path):
            print(f"[!] Model file '{model_path}' not found. Please set up Face ID first to train the model.")
            continue
            
        print("[*] Loading model...")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={'L1Dist': L1Dist})
        except Exception as e:
            print("[!] Failed to load model:", e)
            continue
        
        # OpenCV Real Time Verification
        cap = cv2.VideoCapture(0)
        
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            print("[!] Error: Could not open webcam.")
            continue
            
        cv2.namedWindow('Verification', cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty('Verification', cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
            
        print("\n[*] Camera opened.")
        print("Click on the video window first, then:")
        print("Press 'v' to verify | Press 'q' or ESC to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Error: Failed to read frame from camera.")
                break

            # Resize the cropped square region to 250x250
            frame_sq = resize_frame_to_square(frame, 250)

            # Show resized frame
            cv2.imshow('Verification', frame_sq)

            k = cv2.waitKey(30)
            keycode = k & 0xFF
            if keycode != 0xFF:
                key = chr(keycode).lower()
                print(f"[*] Key detected: '{key}' (code: {k})")
                
                # Verification trigger: 'v' on English layout, 'ر' (code 209/'ñ') on Arabic layout
                if key == 'v' or keycode == 209 or key == 'ñ':
                    input_path = os.path.join('application_data', 'input_images', 'input_image.jpg')
                    os.makedirs(os.path.dirname(input_path), exist_ok=True)
                    if not cv2.imwrite(input_path, frame_sq):
                        print(f"[!] Failed to save {input_path}")
                        break
                    
                    try:
                        results, verified = verify(model, 0.85, 0.5)
                        print("verified ✅" if verified else "not verified ❌")
                    except Exception as e:
                        print("[!] Verification failed:", e)
                        break

                # Quit trigger: 'q' on English layout, 'ض' (code 214/'ö') on Arabic layout, or ESC (keycode 27)
                elif key == 'q' or keycode == 214 or key == 'ö' or keycode == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()
    elif choice == "3":
        # Quit the program
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please enter a valid option.")

