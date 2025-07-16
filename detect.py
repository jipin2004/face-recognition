import cv2
import numpy as np
import os
import sqlite3


class FaceRecognition:
    def __init__(self):
        # Load Haar Cascade
        self.facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Load the recognizer model
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("recognizer/trainingdata.yml")

        # Confidence threshold
        self.CONFIDENCE_THRESHOLD = 70

    def get_profile(self, Id):
        """Fetch complete user profile from database"""
        try:
            conn = sqlite3.connect("sqlite.db")
            cursor = conn.execute("SELECT * FROM PEOPLES WHERE Id=?", (Id,))
            profile = cursor.fetchone()
            conn.close()
            return profile
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None

    def detect_faces(self):
        # Open Camera
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to capture image.")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.facedetect.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y + h, x:x + w]

                # Predict ID and confidence
                Id, conf = self.recognizer.predict(face_roi)

                # Check confidence
                if conf < self.CONFIDENCE_THRESHOLD:
                    # Get full profile
                    profile = self.get_profile(Id)

                    if profile:
                        # Draw green rectangle for known faces
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Prepare display lines
                        display_lines = [
                            f"Name: {profile[1]}",
                            f"Data: {profile[3]}"
                        ]

                        # Display information
                        for i, line in enumerate(display_lines):
                            cv2.putText(img, line,
                                        (x, y + h + 20 + (i * 25)),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                        (0, 255, 127), 2)
                    else:
                        # Unknown face
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(img, "Unknown",
                                    (x, y + h + 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                    (0, 0, 255), 2)
                else:
                    # Low confidence match
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, "Unknown",
                                (x, y + h + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 0, 255), 2)

            # Show the image
            cv2.imshow("Face Recognition", img)

            # Break on 'q' key
            if cv2.waitKey(1) == ord('q'):
                break

        # Cleanup
        cam.release()
        cv2.destroyAllWindows()


def main():
    face_rec = FaceRecognition()
    face_rec.detect_faces()


if __name__ == "__main__":
    main()
