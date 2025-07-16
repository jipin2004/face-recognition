import cv2
import numpy as np
import sqlite3
import os

# Ensure dataset directory exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Load the Haar Cascade Classifier for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the camera
cam = cv2.VideoCapture(0)


def insert_or_update(Id, Name, Data):
    """Insert or update user information in the SQLite database"""
    try:
        conn = sqlite3.connect("sqlite.db")
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute('''CREATE TABLE IF NOT EXISTS PEOPLES(
            Id INTEGER PRIMARY KEY,
            Name TEXT,
            Data TEXT
        )''')

        # Check if record exists
        cursor.execute("SELECT * FROM PEOPLES WHERE Id=?", (Id,))
        existing = cursor.fetchone()

        if existing:
            # Update existing record
            cursor.execute("UPDATE PEOPLES SET Name=?, Data=? WHERE Id=?", (Name, Data, Id))
        else:
            # Insert new record
            cursor.execute("INSERT INTO PEOPLES (Id, Name, Data) VALUES (?, ?, ?)", (Id, Name, Data))

        conn.commit()
        conn.close()
        print("Database operation successful")
    except sqlite3.Error as e:
        print(f"Database error: {e}")


def main():
    # Get user details
    Id = input('Enter User Id: ')
    Name = input('Enter User Name: ')
    Data = input('Enter User Data: ')

    # Insert or update user in database
    insert_or_update(Id, Name, Data)

    # Face capture parameters
    sampleNum = 0
    print("Look directly at the camera. Press 'q' to quit.")

    while True:
        # Read frame from camera
        ret, img = cam.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Increment sample number
            sampleNum += 1

            # Save the face image
            cv2.imwrite(f"dataset/User.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])

            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Small delay between captures
            cv2.waitKey(200)

        # Show the image
        cv2.imshow("Face Capture", img)

        # Display number of samples captured
        cv2.putText(img, f"Samples: {sampleNum}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Break if enough samples or 'q' is pressed
        if sampleNum > 20 or key == ord('q'):
            break

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
