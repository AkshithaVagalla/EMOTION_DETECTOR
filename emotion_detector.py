import cv2
import numpy as np
from deepface import DeepFace

print("✅ Libraries imported successfully.")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Cannot access the webcam.")
    exit()

print("✅ Webcam started.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Frame not captured.")
        break

    try:
        print("🔍 Analyzing frame...")
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        print(f"✅ Detected emotion: {emotion}")
    except Exception as e:
        print(f"❌ DeepFace Error: {e}")
        emotion = "neutral"

    # Display output
    cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Program exited successfully.")
