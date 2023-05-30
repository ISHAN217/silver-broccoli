import cv2
import mediapipe as mp

# Define the hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Define the video capture object
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Flip the frame horizontally for natural hand orientation
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for input to the hand detection model
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the hand detection model on the image
    results = hands.process(image)

    # If at least one hand is detected
    if results.multi_hand_landmarks:
        # Draw landmarks on the frame for each hand
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Show the resulting frame
    cv2.imshow("Sign Language Detection", frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
