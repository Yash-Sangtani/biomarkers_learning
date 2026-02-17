import cv2


def main():
    # Open a connection to the webcam (0 for default camera)
    cap = cv2.VideoCapture(3)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame
        cv2.imshow("Webcam Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
