import cv2


def test_webcam(index):
    """Test if a webcam can be opened at the given index."""
    # Attempt to open the webcam at the specified index
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"Webcam index {index} could not be opened.")
        return False

    print(f"Webcam index {index} opened successfully! Press 'q' to close the window.")

    # Display the webcam feed
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame at index {index}.")
            break

        # Show the frame in a window
        cv2.imshow(f"Webcam Test - Index {index}", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    return True


def main():
    print("Testing webcam indices...")
    print("Enter the index you want to test (e.g., 0, 1, 2) or type 'exit' to quit.")

    while True:
        user_input = input("Enter webcam index to test: ")

        if user_input.lower() == "exit":
            print("Exiting webcam test.")
            break

        try:
            index = int(user_input)
            test_webcam(index)
        except ValueError:
            print("Please enter a valid integer or 'exit'.")


if __name__ == "__main__":
    main()
