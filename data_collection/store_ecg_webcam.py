import asyncio
import csv
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import pyautogui
from bleak import BleakClient, BleakScanner

import config

# Constants for ECG
SAMPLING_RATE = 128
NOTIFY_CHARACTERISTIC_UUID = "34800002-7185-4d5d-b431-630e7050e8f0"
WRITE_CHARACTERISTIC_UUID = "34800001-7185-4d5d-b431-630e7050e8f0"


class RawECGLogger:
    def __init__(self, name):
        self.log_dir = Path(f"data/{name}/ecg_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"ecg_raw_log_{timestamp}.csv"

        # Write CSV headers
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "raw_value", "scaled_value"])

    def log_sample(self, raw_value, scaled_value):
        current_ts = datetime.now().timestamp()
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([current_ts, raw_value, scaled_value])


class DataView:
    def __init__(self, array):
        self.array = array

    def get_uint8(self, index):
        return self.array[index]

    def get_uint32(self, index):
        return int.from_bytes(self.array[index : index + 4], byteorder="little")

    def get_int32(self, index):
        return int.from_bytes(
            self.array[index : index + 4], byteorder="little", signed=True
        )


def webcam_capture(stop_event, name):
    """Run webcam capture in a separate thread."""
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with index {config.WEBCAM_INDEX}")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 30

    # Video writer setup
    video_filename = f"data/{name}/video_recordings/webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    print(f"[{datetime.now()}] Webcam recording started. Press 'q' to stop.")

    last_capture_time = time.time()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        out.write(frame)
        current_time = time.time()
        timestamp = current_time

        # Capture every 0.2 seconds
        if config.WEBCAM_FRAME and current_time - last_capture_time >= 0.2:
            screenshot_name = f"data/{name}/screenshots/screenshot_{timestamp}.png"
            screenshot = pyautogui.screenshot()
            screenshot.save(screenshot_name)

            webcam_frame_name = f"data/{name}/webcam_frames/webcam_{timestamp}.png"
            cv2.imwrite(webcam_frame_name, frame)

            last_capture_time = current_time

    cap.release()
    out.release()
    print(f"[{datetime.now()}] Webcam recording stopped.")


async def run_ble_client(end_of_serial, stop_event, name):
    logger = RawECGLogger(name)  # Pass name to RawECGLogger

    print(f"[{datetime.now()}] Starting BLE client...")
    devices = await BleakScanner.discover()

    # Look for a device whose name ends with the provided serial
    device = next(
        (d for d in devices if d.name and d.name.endswith(end_of_serial)), None
    )
    if not device:
        print(f"Device ending with {end_of_serial} not found!")
        stop_event.set()  # Signal to stop other threads
        return

    print(f"[{datetime.now()}] Device found: {device.name}")
    print(f"[{datetime.now()}] Connecting to device...")

    disconnect_event = asyncio.Event()

    def disconnect_callback(_):
        print(f"[{datetime.now()}] Disconnected!")
        disconnect_event.set()
        stop_event.set()  # Signal to stop other threads

    def notification_handler(_, data):
        dv = DataView(data)
        # Check packet type
        if dv.get_uint8(0) == 2 and dv.get_uint8(1) == 100:
            # Each packet contains 16 samples; process each one
            for i in range(16):
                # Read raw ECG value from the packet
                raw_dv = dv.get_uint32(6 + i * 4)
                sample_mv = dv.get_int32(6 + i * 4) * 0.38 * 0.001
                # Log the sample with the current system timestamp
                logger.log_sample(raw_dv, sample_mv)

    try:
        async with BleakClient(
            device.address, disconnected_callback=disconnect_callback
        ) as client:
            print(f"[{datetime.now()}] Connected. Starting notifications...")

            await client.start_notify(NOTIFY_CHARACTERISTIC_UUID, notification_handler)

            # Subscribe to the ECG stream
            await client.write_gatt_char(
                WRITE_CHARACTERISTIC_UUID,
                bytearray([1, 100]) + bytearray("/Meas/ECG/128", "utf-8"),
                response=True,
            )

            print(f"[{datetime.now()}] Subscribed to ECG stream")

            # Wait until disconnected or stop event is set
            while not stop_event.is_set() and not disconnect_event.is_set():
                await asyncio.sleep(0.1)

            # Cleanup
            if client.is_connected:
                await client.write_gatt_char(
                    WRITE_CHARACTERISTIC_UUID, bytearray([2, 100])
                )
                await client.stop_notify(NOTIFY_CHARACTERISTIC_UUID)
    except Exception as e:
        print(f"[{datetime.now()}] Error in BLE client: {e}")
        stop_event.set()  # Signal to stop other threads


async def main_async(stop_event, end_of_serial, name):
    """Main async function to run ECG collection"""
    await run_ble_client(end_of_serial, stop_event, name)


def main(name):  # Add name as a parameter
    logging.basicConfig(level=logging.INFO)
    os.makedirs(f"data/{name}/webcam_frames", exist_ok=True)
    os.makedirs(f"data/{name}/screenshots", exist_ok=True)
    os.makedirs(f"data/{name}/video_recordings", exist_ok=True)
    os.makedirs(f"data/{name}/ecg_logs", exist_ok=True)
    stop_event = threading.Event()

    # Start webcam capture in a thread
    webcam_thread = threading.Thread(
        target=webcam_capture, args=(stop_event, name)
    )  # Pass name
    webcam_thread.start()

    try:
        # Run ECG collection in the main thread
        asyncio.run(main_async(stop_event, str(config.ECG_SERIAL), name))
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] Program interrupted by user")
    finally:
        # Stop all threads when program ends
        stop_event.set()
        webcam_thread.join()
        print(f"[{datetime.now()}] All processes stopped")


if __name__ == "__main__":
    # count no. of folders in data directory
    count = len(os.listdir("data"))
    main(count)
