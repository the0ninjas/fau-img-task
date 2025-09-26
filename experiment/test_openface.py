import sys
import time
import cv2
import threading
import re
import csv
from datetime import datetime

from psychopy import core

from .utils import load_settings, launch_openface, safe_terminate, MarkerOutlet


def list_available_cameras(max_index: int = 10) -> list:
    available = []
    backends = [
        getattr(cv2, 'CAP_MSMF', None),  # Modern Windows backend
        getattr(cv2, 'CAP_DSHOW', None), # Legacy DirectShow
        getattr(cv2, 'CAP_VFW', None),   # Very old Video for Windows
        None,                            # Default
    ]
    for idx in range(max_index):
        opened = False
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            except Exception:
                cap = None
            if cap is not None and cap.isOpened():
                # Try to read a frame
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                    opened = True
                cap.release()
                if opened:
                    break
    return sorted(set(available))


def choose_camera(available: list, default_idx: int) -> int:
    if not available:
        print('No cameras detected.')
        return -1
    print('Available cameras:')
    for i in available:
        print(f'  - Index {i}')
    prompt = f'Select camera index (Enter for default {default_idx}): '
    choice = input(prompt).strip()
    if choice == '':
        return default_idx if default_idx in available else available[0]
    try:
        selected = int(choice)
        if selected in available:
            return selected
    except ValueError:
        pass
    print('Invalid selection; using first available.')
    return available[0]


def main():
    settings = load_settings()
    participant_id = 'TEST'
    session_id = 'OFACE'

    # Enumerate and select camera
    print('Scanning for webcams...')
    available = list_available_cameras(max_index=10)
    default_cam = settings.get('camera_device_index', 0)
    cam_idx = choose_camera(available, default_cam)

    if cam_idx >= 0:
        print(f'Testing camera index {cam_idx}...')
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f'Camera {cam_idx} OK. Resolution: {w}x{h}')
            else:
                print(f'Camera {cam_idx} opened but failed to read a frame.')
            cap.release()
        else:
            print(f'Failed to open camera {cam_idx}.')

    print('Launching OpenFace...')
    # Create session dir (even if not launching old OpenFace subprocess)
    from datetime import datetime as _dt
    import os as _os
    out_root = settings.get('openface_output_dir', 'openface_out')
    session_dir = _os.path.abspath(_os.path.join(out_root, f'{participant_id}_{session_id}_{_dt.now().strftime("%Y%m%d_%H%M%S")}'))
    _os.makedirs(session_dir, exist_ok=True)
    print(f'Output dir: {session_dir}')

    # Prepare placeholder files for compatibility
    stdout_log_path = None
    parsed_csv_path = None
    if session_dir:
        stdout_log_path = f"{session_dir}\\openface_stdout.log"
        parsed_csv_path = f"{session_dir}\\openface_parsed.csv"
        try:
            with open(stdout_log_path, 'w', encoding='utf-8') as _f:
                pass
            with open(parsed_csv_path, 'w', encoding='utf-8', newline='') as _csvf:
                writer = csv.writer(_csvf)
                writer.writerow(["timestamp_iso", "channel", "payload"])  # header only
        except Exception:
            pass

    outlet = MarkerOutlet(settings.get('lsl_stream_name', 'psychopy_markers'), settings.get('lsl_stream_type', 'Markers'))
    print('Pushing a few test markers over 10 seconds...')
    t0 = core.getTime()

    # Start OpenFace streaming in-process
    from .openface_stream import OpenFaceStreamer
    streamer = OpenFaceStreamer(camera_index=cam_idx if cam_idx >= 0 else settings.get('camera_device_index', 0))
    stream_csv = streamer.stream(duration_seconds=10.0, session_dir=session_dir)
    print(f'OpenFace stream saved to: {stream_csv}')

    while core.getTime() - t0 < 10.0:
        ts = outlet.push('TEST_MARKER')
        print(f'Sent TEST_MARKER @ {ts:.3f}')
        core.wait(1.0)

    print('Stopping OpenFace...')
    print('Done.')


if __name__ == '__main__':
    main()


