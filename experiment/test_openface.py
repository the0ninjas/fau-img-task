import sys
import time
import cv2
from pathlib import Path

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

    # Ensure selected camera index is used by FeatureExtraction
    settings['camera_device_index'] = cam_idx if cam_idx >= 0 else default_cam

    print('Launching OpenFace 2.2 FeatureExtraction...')
    of_proc, session_dir = launch_openface(settings, participant_id, session_id)
    if not of_proc or not session_dir:
        print('Failed to launch OpenFace FeatureExtraction.exe. Check path and permissions.')
        return
    print(f'Output dir: {session_dir}')

    outlet = MarkerOutlet(settings.get('lsl_stream_name', 'psychopy_markers'), settings.get('lsl_stream_type', 'Markers'))
    duration_s = 10.0
    print(f'Pushing a few test markers over {duration_s:.0f} seconds while OpenFace runs...')
    t0 = core.getTime()
    while core.getTime() - t0 < duration_s:
        ts = outlet.push('TEST_MARKER')
        print(f'Sent TEST_MARKER @ {ts:.3f}')
        core.wait(1.0)

    print('Stopping OpenFace FeatureExtraction...')
    safe_terminate(of_proc)

    # Locate newest CSV written by FeatureExtraction
    csv_path = None
    try:
        pdir = Path(session_dir)
        candidates = list(pdir.glob('webcam*.csv')) + list(pdir.glob('*.csv'))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            csv_path = str(candidates[0])
    except Exception:
        csv_path = None

    if csv_path:
        print(f'OpenFace CSV written: {csv_path}')
    else:
        print('No CSV found in session directory yet. It may be flushed shortly after termination.')


if __name__ == '__main__':
    main()


