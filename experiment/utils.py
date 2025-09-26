import os
import sys
import json
import subprocess
import time
from datetime import datetime

from psychopy import visual

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
    LSL_AVAILABLE = True
except Exception:
    LSL_AVAILABLE = False


def load_settings():
    settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.json')
    settings_path = os.path.abspath(settings_path)
    with open(settings_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def launch_openface(settings, participant_id: str, session_id: str):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Prefer launching a Python script if configured
    script_path = settings.get('openface_script')
    if script_path and not os.path.isabs(script_path):
        script_path = os.path.abspath(os.path.join(project_root, script_path))

    if script_path and os.path.isfile(script_path):
        out_root = settings.get('openface_output_dir', 'openface_out')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.abspath(os.path.join(out_root, f'{participant_id}_{session_id}_{timestamp}'))
        os.makedirs(session_dir, exist_ok=True)

        args = [sys.executable, script_path]
        args += settings.get('openface_args', [])

        creationflags = 0x08000000 if sys.platform.startswith('win') else 0
        try:
            proc = subprocess.Popen(
                args,
                cwd=os.path.dirname(script_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
            return proc, session_dir
        except Exception as e:
            print(f'Warning: Failed to launch OpenFace script: {e}')
            return None, None

    # Fallback: old executable-based launch (if users provide a path to the official binary)
    exe_path = settings.get('openface_feature_exe')
    if exe_path and not os.path.isabs(exe_path):
        exe_path = os.path.abspath(os.path.join(project_root, exe_path))

    resolved_exe = None
    if exe_path:
        if os.path.isfile(exe_path):
            resolved_exe = exe_path
        elif os.path.isdir(exe_path):
            for root, _, files in os.walk(exe_path):
                if 'FeatureExtraction.exe' in files:
                    resolved_exe = os.path.join(root, 'FeatureExtraction.exe')
                    break

    if not resolved_exe or not os.path.exists(resolved_exe):
        print('Warning: No OpenFace Python script or FeatureExtraction.exe found; proceeding without OpenFace capture.')
        return None, None

    out_root = settings.get('openface_output_dir', 'openface_out')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.abspath(os.path.join(out_root, f'{participant_id}_{session_id}_{timestamp}'))
    os.makedirs(session_dir, exist_ok=True)

    args = [resolved_exe]
    args += settings.get('openface_args', [])
    args += ['-out_dir', session_dir]
    args += ['-pose', '-2Dfp', '-3Dfp']

    creationflags = 0x08000000 if sys.platform.startswith('win') else 0

    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
        return proc, session_dir
    except Exception as e:
        print(f'Warning: Failed to launch OpenFace: {e}')
        return None, None


def safe_terminate(proc):
    if proc is None:
        return
    try:
        proc.terminate()
    except Exception:
        pass


class MarkerOutlet:
    def __init__(self, name: str, stream_type: str):
        self.enabled = False
        self.outlet = None
        if LSL_AVAILABLE:
            try:
                info = StreamInfo(name=name, type=stream_type, channel_count=1, nominal_srate=0,
                                  channel_format='string', source_id=f'{name}_{stream_type}')
                self.outlet = StreamOutlet(info)
                self.enabled = True
            except Exception as e:
                print(f'Warning: Failed to create LSL outlet: {e}')

    def push(self, marker: str):
        ts = local_clock() if LSL_AVAILABLE else time.time()
        if self.enabled and self.outlet is not None:
            try:
                self.outlet.push_sample([marker], timestamp=ts)
            except Exception as e:
                print(f'Warning: LSL push failed: {e}')
        return ts


def draw_red_frame(win: visual.Window, image_stim: visual.ImageStim, thickness: float = 8):
    color = [1, -1, -1]
    # Use the image stimulus size and position to draw a border around it
    img_w, img_h = image_stim.size
    frame = visual.Rect(
        win,
        width=img_w,
        height=img_h,
        lineColor=color,
        lineWidth=thickness,
        fillColor=None,
        pos=image_stim.pos,
        units='pix'
    )
    return [frame]


def get_image_list(stim_dir: str):
    if not os.path.exists(stim_dir):
        os.makedirs(stim_dir, exist_ok=True)
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    images = [os.path.join(stim_dir, f) for f in os.listdir(stim_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    images.sort()
    return images


