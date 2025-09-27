import os
import sys
import json
import subprocess
import time
from datetime import datetime
import numpy as np

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


# ------------------------
# Valence rating utilities
# ------------------------

def build_valence_ui(win: visual.Window, bar_height: float = 14):
    """Create and return the valence rating UI elements bound to the given window.

    Returns a dictionary with keys: rating_bar, grad_img, neg_label, pos_label,
    neg_emoji, pos_emoji, ticks, rating_instr, bar_y, bar_width, bar_height.
    """
    w, h = win.size
    bar_width = min(900, int(w * 0.8))
    # Compress rating section and push it closer to bottom
    bar_y = -h / 2 + 60

    # Border-only bar; gradient drawn via single ImageStim
    rating_bar = visual.Rect(
        win, width=bar_width, height=bar_height, lineColor=[1, 1, 1], fillColor=None, pos=(0, bar_y), units='pix'
    )

    # Pre-render gradient array (float RGB 0..1) from red->green left to right
    grad_arr = np.zeros((int(bar_height), int(bar_width), 3), dtype=np.float32)
    for x in range(int(bar_width)):
        t = x / max(1, int(bar_width) - 1)
        r = 1.0 - t
        g = t
        grad_arr[:, x, 0] = r
        grad_arr[:, x, 1] = g
        grad_arr[:, x, 2] = 0.0
    grad_img = visual.ImageStim(win, image=grad_arr, size=(bar_width, bar_height), pos=(0, bar_y), units='pix', interpolate=True)

    # Labels, emojis, ticks, instruction
    neg_label = visual.TextStim(win, text='Negative', color=[1, -1, -1], height=18, pos=(-(bar_width/2) - 100, bar_y))
    pos_label = visual.TextStim(win, text='Positive', color=[-1, 1, -1], height=18, pos=((bar_width/2) + 90, bar_y))
    neg_emoji = visual.TextStim(win, text='☹', color=[1, 1, 1], height=28, pos=(-(bar_width/2), bar_y + 34))
    neutral_emoji = visual.TextStim(win, text='Neutral', color=[1, 1, 1], height=18, pos=(0, bar_y + 34))
    pos_emoji = visual.TextStim(win, text='☺', color=[1, 1, 1], height=28, pos=((bar_width/2), bar_y + 34))

    ticks = []
    step = bar_width / 6.0
    for i in range(7):
        x = -bar_width/2 + i * step
        ticks.append(visual.TextStim(win, text=str(i+1), color=[1, 1, 1], height=16, pos=(x, bar_y - 26)))

    rating_instr = visual.TextStim(
        win,
        text='How do you feel? (1 = most negative, 4 = neutral, 7 = most positive)',
        color=[1, 1, 1], height=18, pos=(0, bar_y + 60), wrapWidth=1600
    )

    return {
        'rating_bar': rating_bar,
        'grad_img': grad_img,
        'neg_label': neg_label,
        'pos_label': pos_label,
        'neg_emoji': neg_emoji,
        'neutral_emoji': neutral_emoji,
        'pos_emoji': pos_emoji,
        'ticks': ticks,
        'rating_instr': rating_instr,
        'bar_y': bar_y,
        'bar_width': bar_width,
        'bar_height': bar_height,
    }


def draw_valence_ui(ui: dict, draw_instruction: bool = True):
    """Draw the valence rating UI elements in the correct order."""
    if draw_instruction and ui.get('rating_instr') is not None:
        ui['rating_instr'].draw()
    ui['grad_img'].draw()
    ui['rating_bar'].draw()
    ui['neg_label'].draw(); ui['pos_label'].draw()
    ui['neg_emoji'].draw(); ui['pos_emoji'].draw();
    if 'neutral_emoji' in ui and ui['neutral_emoji'] is not None:
        ui['neutral_emoji'].draw()
    for t in ui['ticks']:
        t.draw()


def layout_image_above_valence(win: visual.Window, image_stim: visual.ImageStim, ui: dict,
                               top_margin: float = 40.0, gap: float = 30.0, rating_section_total: float = 100.0):
    """Scale and center the image within the available region above the rating area to avoid overlap."""
    w, h = win.size
    bar_y = ui.get('bar_y', -h/2 + 60)
    rating_top_y = bar_y + (rating_section_total / 2.0)
    image_area_top = (h / 2.0) - top_margin
    image_area_bottom = rating_top_y + gap
    available_h = max(100.0, image_area_top - image_area_bottom)
    orig_w, orig_h = image_stim.size
    scale = min(1.0, available_h / orig_h, (w * 0.9) / orig_w)
    new_w, new_h = orig_w * scale, orig_h * scale
    image_stim.size = (new_w, new_h)
    pos_y = (image_area_top + image_area_bottom) / 2.0
    image_stim.pos = (0, pos_y)
    return image_stim


# ------------------------
# Countdown (ring) utilities (static grey ring)
# ------------------------

def build_countdown_ui(win: visual.Window, duration_s: float, offset_xy: tuple | None = None,
                       radius: int = 32) -> dict:
    w, h = win.size
    if offset_xy is None:
        center = (w/2 - 60, h/2 - 60)
    else:
        center = offset_xy
    track = visual.Circle(win, radius=radius, edges=128, lineColor=[0.3, 0.3, 0.3], lineWidth=4,
                          fillColor=None, pos=center, units='pix')
    text = visual.TextStim(win, text='', color=[0.7, 0.7, 0.7], height=24, pos=center, units='pix')
    return {'track': track, 'text': text, 'center': center, 'radius': radius}


def update_countdown_ui(ui: dict, remaining_s: float):
    try:
        ui['text'].text = f"{int(max(0, round(remaining_s)))}"
    except Exception:
        pass


def draw_countdown_ui(ui: dict):
    ui['track'].draw()
    if 'text' in ui and ui['text'] is not None:
        ui['text'].draw()

