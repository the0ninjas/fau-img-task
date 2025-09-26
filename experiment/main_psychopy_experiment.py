import os
import random

from psychopy import visual, core, event, data
from psychopy.hardware import keyboard

from .utils import load_settings, launch_openface, MarkerOutlet, draw_red_frame


def run_experiment():
    settings = load_settings()

    # Parameters from settings.json (with defaults)
    ex = settings.get('experiment_params', {})
    n_trials = int(ex.get('n_trials', 40))
    n_sample_trials = int(ex.get('n_sample_trials', 4))
    frame_probability = float(ex.get('frame_probability', 0.3))
    fixation_duration = float(ex.get('fixation_duration', 1.0))
    image_duration = float(ex.get('image_duration', 4.0))
    emotion_duration = float(ex.get('emotion_duration', 5.0))
    iti_duration = float(ex.get('iti_duration', 2.0))
    break_duration = float(ex.get('break_duration', 180.0))

    emotion_labels = [
        'amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness'
    ]

    # Image list: gather from a local folder `stimuli/images` (all files with typical image extensions)
    stim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stimuli', 'images'))
    if not os.path.exists(stim_dir):
        os.makedirs(stim_dir, exist_ok=True)
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    images = [os.path.join(stim_dir, f) for f in os.listdir(stim_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    images.sort()
    if len(images) < n_trials:
        print(f'Warning: Found only {len(images)} images; n_trials set to {len(images)}.')
        n_trials = len(images)

    # Participant/session metadata
    participant_id = data.getDateStr(format='%Y%m%d%H%M%S')
    session_id = 'S1'

    # Launch OpenFace
    of_proc, of_session_dir = launch_openface(settings, participant_id, session_id)

    # LSL marker stream
    outlet = MarkerOutlet(settings.get('lsl_stream_name', 'psychopy_markers'), settings.get('lsl_stream_type', 'Markers'))

    # Window
    win = visual.Window(fullscr=True, color=[0, 0, 0], units='pix', allowGUI=False)
    kb = keyboard.Keyboard()
    default_clock = core.Clock()

    # Preload image stimuli
    image_stims = {path: visual.ImageStim(win, image=path, size=None, units='pix') for path in images}
    fixation = visual.TextStim(win, text='+', color=[1, 1, 1], height=40)
    feedback_text = visual.TextStim(win, text='', color=[1, 1, 1], height=28, wrapWidth=1200)
    instruction = visual.TextStim(win, text=(
        'Welcome to the experiment\n\n'
        'In each trial, you will see an image presented on the screen for 3 seconds.\n'
        'Some images will have a red frame.\n'
        'Press SPACE if you see the red frame while the image is displayed; do not press otherwise.\n\n'
        'After each image, you will have 3 seconds to select which emotion best describes your reaction:\n'
        'amusement, awe, contentment, excitement, anger, disgust, fear, sadness.\n\n'
        'There will be a short break halfway through.\n\n'
        'Press SPACE to begin practice.'
    ), color=[1, 1, 1], height=28, wrapWidth=1400)
    thankyou = visual.TextStim(win, text='Thank you for participating!', color=[1, 1, 1], height=32)

    # Valence rating visuals (1 = most negative, 4 = neutral, 7 = most positive)
    bar_width = 900
    bar_height = 18
    bar_y = -100
    rating_bar = visual.Rect(win, width=bar_width, height=bar_height, lineColor=[1, 1, 1], fillColor=[0.2, 0.2, 0.2], pos=(0, bar_y), units='pix')
    neg_label = visual.TextStim(win, text='- Negative', color=[1, -1, -1], height=24, pos=(-(bar_width/2) - 120, bar_y))
    pos_label = visual.TextStim(win, text='+ Positive', color=[-1, 1, -1], height=24, pos=((bar_width/2) + 110, bar_y))
    neg_emoji = visual.TextStim(win, text='☹', color=[1, 1, 1], height=40, pos=(-(bar_width/2), bar_y + 45))
    pos_emoji = visual.TextStim(win, text='☺', color=[1, 1, 1], height=40, pos=((bar_width/2), bar_y + 45))
    ticks = []
    step = bar_width / 6.0
    for i in range(7):
        x = -bar_width/2 + i * step
        ticks.append(visual.TextStim(win, text=str(i+1), color=[1, 1, 1], height=22, pos=(x, bar_y - 40)))
    rating_instr = visual.TextStim(win, text='How do you feel? (1 = most negative, 4 = neutral, 7 = most positive)', color=[1, 1, 1], height=24, pos=(0, bar_y + 90), wrapWidth=1600)

    # Show instructions
    instruction.draw()
    win.flip()
    event.clearEvents()
    event.waitKeys(keyList=['space'])

    # Prepare trials
    all_indices = list(range(n_trials))
    random.shuffle(all_indices)
    practice_indices = all_indices[:n_sample_trials]
    main_indices = all_indices[n_sample_trials:]

    # Prepare CSV logging
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f'behavior_{participant_id}_{session_id}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        headers = [
            'trial_index', 'block', 'image_id', 'frame_present', 'response', 'reaction_time', 'correct', 'valence_rating', 'valence_rt', 'feedback',
            'timestamp_trial_start', 'timestamp_img_onset', 'timestamp_response', 'timestamp_emotion', 'timestamp_trial_end'
        ]
        f.write(','.join(headers) + '\n')

    def run_block(block_name: str, trial_indices: list, give_feedback: bool):
        nonlocal frame_probability
        trials_completed = 0
        break_inserted = False
        for i, idx in enumerate(trial_indices):
            # Break at halfway point for main block
            if block_name == 'main' and not break_inserted and i == max(1, len(trial_indices)//2):
                outlet.push('BREAK_START')
                # Show countdown
                end_time = core.getTime() + break_duration
                while True:
                    remaining = end_time - core.getTime()
                    if remaining <= 0:
                        break
                    mins = int(remaining // 60)
                    secs = int(remaining % 60)
                    text = f'Break: {mins:01d}:{secs:02d} remaining\nPress ESC to skip.'
                    feedback_text.text = text
                    feedback_text.draw()
                    win.flip()
                    if event.getKeys(keyList=['escape']):
                        break
                    core.wait(0.2)
                outlet.push('BREAK_END')
                break_inserted = True

            image_path = images[idx]
            image_id = os.path.basename(image_path)
            frame_present = 1 if random.random() < frame_probability else 0

            # Trial start marker
            trial_start_ts = outlet.push('TRIAL_START')
            default_clock.reset()

            # Fixation
            fixation_on = core.getTime()
            fixation.draw()
            win.flip()
            outlet.push('FIXATION_ONSET')
            core.wait(fixation_duration)

            # Image presentation with optional red frame
            kb.clearEvents()
            event.clearEvents()
            img = image_stims[image_path]
            img.draw()
            frame_rects = []
            if frame_present:
                frame_rects = draw_red_frame(win, img)
                for r in frame_rects:
                    r.draw()
            img_onset_ts = outlet.push(f'IMG_ONSET:{image_id}')
            outlet.push('FRAME_PRESENT' if frame_present else 'FRAME_ABSENT')
            win.flip()

            # Reaction window during image
            response_key = None
            response_ts = ''
            rt = ''
            image_phase_clock = core.Clock()
            while image_phase_clock.getTime() < image_duration:
                keys = kb.getKeys(keyList=['space', 'escape'], waitRelease=False, clear=False)
                for k in keys:
                    if k.name == 'escape':
                        outlet.push('EXPT_ABORT')
                        if of_proc:
                            try:
                                of_proc.terminate()
                            except Exception:
                                pass
                        win.close()
                        core.quit()
                    if response_key is None and k.name == 'space':
                        response_key = 'space'
                        response_ts = outlet.push('KEYPRESS:space')
                        rt = k.rt
                core.wait(0.005)

            # Remove image
            win.flip()

            # Compute correctness
            responded = 1 if response_key == 'space' else 0
            correct = 1 if ((frame_present == 1 and responded == 1) or (frame_present == 0 and responded == 0)) else 0
            outlet.push('CORRECT' if correct else 'INCORRECT')

            # Optional feedback for practice
            feedback_msg = ''
            if give_feedback:
                if frame_present == 1 and responded == 1:
                    feedback_msg = 'Correct – You pressed SPACE when the red frame was present.'
                elif frame_present == 1 and responded == 0:
                    feedback_msg = 'Incorrect – Please press SPACE when a red frame is present.'
                elif frame_present == 0 and responded == 1:
                    feedback_msg = 'Incorrect – Do not press when there is no red frame.'
                else:
                    feedback_msg = 'Correct – You did not press when no frame was present.'
                feedback_text.text = feedback_msg
                feedback_text.draw()
                win.flip()
                core.wait(2.0)

            # Valence rating window (duration from visual onset)
            valence_choice = ''
            valence_rt = ''
            # Draw rating visuals once; keep on screen for the entire window
            rating_instr.draw()
            rating_bar.draw()
            neg_label.draw(); pos_label.draw()
            neg_emoji.draw(); pos_emoji.draw()
            for t in ticks:
                t.draw()
            prompt_onset_visual_ts = win.flip()
            kb.clock.reset()
            outlet.push('EMO_PROMPT_ONSET')
            emo_end_time = prompt_onset_visual_ts + emotion_duration
            choice_made = False
            while core.getTime() < emo_end_time:
                keys = kb.getKeys(keyList=['1','2','3','4','5','6','7','escape'], waitRelease=False, clear=False)
                if keys:
                    k = keys[-1]
                    if k.name == 'escape':
                        outlet.push('EXPT_ABORT')
                        if of_proc:
                            try:
                                of_proc.terminate()
                            except Exception:
                                pass
                        win.close()
                        core.quit()
                    if not choice_made and k.name in [str(i) for i in range(1, 8)]:
                        valence_choice = k.name
                        valence_rt = k.rt
                        outlet.push(f'VALENCE_RATING:{valence_choice}')
                        choice_made = True
                core.wait(0.005)

            # ITI
            win.flip()
            core.wait(iti_duration)

            trial_end_ts = outlet.push('TRIAL_END')

            # Write CSV row
            with open(csv_path, 'a', encoding='utf-8') as f:
                row = [
                    str(i), block_name, image_id, str(frame_present), str(responded), str(rt), str(correct), valence_choice, str(valence_rt), feedback_msg,
                    str(trial_start_ts), str(img_onset_ts), str(response_ts), str(prompt_onset_visual_ts), str(trial_end_ts)
                ]
                f.write(','.join(row) + '\n')

        return True

    # Practice block
    outlet.push('BLOCK_START:practice')
    run_block('practice', practice_indices, give_feedback=True)
    outlet.push('BLOCK_END:practice')

    # Instructions to start main block
    instruction.text = 'Practice complete. Press SPACE to begin the main block.'
    instruction.draw()
    win.flip()
    event.clearEvents()
    event.waitKeys(keyList=['space'])

    # Main block
    outlet.push('BLOCK_START:main')
    run_block('main', main_indices, give_feedback=False)
    outlet.push('BLOCK_END:main')

    outlet.push('EXPT_END')
    thankyou.draw()
    win.flip()
    core.wait(2.0)
    win.close()

    # Stop OpenFace if running
    if of_proc:
        try:
            of_proc.terminate()
        except Exception:
            pass


if __name__ == '__main__':
    run_experiment()


