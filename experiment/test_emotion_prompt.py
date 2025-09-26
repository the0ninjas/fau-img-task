from psychopy import visual, core
from psychopy.hardware import keyboard


def main():
    emotions = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']
    win = visual.Window(fullscr=False, size=(800, 600), color=[0, 0, 0], units='pix')
    prompt = visual.TextStim(win, text=(
        'Press a number (1-8) to choose an emotion:\n\n'
        '1) amusement   2) awe   3) contentment   4) excitement\n'
        '5) anger       6) disgust   7) fear   8) sadness\n\n'
        'Press ESC to quit.'
    ), color=[1, 1, 1], height=24, wrapWidth=1000)

    kb = keyboard.Keyboard()
    while True:
        prompt.draw(); win.flip()
        keys = kb.getKeys(keyList=[str(i) for i in range(1, 9)] + ['escape'], waitRelease=False)
        if keys:
            k = keys[-1]
            if k.name == 'escape':
                break
            if k.name in [str(i) for i in range(1, 9)]:
                idx = int(k.name) - 1
                print(f'Chose {k.name}: {emotions[idx]} (rt={k.rt:.3f}s)')
                core.wait(0.5)

    win.close()


if __name__ == '__main__':
    main()


