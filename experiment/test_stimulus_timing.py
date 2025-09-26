from psychopy import visual, core, event


def main():
    fixation_duration = 1.0
    image_duration = 2.0
    iti_duration = 1.0

    win = visual.Window(fullscr=False, size=(800, 600), color=[0, 0, 0], units='pix')
    fixation = visual.TextStim(win, text='+', color=[1, 1, 1], height=40)
    stim = visual.TextStim(win, text='IMAGE', color=[1, 1, 1], height=40)

    clock = core.Clock()

    print('Press any key to start timing test (ESC to quit)')
    win.flip()
    event.waitKeys()

    for t in range(5):
        clock.reset()
        fixation.draw(); win.flip(); core.wait(fixation_duration)
        stim.draw(); win.flip(); core.wait(image_duration)
        win.flip(); core.wait(iti_duration)
        print(f'Trial {t+1} total: {clock.getTime():.3f}s')

    print('Done. Press any key to exit.')
    win.flip(); event.waitKeys()
    win.close()


if __name__ == '__main__':
    main()


