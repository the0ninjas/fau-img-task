from psychopy import core

from .utils import load_settings, MarkerOutlet


def main():
    settings = load_settings()
    outlet = MarkerOutlet(settings.get('lsl_stream_name', 'psychopy_markers'), settings.get('lsl_stream_type', 'Markers'))
    print('Sending 10 markers (0.5s interval)...')
    for i in range(10):
        ts = outlet.push(f'TEST_LSL:{i}')
        print(f'Sent TEST_LSL:{i} at {ts:.4f}')
        core.wait(0.5)
    print('Done.')


if __name__ == '__main__':
    main()


