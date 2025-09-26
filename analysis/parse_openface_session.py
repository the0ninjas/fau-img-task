import os
import sys
import csv
import json
import re
from datetime import datetime


def load_parsed_csv(session_dir: str):
    path = os.path.join(session_dir, 'openface_parsed.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_payload_text(channel: str, payload: str):
    # Try JSON first
    try:
        data = json.loads(payload)
        return data
    except Exception:
        pass
    # Fallback: extract numbers and simple lists
    nums = re.findall(r'-?\d+\.\d+|-?\d+', payload)
    if channel == 'emotion' and nums:
        return {f'e{i}': float(v) for i, v in enumerate(nums)}
    if channel == 'gaze' and nums:
        return {f'g{i}': float(v) for i, v in enumerate(nums)}
    if channel == 'au' and nums:
        return {f'au{i}': float(v) for i, v in enumerate(nums)}
    return {'raw': payload}


def write_stream_csv(session_dir: str, rows: list):
    out_dir = os.path.join(session_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'stream.csv')

    # Build merged fieldnames
    fieldnames = ['timestamp_iso', 't_rel_s', 'channel']
    # Discover keys from parsed payloads
    parsed_rows = []
    t0 = None
    for r in rows:
        ts_iso = r.get('timestamp_iso')
        try:
            ts = datetime.fromisoformat(ts_iso)
        except Exception:
            ts = None
        if t0 is None and ts is not None:
            t0 = ts
        payload = parse_payload_text(r.get('channel', ''), r.get('payload', ''))
        parsed_rows.append((ts, r.get('channel', ''), payload, ts_iso))
        for k in payload.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ts, channel, payload, ts_iso in parsed_rows:
            t_rel = (ts - t0).total_seconds() if (ts and t0) else ''
            row = {'timestamp_iso': ts_iso, 't_rel_s': t_rel, 'channel': channel}
            row.update(payload)
            writer.writerow(row)

    return out_path


def print_brief_summary(rows: list):
    counts = {}
    for r in rows:
        ch = r.get('channel', '')
        counts[ch] = counts.get(ch, 0) + 1
    print('Summary:')
    for ch, n in sorted(counts.items()):
        print(f'  {ch}: {n} events')


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m analysis.parse_openface_session <SESSION_DIR>')
        print('Example:')
        print('  python -m analysis.parse_openface_session C:\\Users\\ARCLP\\Documents\\Code\\experiment\\openface_out\\TEST_OFACE_20250926_122903')
        sys.exit(1)

    session_dir = sys.argv[1]
    if not os.path.isdir(session_dir):
        print(f'Not a directory: {session_dir}')
        sys.exit(1)

    try:
        rows = load_parsed_csv(session_dir)
    except Exception as e:
        print(f'Failed to load parsed CSV: {e}')
        sys.exit(1)

    out_path = write_stream_csv(session_dir, rows)
    print_brief_summary(rows)
    print(f'Wrote merged stream CSV: {out_path}')


if __name__ == '__main__':
    main()


