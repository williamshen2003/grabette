"""A missing depth sample must not shift the remaining camera/depth pairs."""
import json

import pytest

from grabette_postprocess import convert


def episode(tmp_path, monkeypatch, *, depth_count=2):
    left = [{'seq': i, 'host_ms': i * 20} for i in (10, 11, 12)]
    depth = [left[0], left[2]]
    for name, data in [('dcam_left_timestamps.json', {'samples': left}),
                       ('dcam_depth_timestamps.json', {'samples': depth}),
                       ('dcam_calib_offline.json', {})]:
        (tmp_path / name).write_text(json.dumps(data))
    for name in ('dcam_left.mp4', 'dcam_depth.mkv'):
        (tmp_path / name).touch()

    def extract(count, label):
        def run(path, out):
            out.mkdir(parents=True)
            for i in range(count):
                (out / f'{i:06d}.png').write_text(f'{label}{i}')
            return count
        return run
    monkeypatch.setattr(convert, '_extract_mp4_frames', extract(3, 'left'))
    monkeypatch.setattr(convert, '_extract_depth_video', extract(depth_count, 'depth'))
    return tmp_path


def test_depth_gap_keeps_original_camera_index_and_elapsed_time(tmp_path, monkeypatch):
    out = convert.convert_episode(episode(tmp_path, monkeypatch))
    assert (out / 'frames/000001.png').read_text() == 'left2'
    assert (out / 'depth/000001.png').read_text() == 'depth1'
    assert (out / 'timestamps.csv').read_text().splitlines() == [
        'idx,timestamp_ns', '0,200000000', '1,240000000']
    report = json.loads((out / 'conversion_report.json').read_text())
    assert report['unmatched_left_sequences'] == [11]


@pytest.mark.parametrize('depth_count', [1, 3])
def test_depth_count_mismatch_is_rejected_before_partial_output(tmp_path, monkeypatch, depth_count):
    ep = episode(tmp_path, monkeypatch, depth_count=depth_count)
    with pytest.raises(ValueError, match='depth.*frames.*timestamps'):
        convert.convert_episode(ep)
    assert not (ep / 'oak').exists()


def test_incomplete_conversion_is_never_reused(tmp_path, monkeypatch):
    ep = episode(tmp_path, monkeypatch)
    (ep / 'oak').mkdir()
    with pytest.raises(ValueError, match='force'):
        convert.convert_episode(ep)


def test_video_extraction_does_not_insert_frames_across_timestamp_gaps(tmp_path):
    import av
    import numpy as np
    from fractions import Fraction
    source = tmp_path / 'left.mp4'
    with av.open(str(source), 'w') as output:
        stream = output.add_stream('libx264', rate=50)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        stream.options = {'bf': '0'}
        for i, pts in enumerate([0, 1, 2, 3, 4, 8, 9, 10, 11, 12]):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 20, np.uint8), format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 50)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    assert convert._extract_mp4_frames(source, tmp_path / 'decoded') == 10
