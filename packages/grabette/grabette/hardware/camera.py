"""Video capture using picamera2 with H.264 encoding.

Ported from grabette-capture/grabette_capture/video.py.
"""

import gc
import io
import logging
import subprocess
from pathlib import Path

from .sync import SyncManager
from .recording_buffer import RecordingBuffer, VIDEO_BUFFER_BYTES

logger = logging.getLogger(__name__)


def _buffered_output(path, pts):
    # Import only on the Pi; FileOutput retains its keyframe gate and PTS format.
    from picamera2.outputs import FileOutput

    class BufferedOutput(FileOutput):
        def __init__(self):
            super().__init__(str(path), pts=pts)
            self._stopped = False
            self.buffer = RecordingBuffer("wrist", VIDEO_BUFFER_BYTES, self._save, self.close)

        def _write(self, frame, timestamp=None):
            data = bytes(frame)  # encoder buffers may be reused on return
            self.buffer.submit((data, timestamp), len(data))

        def _save(self, payload):
            frame, timestamp = payload
            if self.fileoutput.write(frame) != len(frame):
                raise OSError("Short wrist video write")
            self.fileoutput.flush()
            self.outputtimestamp(timestamp)  # only successfully saved frames

        def stop(self):
            if not self._stopped:
                self.recording = False
                self.buffer.close()  # finish callback closes FileOutput, reporting errors
                self._stopped = True

    return BufferedOutput()


class VideoCapture:
    """Captures video from CSI camera using picamera2.

    Default configuration:
        - Resolution: 1296x972 (native OV5647 binned mode)
        - Frame rate: 50 fps (CFR)
        - Codec: H.264 at ~5 Mbps
    """

    DEFAULT_RESOLUTION = (1296, 972)
    DEFAULT_FPS = 50
    DEFAULT_BITRATE = 5_000_000

    def __init__(
        self,
        sync_manager: SyncManager,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        fps: int = DEFAULT_FPS,
        bitrate: int = DEFAULT_BITRATE,
        preview: bool = False,
    ):
        self.sync = sync_manager
        self.resolution = resolution
        self.fps = fps
        self.bitrate = bitrate
        self.preview = preview

        self._picam2 = None
        self._encoder = None
        self._output_path: Path | None = None
        self._h264_path: Path | None = None
        self._frame_timestamps: list[float] = []
        self._encoder_pts = io.StringIO()
        self._recording = False
        self._frame_count: int = 0
        self._buffered_output = None
        self.buffer_stats: dict = {}

    def init_camera(self) -> None:
        """Initialize picamera2 with CFR configuration."""
        from picamera2 import Picamera2, Preview
        from picamera2.encoders import H264Encoder

        self._picam2 = Picamera2()
        frame_duration_us = int(1_000_000 / self.fps)

        if self.preview:
            video_config = self._picam2.create_video_configuration(
                main={"size": self.resolution, "format": "YUV420"},
                lores={"size": (640, 480), "format": "YUV420"},
                display="lores",
                controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            )
        else:
            video_config = self._picam2.create_video_configuration(
                main={"size": self.resolution, "format": "YUV420"},
                controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            )
        self._picam2.configure(video_config)
        self._encoder = H264Encoder(bitrate=self.bitrate)

        if self.preview:
            try:
                self._picam2.start_preview(Preview.QTGL)
            except Exception:
                try:
                    self._picam2.start_preview(Preview.DRM)
                except Exception:
                    logger.warning("Could not start preview")

        self._picam2.start()

    def start_recording(self, output_path: Path) -> None:
        if self._recording:
            raise RuntimeError("Video capture already running")
        if self._picam2 is None:
            raise RuntimeError("Camera not initialized. Call init_camera() first.")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before video capture")

        self._output_path = Path(output_path)
        self._h264_path = self._output_path.with_suffix(".h264")
        self._frame_timestamps = []
        self._frame_count = 0
        self._encoder_pts = io.StringIO()
        self.buffer_stats = {}
        self._buffered_output = _buffered_output(self._h264_path, self._encoder_pts)

        self._recording = True
        gc.disable()  # Prevent GC pauses from dropping frames during recording
        try:
            self._picam2.start_encoder(self._encoder, self._buffered_output)
        except Exception:
            self._recording = False
            gc.enable()
            self._buffered_output.stop()
            raise

    def stop(self) -> list[float]:
        if not self._recording:
            return self._frame_timestamps

        self._recording = False
        try:
            self._picam2.stop_encoder()  # output.stop drains before closing file
        finally:
            gc.enable()
            # Also drain if encoder teardown itself raised before output.stop.
            self._buffered_output.stop()
            self.buffer_stats = {"wrist": self._buffered_output.buffer.stats()}
        # FileOutput writes PTS only for saved frames, including the stop drain.
        # PTS is milliseconds relative to the encoder's first SensorTimestamp
        # (microseconds, CLOCK_BOOTTIME), not relative to recording start.
        first_timestamp = self._encoder.firsttimestamp
        if self.preview:
            try:
                self._picam2.stop_preview()
            except Exception:
                pass
        self._picam2.stop()
        self._picam2.close()
        self._picam2 = None
        self._encoder = None

        if first_timestamp is None:
            raise RuntimeError("Camera encoder produced no timestamped frames")
        origin_ms = self.sync.boottime_ns_to_ms(first_timestamp * 1000)
        self._frame_timestamps = [
            origin_ms + float(pts) for pts in self._encoder_pts.getvalue().splitlines()
        ]
        if self.buffer_stats["wrist"]["complete"]:
            self._mux_to_mp4()
        else:
            logger.error("Incomplete wrist recording; preserving raw H.264")
        return self._frame_timestamps

    def _mux_to_mp4(self) -> None:
        if self._h264_path is None or self._output_path is None:
            return
        if not self._h264_path.exists():
            raise RuntimeError(f"H.264 file not found: {self._h264_path}")

        actual_fps = self.fps
        if len(self._frame_timestamps) >= 2:
            duration_ms = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if duration_ms > 0:
                actual_fps = (len(self._frame_timestamps) - 1) / (duration_ms / 1000.0)

        cmd = [
            "ffmpeg", "-y", "-fflags", "+genpts",
            "-r", str(actual_fps), "-i", str(self._h264_path),
            "-c", "copy", "-video_track_timescale", "90000",
            str(self._output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg muxing failed: {result.stderr}")
        self._frame_count = self._count_frames_ffprobe()
        if not self._frame_count or self._frame_count != len(self._frame_timestamps):
            raise RuntimeError(
                f"Camera video/timestamp mismatch: {self._frame_count} frames, "
                f"{len(self._frame_timestamps)} timestamps; keeping raw H.264"
            )
        self._h264_path.unlink()

    def _count_frames_ffprobe(self) -> int:
        if self._output_path is None or not self._output_path.exists():
            return 0
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-count_packets",
                    "-show_entries", "stream=nb_read_packets",
                    "-of", "csv=p=0",
                    str(self._output_path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            val = result.stdout.strip()
            return int(val) if val.isdigit() else 0
        except Exception:
            return 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_open(self) -> bool:
        """True while the picamera2 device is initialized (not yet closed)."""
        return self._picam2 is not None
