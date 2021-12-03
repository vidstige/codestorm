from pathlib import Path
import subprocess
from typing import BinaryIO, List, Optional, Union, Sequence, Tuple


class Resolution:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
    
    @staticmethod
    def parse(s: str) -> 'Resolution':
        parts = s.split('x')
        if len(parts) != 2:
            raise ValueError('Invalid Resolution: {}'.format(s))
        return Resolution(int(parts[0]), int(parts[1]))

    def pair(self) -> Tuple[int, int]:
        return self.width, self.height
    
    def aspect(self) -> float:
        return self.width / self.height

    def __str__(self) -> str:
        return '{}x{}'.format(self.width, self.height)


class VideoFormat:
    FORMAT = ''
    def arguments(self) -> List[str]:
        pass


class Raw(VideoFormat):
    FORMAT = 'rawvideo'
    def __init__(self, pixel_format: str, video_size: Resolution, framerate: Optional[float]=None):
        self.pixel_format = pixel_format
        self.video_size = video_size
        self.framerate = framerate

    def arguments(self) -> List[str]:
        #-pixel_format rgb32 -video_size $VIDEO_SIZE -framerate 30
        args = ['-pixel_format', self.pixel_format, '-video_size', str(self.video_size)]
        if self.framerate:
            args += ['-framerate', str(self.framerate)]
        return args


class H264(VideoFormat):
    FORMAT = 'h264'
    def __init__(self, pixel_format: str, crf: int, preset: Optional[str]=None):
        self.pixel_format = pixel_format
        self.crf = crf
        self.preset = preset

    def arguments(self) -> List[str]:
        args = ['-c:v', 'libx264']
        if self.preset:
            args += ['-preset', self.preset]
        args += [
            '-crf', str(self.crf),
            '-pix_fmt', self.pixel_format
        ]
        return args


class Session:
    def __init__(self, command: Sequence[Union[str, Path]]) -> None:
        self.process = None
        self.command = command
    
    @property
    def buffer(self) -> BinaryIO:
        if self.process:
            return self.process.stdin
        raise RuntimeError('ffmpeg not runnig')

    def __enter__(self):
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.process.stdin.flush()
        self.process.stdin.close()
        self.process.wait()


class FFmpeg:
    def __init__(self, executable: str='ffmpeg'):
        self.executable = executable
    
    def convert(
            self, video_format: VideoFormat,
            target: Optional[Path]=None,
            target_format: Optional[VideoFormat]=None) -> Session:
        #ffmpeg -f rawvideo -pixel_format rgb32 -video_size $VIDEO_SIZE -framerate 30 -i bw.raw -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p bw.mkv

        command = [self.executable, '-f', video_format.FORMAT] + video_format.arguments() + ['-i', '-']
        if target:
            if target_format:
                command += target_format.arguments()
            command += [str(target)]
        return Session(command)

#
#with FFmpeg().convert('x.mkv', Raw('rgb32', Resolution.parse('320x200'), 30)) as session:
#    print(session.buffer)
