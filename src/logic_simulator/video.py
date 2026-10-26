from fractions import Fraction
import av
from .contracts import FramePacket


def read_video(path,start_ms=0,end_ms=None):
    with av.open(str(path)) as container:
        origin=None
        first=None
        previous=None
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                raise ValueError('Source frame has no PTS')
            pts=frame.pts*frame.time_base
            if origin is None:
                origin=pts
            original=pts-origin
            if previous is not None and original<=previous:
                raise ValueError('Source PTS must increase strictly')
            previous=original
            if original*1000 < start_ms:
                continue
            if end_ms is not None and original*1000 >= end_ms:
                break
            if first is None:
                first=original
            yield FramePacket(index,original,original-first,frame.to_ndarray(format='bgr24'))


class VideoWriter:
    def __init__(self,path,width,height,rate=30):
        self.container=av.open(str(path),'w')
        self.stream=self.container.add_stream('libx264',rate=rate)
        # yuv420p requires even dimensions; source pixels stay untouched in logic.
        self.stream.width=width+(width%2)
        self.stream.height=height+(height%2)
        self.stream.pix_fmt='yuv420p'
        self.stream.time_base=Fraction(1,90000)
        self.stream.codec_context.time_base=Fraction(1,90000)
        self.stream.options={'crf':'23','preset':'fast'}

    def write(self,image,time):
        frame=av.VideoFrame.from_ndarray(image,format='bgr24')
        frame=frame.reformat(width=self.stream.width,height=self.stream.height,format='yuv420p')
        frame.pts=round(time*90000)
        frame.time_base=Fraction(1,90000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
