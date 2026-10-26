"""Length-prefixed JSON packets; images are PNG base64, never pickle."""
import json
import select
import struct
import time

MAX_PACKET=64*1024*1024


def wait_ready(stream,deadline,writing=False):
    if deadline is None:
        return
    remaining=deadline-time.monotonic()
    if remaining<=0:
        raise TimeoutError('Inference worker timeout')
    if hasattr(stream,'channel'):
        stream.channel.settimeout(remaining)
    else:
        readable,writable,_=select.select([] if writing else [stream],[stream] if writing else [],[],remaining)
        if not (writable if writing else readable):
            raise TimeoutError('Inference worker timeout')


def send(stream,value,timeout=None):
    payload=json.dumps(value,separators=(',',':')).encode()
    if len(payload)>MAX_PACKET:
        raise ValueError('Packet too large')
    packet=memoryview(struct.pack('!I',len(payload))+payload)
    deadline=None if timeout is None else time.monotonic()+timeout
    while packet:
        wait_ready(stream,deadline,writing=True)
        # <= PIPE_BUF: a writable unbuffered local pipe cannot block on a large write.
        chunk=packet[:4096]
        written=stream.write(chunk)
        # Paramiko BufferedFile.write returns None after writing all supplied bytes.
        written=len(chunk) if written is None else written
        if written<=0:
            raise EOFError('Inference worker stopped accepting data')
        packet=packet[written:]
    stream.flush()


def receive(stream,timeout=None):
    deadline=None if timeout is None else time.monotonic()+timeout
    def exact(size):
        chunks=[]
        while size:
            wait_ready(stream,deadline)
            block=stream.read(size)
            if not block:
                raise EOFError('Inference worker disconnected/truncated packet')
            chunks.append(block)
            size-=len(block)
        return b''.join(chunks)
    size=struct.unpack('!I',exact(4))[0]
    if size>MAX_PACKET:
        raise ValueError('Packet too large')
    return json.loads(exact(size))
