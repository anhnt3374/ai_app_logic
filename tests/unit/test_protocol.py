import io
import pytest
from logic_simulator.inference.protocol import send, receive


def test_protocol_roundtrip_and_truncated_message():
    stream=io.BytesIO()
    send(stream,{'request_id':'a','items':[1,2]})
    stream.seek(0)
    assert receive(stream)['request_id']=='a'
    with pytest.raises(EOFError):
        receive(io.BytesIO(b'\x00\x00'))


def test_protocol_retries_partial_writes():
    class ShortWriter(io.BytesIO):
        def write(self,value):
            return super().write(value[:2])
    output=ShortWriter()
    send(output,{'value':'this payload needs many writes'})
    assert receive(io.BytesIO(output.getvalue()))=={'value':'this payload needs many writes'}


def test_timeout_applies_while_worker_is_not_reading():
    import subprocess
    import sys
    import time
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'],stdin=subprocess.PIPE,bufsize=0)
    start=time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            send(process.stdin,{'payload':'x'*(2*1024*1024)},timeout=.05)
        assert time.monotonic()-start<1
    finally:
        process.terminate()
        process.wait()
        process.stdin.close()
