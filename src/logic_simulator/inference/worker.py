import base64
import sys
import cv2
import numpy as np
from pathlib import Path

from .adapters import Adapter
from .backends import select_backend
from .protocol import receive,send
from ..config import TargetConfig
from ..run_store import digest,file_hash


def main():
    cv2.setNumThreads(1)
    adapters={}
    while True:
        try:
            request=receive(sys.stdin.buffer)
        except EOFError:
            break
        try:
            target=TargetConfig.model_validate(request['target'])
            backend=select_backend(target)
            if request.get('command')=='probe':
                send(sys.stdout.buffer,{'request_id':request['request_id'],'backend':backend})
                continue
            manifest=request['manifest']
            key=digest([manifest,backend])
            if key not in adapters:
                if file_hash(manifest['path'])!=manifest['sha256']:
                    raise ValueError('Model checksum mismatch')
                adapters[key]=Adapter(manifest,backend)
            image=cv2.imdecode(np.frombuffer(base64.b64decode(request['image']),np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Invalid PNG image')
            payload=adapters[key].infer(image,request.get('context',{}))
            send(sys.stdout.buffer,{'request_id':request['request_id'],'payload':payload,'backend':backend})
        except Exception as error:
            # No connection secrets or raw input payloads in the diagnostic channel.
            send(sys.stdout.buffer,{'request_id':request.get('request_id'),'error':str(error)[:500]})


if __name__=='__main__':
    main()
