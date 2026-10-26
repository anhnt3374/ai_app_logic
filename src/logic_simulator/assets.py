import json
import os
import shutil
import urllib.request
from pathlib import Path
from fractions import Fraction
import av
import cv2
import numpy as np
import yaml

from .inference.transport import WorkerClient
from .run_store import file_hash
from .config import TargetConfig


def fetch_models(registry_path='configs/models.yaml'):
    registry=yaml.safe_load(Path(registry_path).read_text())
    for name,manifest in registry.items():
        path=Path(manifest['path'])
        if path.exists() and file_hash(path)==manifest['sha256']:
            print(f'{name}: checksum OK')
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.part')
        try:
            with urllib.request.urlopen(manifest['url'],timeout=120) as source,temporary.open('wb') as output:
                shutil.copyfileobj(source,output)
            if file_hash(temporary)!=manifest['sha256']:
                raise ValueError(f'Model checksum mismatch: {name}')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'{name}: downloaded and verified')


def make_demo(path='data/input/demo.mp4',image_path=None,frames=90):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(image_path)) if image_path else None
    if image_path and image is None:
        raise ValueError('Cannot decode demo image')
    h,w=image.shape[:2] if image is not None else (480,640)
    with av.open(str(path),'w') as out:
        stream=out.add_stream('libx264',rate=30)
        stream.width,stream.height=w+(w%2),h+(h%2)
        stream.pix_fmt='yuv420p'
        for index in range(frames):
            if image is None:
                pixels=np.zeros((h,w,3),np.uint8)
                pixels[:,:,1]=30+(index%90)
                for x,color in [(32,(50,180,230)),(352,(180,100,50))]:
                    cv2.rectangle(pixels,(x,48),(x+255,455),color,-1)
                    cv2.circle(pixels,(x+127,130),45,(180,200,220),-1)
                cv2.putText(pixels,str(index),(260,460),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            else:
                pixels=cv2.warpAffine(image,np.float32([[1,0,(index%5)-2],[0,1,0]]),(w,h),borderMode=cv2.BORDER_REFLECT)
            frame=av.VideoFrame.from_ndarray(pixels,format='bgr24')
            frame.pts=index
            frame.time_base=Fraction(1,30)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def enroll(images,output,registry_path='configs/models.yaml',face_model='yunet',embedding_model='sface'):
    registry=yaml.safe_load(Path(registry_path).read_text())
    people=[]
    with_worker=WorkerClient(TargetConfig())
    from uuid import uuid4
    try:
        for folder in sorted(Path(images).iterdir()):
            if not folder.is_dir():
                continue
            metadata=json.loads((folder/'person.json').read_text()) if (folder/'person.json').exists() else {}
            embeddings=[]
            for photo in sorted(folder.iterdir()):
                if photo.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}:
                    continue
                pixels=cv2.imread(str(photo))
                if pixels is None:
                    raise ValueError('Cannot decode enrollment image: '+str(photo))
                request={'request_id':uuid4().hex,'manifest':registry[face_model],
                         'target':TargetConfig().model_dump(),'context':{}}
                faces=with_worker.call(request,pixels)['payload']['detections']
                if len(faces)!=1:
                    raise ValueError(f'Enrollment needs exactly one face: {photo}, found {len(faces)}')
                request.update(request_id=uuid4().hex,manifest=registry[embedding_model],context={'face':faces[0]})
                result=with_worker.call(request,pixels)['payload']
                embeddings.append(result['embedding'])
            if not embeddings:
                raise ValueError('No enrollment images: '+str(folder))
            people.append({'id':folder.name,'name':metadata.get('name',folder.name),
                           'blacklist':bool(metadata.get('blacklist',False)),'embeddings':embeddings})
        if not people:
            raise ValueError('No person folders found for enrollment')
        payload={'schema_version':1,'embedding_space':registry[embedding_model]['sha256'],
                 'model_id':embedding_model,'people':people}
        output=Path(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(payload,indent=2,ensure_ascii=False))
        return payload
    finally:
        with_worker.close()


def provision_models(target,registry_path='configs/models.yaml'):
    """Push checksummed artifacts to an explicitly configured board model root."""
    from pathlib import PurePosixPath
    import shlex
    if target.location!='board':
        raise ValueError('provision requires a board target')
    root=PurePosixPath(target.model_root)
    if not root.is_absolute() or '..' in root.parts or str(root)=='/':
        raise ValueError('model_root must be an absolute dedicated directory')
    registry=yaml.safe_load(Path(registry_path).read_text())
    client=WorkerClient(target)
    try:
        sftp=client.ssh.open_sftp()
        current=PurePosixPath('/')
        for part in root.parts[1:]:
            current=current/part
            try:
                sftp.stat(str(current))
            except FileNotFoundError:
                sftp.mkdir(str(current))
        results={}
        for name,manifest in registry.items():
            source=Path(manifest['path'])
            if file_hash(source)!=manifest['sha256']:
                raise ValueError('Local model checksum mismatch: '+name)
            destination=str(root/source.name)
            temporary=destination+'.upload'
            sftp.put(str(source),temporary)
            _,stdout,_=client.ssh.exec_command('sha256sum -- '+shlex.quote(temporary))
            remote_hash=stdout.read().decode().split()[0]
            if remote_hash!=manifest['sha256']:
                sftp.remove(temporary)
                raise ValueError('Remote model checksum mismatch: '+name)
            sftp.posix_rename(temporary,destination)
            results[name]=remote_hash
        sftp.close()
        return results
    finally:
        client.close()
