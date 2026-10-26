"""Normalized OpenCV Zoo adapters. See THIRD_PARTY.md for upstream references."""
import cv2
import numpy as np

from ..gallery import normalized

ADAPTER_VERSION='1'


def align_face(image,landmarks):
    source=np.asarray(landmarks,dtype=np.float64)
    target=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                     [41.5493,92.3655],[70.7299,92.2041]])
    src=source-source.mean(axis=0)
    dst=target-target.mean(axis=0)
    u,s,vt=np.linalg.svd(dst.T@src/5)
    signs=np.ones(2)
    if np.linalg.det(u)*np.linalg.det(vt)<0:
        signs[-1]=-1
    rotation=u@np.diag(signs)@vt
    variance=np.mean(np.sum(src**2,axis=1))
    if variance<=1e-10:
        raise ValueError('Degenerate face landmarks')
    scale=np.sum(s*signs)/variance
    matrix=np.column_stack((rotation*scale,target.mean(axis=0)-scale*rotation@source.mean(axis=0)))
    return cv2.warpAffine(image,matrix,(112,112))


def decode_yolox(outputs,scale,width,height,threshold=.35,nms=.45):
    dets=outputs[0].copy()
    grids=[]
    strides=[]
    for stride in [8,16,32]:
        y,x=np.mgrid[:640//stride,:640//stride]
        grids.append(np.stack([x,y],axis=-1).reshape(-1,2))
        strides.extend([stride]*(640//stride)**2)
    strides=np.asarray(strides)[:,None]
    centers=(dets[:,:2]+np.concatenate(grids))*strides
    sizes=np.exp(np.clip(dets[:,2:4],-20,20))*strides
    scores=dets[:,4]*dets[:,5]  # COCO class 0 = person
    mask=(scores>=threshold)&(np.argmax(dets[:,5:],axis=1)==0)
    boxes=np.concatenate([centers-sizes/2,sizes],axis=1)[mask]/scale
    confidence=scores[mask]
    indices=cv2.dnn.NMSBoxes(boxes.tolist(),confidence.tolist(),threshold,nms)
    results=[]
    for index in np.asarray(indices).reshape(-1):
        x,y,w,h=boxes[index]
        box=[float(np.clip(x,0,width)),float(np.clip(y,0,height)),
             float(np.clip(x+w,0,width)),float(np.clip(y+h,0,height))]
        if box[2]>box[0] and box[3]>box[1]:
            results.append({'bbox':box,'score':float(confidence[index]),'class':'person'})
    return results


class Adapter:
    def __init__(self,manifest,backend):
        self.manifest=manifest
        self.kind=manifest['adapter']
        self.backend=backend
        self.path=manifest['path']
        self.options=manifest.get('options',{})
        self.ov_model=None
        if backend['runtime']=='openvino':
            import openvino as ov
            self.ov_core=ov.Core()
            model=self.ov_core.read_model(self.path)
            self.ov_model=self.ov_core.compile_model(model,backend['provider'],backend.get('options',{}))
            self.session=None
            self.net=None
        elif backend['runtime']=='onnxruntime':
            import onnxruntime as ort
            options=ort.SessionOptions()
            options.intra_op_num_threads=1
            options.inter_op_num_threads=1
            options.log_severity_level=3
            if backend['device']!='cpu':
                options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
            self.session=ort.InferenceSession(self.path,sess_options=options,
                providers=[(backend['provider'],backend.get('options',{}))])
            if backend['provider'] not in self.session.get_providers():
                raise RuntimeError('Requested execution provider did not initialize')
            self.session.disable_fallback()
            self.net=None
        else:
            self.session=None
            if self.kind=='yunet':
                self.net=cv2.FaceDetectorYN.create(self.path,'',(320,320),
                    self.options.get('threshold',.7),.3,5000,backend['backend'],backend['target'])
            elif self.kind=='sface':
                self.net=cv2.FaceRecognizerSF.create(self.path,'',backend['backend'],backend['target'])
            elif self.kind=='yolox':
                self.net=cv2.dnn.readNet(self.path)
                self.net.setPreferableBackend(backend['backend'])
                self.net.setPreferableTarget(backend['target'])
            else:
                raise ValueError('Unknown adapter: '+self.kind)

    def forward(self,blob):
        if self.ov_model is not None:
            output=self.ov_model({0:blob})
            return [np.array(output[port]) for port in self.ov_model.outputs]
        if self.session:
            return self.session.run(None,{self.session.get_inputs()[0].name:blob})
        self.net.setInput(blob)
        return self.net.forward(self.net.getUnconnectedOutLayersNames())

    def infer(self,image,context):
        if self.kind=='yolox':
            h,w=image.shape[:2]
            ratio=min(640/h,640/w)
            resized=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(int(w*ratio),int(h*ratio)))
            padded=np.full((640,640,3),114,dtype=np.float32)
            padded[:resized.shape[0],:resized.shape[1]]=resized
            outputs=self.forward(padded.transpose(2,0,1)[None])
            return {'detections':decode_yolox(outputs[0],ratio,w,h,self.options.get('threshold',.35))}
        if self.kind=='yunet':
            h,w=image.shape[:2]
            if self.session or self.ov_model is not None:
                # YuNet 2023 has fixed 640x640 ORT input; newer exports may be dynamic.
                shape=list(self.ov_model.input(0).shape) if self.ov_model is not None else self.session.get_inputs()[0].shape
                ph=shape[2] if isinstance(shape[2],int) else (h+31)//32*32
                pw=shape[3] if isinstance(shape[3],int) else (w+31)//32*32
                ratio=min(1.,ph/h,pw/w)
                resized=cv2.resize(image,(max(1,int(w*ratio)),max(1,int(h*ratio))))
                padded=np.zeros((ph,pw,3),np.float32)
                padded[:resized.shape[0],:resized.shape[1]]=resized
                outputs=self.forward(padded.transpose(2,0,1)[None])
                if self.ov_model is not None:
                    tensors={name:output for port,output in zip(self.ov_model.outputs,outputs) for name in port.get_names()}
                else:
                    tensors=dict(zip([v.name for v in self.session.get_outputs()],outputs))
                rows=[]
                for stride in [8,16,32]:
                    cls=tensors[f'cls_{stride}'].reshape(-1)
                    obj=tensors[f'obj_{stride}'].reshape(-1)
                    boxes=tensors[f'bbox_{stride}'].reshape(-1,4)
                    kps=tensors[f'kps_{stride}'].reshape(-1,5,2)
                    gy,gx=np.mgrid[:ph//stride,:pw//stride]
                    grid=np.stack([gx,gy],axis=-1).reshape(-1,2)
                    scores=np.sqrt(np.clip(cls,0,1)*np.clip(obj,0,1))
                    for i in np.flatnonzero(scores>=self.options.get('threshold',.7)):
                        center=(grid[i]+boxes[i,:2])*stride
                        size=np.exp(boxes[i,2:])*stride
                        xy=center-size/2
                        landmarks=(grid[i]+kps[i])*stride
                        rows.append([*(xy/ratio),*(size/ratio),*(landmarks.reshape(-1)/ratio),scores[i]])
                if rows:
                    keep=cv2.dnn.NMSBoxes([r[:4] for r in rows],[r[-1] for r in rows],self.options.get('threshold',.7),.3)
                    faces=[rows[i] for i in np.asarray(keep).reshape(-1)]
                else:
                    faces=[]
            else:
                self.net.setInputSize((w,h))
                _,faces=self.net.detect(image)
                faces=[] if faces is None else faces
            return {'detections':[{'bbox':[float(r[0]),float(r[1]),float(r[0]+r[2]),float(r[1]+r[3])],
                'landmarks':np.asarray(r[4:14]).reshape(5,2).tolist(),'score':float(r[14])} for r in faces]}
        if self.kind=='sface':
            face=context['face']
            if self.session or self.ov_model is not None:
                aligned=align_face(image,face['landmarks'])
                blob=cv2.dnn.blobFromImage(aligned,1,(112,112),(0,0,0),swapRB=True)
                feature=self.forward(blob)[0]
            else:
                x1,y1,x2,y2=face['bbox']
                row=np.array([x1,y1,x2-x1,y2-y1,*np.asarray(face['landmarks']).reshape(-1),face['score']],dtype=np.float32)
                aligned=self.net.alignCrop(image,row)
                feature=self.net.feature(aligned)
            return {'embedding':normalized(feature).tolist(),'embedding_space':self.manifest['sha256']}
        raise ValueError('Unknown adapter: '+self.kind)
