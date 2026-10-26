import numpy as np


class MockProvider:
    def __init__(self,scenario,seed):
        self.scenario=scenario
        self.seed=seed

    def infer(self,request,image):
        time=request['original_ms']
        people=[p for p in self.scenario['people'] if p.get('start_ms',0)<=time<p.get('end_ms',float('inf'))]
        if request['stage']=='person':
            h,w=image.shape[:2]
            return {'detections':[{'bbox':[p['bbox'][0]*w,p['bbox'][1]*h,p['bbox'][2]*w,p['bbox'][3]*h],
                    'score':1.,'class':'person'} for p in people]}
        if request['stage']=='face':
            h,w=image.shape[:2]
            if not people:
                return {'detections':[]}
            # Select scenario person by overlap with parent person bbox (normalized).
            parent=request['context']['parent_normalized']
            from ..tracking import iou
            person=max(people,key=lambda p:iou(p['bbox'],parent))
            if not person.get('face_visible',True):
                return {'detections':[]}
            box=person.get('face_bbox',[.25,.05,.75,.4])
            x1,y1,x2,y2=box[0]*w,box[1]*h,box[2]*w,box[3]*h
            landmarks=[[x1+(x2-x1)*a,y1+(y2-y1)*b] for a,b in [(.3,.35),(.7,.35),(.5,.55),(.35,.75),(.65,.75)]]
            return {'detections':[{'bbox':[x1,y1,x2,y2],'score':1.,'landmarks':landmarks,
                                   'mock_embedding':person['embedding']}]}
        vector=request['context']['face'].get('mock_embedding')
        if vector is None:
            raise ValueError('Mock embedding requires scenario-linked face observation')
        vector=np.asarray(vector,dtype=float)
        noise=float(self.scenario.get('embedding_noise',0))
        if noise:
            rng=np.random.default_rng(self.seed+request['frame_id']+request['context'].get('track_id',0))
            vector=vector+rng.normal(0,noise,vector.shape)
        vector=vector/np.linalg.norm(vector)
        return {'embedding':vector.tolist(),'embedding_space':'mock-v1'}
