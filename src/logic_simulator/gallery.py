import json
from pathlib import Path
import numpy as np
from .run_store import file_hash


def normalized(vector):
    a=np.asarray(vector,dtype=np.float32).reshape(-1)
    norm=np.linalg.norm(a)
    if not np.isfinite(a).all() or norm<=0:
        raise ValueError('Invalid embedding')
    return a/norm


class Gallery:
    def __init__(self,path=None):
        self.data=json.loads(Path(path).read_text()) if path else {
            'embedding_space':'mock-v1','people':[
                {'id':'alice','name':'Alice','blacklist':False,'embeddings':[[1,0,0]]},
                {'id':'bob','name':'Bob','blacklist':True,'embeddings':[[0,1,0]]}]}
        self.people={p['id']:p for p in self.data['people']}
        self.fingerprint=file_hash(path) if path else 'mock-gallery-v1'
        self.vectors=[(p['id'],normalized(e)) for p in self.people.values() for e in p['embeddings']]

    def match(self,embedding,space,threshold,margin):
        if space!=self.data['embedding_space']:
            raise ValueError('Gallery embedding-space mismatch: enroll again for selected model')
        vector=normalized(embedding)
        scores={}
        for pid,saved in self.vectors:
            if vector.shape != saved.shape:
                raise ValueError('Gallery embedding dimension mismatch')
            scores[pid]=max(scores.get(pid,-1),float(np.dot(vector,saved)))
        ranked=sorted(scores.items(),key=lambda pair:pair[1],reverse=True)
        if not ranked:
            return None,0.,'empty_gallery'
        best,score=ranked[0]
        if score<threshold:
            return None,score,'below_threshold'
        if len(ranked)>1 and score-ranked[1][1]<margin:
            return None,score,'ambiguous'
        return best,score,'matched'
