"""Baseline face recognition implementation; versioned plugins may replace it."""
from datetime import timedelta
import cv2
import numpy as np

from .gallery import Gallery
from .recognition import EmbeddingPolicy,RecognitionState
from .run_store import append_json
from .scheduler import PeriodicSchedule
from .tracking import Tracker


def crop(image,box):
    h,w=image.shape[:2]
    x1,y1,x2,y2=[int(round(v)) for v in box]
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
    if x2<=x1 or y2<=y1:
        return None,None
    return image[y1:y2,x1:x2].copy(),[x1,y1,x2,y2]


class FacePipeline:
    def __init__(self,config,gateway,run_path):
        self.config=config
        self.business=config.business
        self.gateway=gateway
        self.path=run_path
        self.gallery=Gallery(config.business.gallery)
        self.tracker=Tracker(self.business.track_ttl_ms,self.business.track_iou_threshold)
        self.policy=EmbeddingPolicy(self.business.embedding_scheduler.max_faces_per_turn)
        self.state=RecognitionState(self.business.confirm_observations,self.business.confirm_window_ms)
        self.schedules={role:PeriodicSchedule(fps=stage.fps,interval_ms=stage.interval_ms,offset_ms=stage.offset_ms)
                        for role,stage in config.stages.items() if stage.trigger=='periodic' and stage.enabled}
        self.attendance=set()
        self.blacklist={}
        self.faces={}
        self.turn=0
        self.event_count=0

    def trace(self,value):
        append_json(self.path/'trace.jsonl',value)

    def tick(self,frame):
        expired=self.tracker.expire(frame.ms)
        for tid in expired:
            self.state.remove(tid)
            self.faces.pop(tid,None)
        due={role:schedule.due(frame.time) for role,schedule in self.schedules.items()}
        current={}
        if due.get('person') is not None:
            result=self.gateway.infer('person',frame.pixels,frame,due['person'])
            current=self.tracker.update(result['payload']['detections'],frame.ms)
        if due.get('face') is None:
            return
        if due.get('person') is None:
            self.trace({'type':'skipped_dependency','stage':'face','time_ms':frame.ms})
            return
        valid={}
        self.turn+=1
        for tid,track in current.items():
            image,box=crop(frame.pixels,track.bbox)
            if image is None:
                continue
            h,w=frame.pixels.shape[:2]
            context={'parent_normalized':[box[0]/w,box[1]/h,box[2]/w,box[3]/h],'parent_box':box}
            result=self.gateway.infer('face',image,frame,due['face'],context)
            detections=result['payload']['detections']
            # Research baseline: one face per person; choose largest visible face.
            if not detections:
                self.trace({'type':'face_rejected','track_id':tid,'time_ms':frame.ms,'reason':'no_face'})
                continue
            face=max(detections,key=lambda f:(f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]))
            pixels,face_box=crop(image,face['bbox'])
            sharpness=float(cv2.Laplacian(cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()) if pixels is not None else 0.
            quality=pixels is not None and min(pixels.shape[:2])>=self.business.min_face_size and sharpness>=self.business.min_sharpness
            self.faces[tid]={'bbox':[face['bbox'][0]+box[0],face['bbox'][1]+box[1],face['bbox'][2]+box[0],face['bbox'][3]+box[1]],'seen_at':frame.ms}
            if quality and (not self.business.embedding_scheduler.stop_when_confirmed or tid not in self.state.confirmed):
                valid[tid]=(image,face,box)
            else:
                self.trace({'type':'face_rejected','track_id':tid,'time_ms':frame.ms,
                            'reason':'already_confirmed' if tid in self.state.confirmed else 'quality_gate','sharpness':sharpness})
        alive=[tid for tid in self.tracker.tracks if not self.business.embedding_scheduler.stop_when_confirmed or tid not in self.state.confirmed]
        selected=self.policy.select(list(valid),alive)
        self.trace({'type':'embedding_selection','turn':self.turn,'time_ms':frame.ms,'eligible':list(valid),
            'selected':selected,'deferred':[tid for tid in valid if tid not in selected],
            'queue':list(self.policy.queue),'max_faces':self.policy.max_faces,'reason':'round_robin' if selected else 'no_eligible_face'})
        if not self.config.stages['embedding'].enabled:
            return
        for tid in selected:
            image,face,box=valid[tid]
            result=self.gateway.infer('embedding',image,frame,context={'face':face,'track_id':tid,'parent_box':box})
            payload=result['payload']
            pid,score,reason=self.gallery.match(payload['embedding'],payload['embedding_space'],self.business.similarity_threshold,self.business.ambiguity_margin)
            confirmed=self.state.observe(tid,pid,frame.ms,result['observation_id'])
            self.trace({'type':'recognition','time_ms':frame.ms,'track_id':tid,'identity':pid,'score':score,
                'reason':('confirmed' if confirmed else 'awaiting_evidence') if pid else reason,
                'observation_id':result['observation_id']})
            if confirmed:
                self.tracker.tracks[tid].identity=pid
                self.emit_events(frame,tid,pid,score,result['observation_id'])

    def emit_events(self,frame,tid,pid,score,observation):
        person=self.gallery.people[pid]
        business_time=self.config.input.business_start_at+timedelta(seconds=float(frame.original_pts))
        evidence=f'evidence/frame-{frame.frame_id:08d}.jpg'
        cv2.imwrite(str(self.path/evidence),frame.pixels)
        event={'time_ms':frame.ms,'original_ms':float(frame.original_pts*1000),'business_time':business_time.isoformat(),
            'identity':pid,'name':person.get('name',pid),'track_id':tid,'score':score,'evidence':evidence,
            'observation_id':observation,'rule':{'confirm_observations':self.business.confirm_observations,
            'confirm_window_ms':self.business.confirm_window_ms,'threshold':self.business.similarity_threshold}}
        key=(pid,business_time.date().isoformat())
        if key not in self.attendance:
            self.attendance.add(key)
            self.emit({**event,'type':'attendance'})
        if person.get('blacklist') and frame.ms-self.blacklist.get(pid,float('-inf'))>=self.business.blacklist_cooldown_ms:
            self.blacklist[pid]=frame.ms
            self.emit({**event,'type':'blacklist'})

    def emit(self,event):
        self.event_count+=1
        append_json(self.path/'events.jsonl',event)

    def render(self,frame):
        image=frame.pixels.copy()
        cv2.putText(image,f'{frame.ms/1000:.3f}s | face-recognition',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        for tid,track in self.tracker.tracks.items():
            box=[int(v) for v in track.bbox]
            color=(0,0,255) if track.identity and self.gallery.people[track.identity].get('blacklist') else (0,230,0)
            cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),color,2)
            label=f'#{tid} {track.identity or "candidate"} age={int(frame.ms-track.seen_at)}ms'
            cv2.putText(image,label,(box[0],max(35,box[1])),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
        return image

    def finalize(self):
        return {'events':self.event_count,'gallery_hash':self.gallery.fingerprint,
            'detection':{role:{'calls':s.calls,'coalesced_slots':s.coalesced} for role,s in self.schedules.items()}}
