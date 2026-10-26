from dataclasses import dataclass


def iou(a,b):
    width=max(0,min(a[2],b[2])-max(a[0],b[0]))
    height=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=width*height
    total=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/total if total>0 else 0


@dataclass
class Track:
    id: int
    bbox: list
    seen_at: float
    identity: str | None = None


class Tracker:
    def __init__(self,ttl_ms=1000,iou_threshold=.3):
        self.ttl=ttl_ms
        self.threshold=iou_threshold
        self.tracks={}
        self.next_id=1

    def expire(self,time_ms):
        expired=[t for t,v in self.tracks.items() if time_ms-v.seen_at>self.ttl]
        for t in expired:
            del self.tracks[t]
        return expired

    def update(self,detections,time_ms):
        self.expire(time_ms)
        candidates=sorted(((iou(d['bbox'],t.bbox), index,tid)
            for index,d in enumerate(detections) for tid,t in self.tracks.items()),reverse=True)
        assigned={}
        used=set()
        for score,index,tid in candidates:
            if score>=self.threshold and index not in assigned and tid not in used:
                assigned[index]=tid
                used.add(tid)
        current={}
        for index,detection in enumerate(detections):
            tid=assigned.get(index)
            if tid is None:
                tid=self.next_id
                self.next_id+=1
                self.tracks[tid]=Track(tid,detection['bbox'],time_ms)
            track=self.tracks[tid]
            track.bbox=detection['bbox']
            track.seen_at=time_ms
            current[tid]=track
        return current
