from collections import deque


class EmbeddingPolicy:
    """Bounded round-robin: update pixels outside this policy; queue holds track IDs."""
    def __init__(self, max_faces=1):
        self.max_faces=max_faces
        self.queue=deque()

    def select(self, eligible, alive=None):
        valid=set(eligible)
        alive=set(eligible if alive is None else alive)
        self.queue=deque(t for t in self.queue if t in alive)
        known=set(self.queue)
        self.queue.extend(t for t in eligible if t not in known)
        selected=[]
        for _ in range(len(self.queue)):
            track=self.queue.popleft()
            self.queue.append(track)
            if track in valid:
                selected.append(track)
                if len(selected)==self.max_faces:
                    break
        return selected


class RecognitionState:
    def __init__(self, confirmations=3, window_ms=2000):
        self.required=confirmations
        self.window=window_ms
        self.candidates={}
        self.confirmed={}
        self.seen={}

    def observe(self, track, identity, time_ms, observation):
        seen=self.seen.setdefault(track,set())
        if observation in seen:
            return False
        seen.add(observation)
        if identity is None:
            self.candidates.pop(track,None)
            return False
        previous,times=self.candidates.get(track,(None,[]))
        times=[t for t in times if time_ms-t<=self.window] if previous==identity else []
        times.append(time_ms)
        self.candidates[track]=(identity,times)
        if len(times)>=self.required:
            self.confirmed[track]=identity
            return True
        return False

    def remove(self, track):
        self.candidates.pop(track,None)
        self.confirmed.pop(track,None)
        self.seen.pop(track,None)
