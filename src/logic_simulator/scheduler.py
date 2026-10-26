from fractions import Fraction


class PeriodicSchedule:
    def __init__(self, fps=None, interval_ms=None, offset_ms=0):
        self.period=Fraction(1)/Fraction(str(fps)) if fps is not None else Fraction(str(interval_ms))/1000
        if self.period <= 0:
            raise ValueError('period must be positive')
        self.next=Fraction(str(offset_ms or 0))/1000
        self.coalesced=0
        self.calls=0

    def due(self, time):
        if time < self.next:
            return None
        scheduled=self.next
        slots=(time-self.next)//self.period+1
        self.coalesced+=slots-1
        self.next+=slots*self.period
        self.calls+=1
        return scheduled
