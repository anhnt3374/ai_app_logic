from fractions import Fraction
from logic_simulator.scheduler import PeriodicSchedule


def test_18fps_on_30fps_source():
    schedule = PeriodicSchedule(fps=18)
    frames = [i for i in range(30) if schedule.due(Fraction(i, 30)) is not None]
    assert frames == [0,2,4,5,7,9,10,12,14,15,17,19,20,22,24,25,27,29]


def test_vfr_slots_are_coalesced():
    schedule = PeriodicSchedule(interval_ms=200)
    due = [schedule.due(Fraction(t,1000)) for t in [0,40,210,700]]
    assert due == [Fraction(0), None, Fraction(1,5), Fraction(2,5)]
    assert schedule.coalesced == 1


def test_fractional_fps_has_no_drift():
    schedule = PeriodicSchedule(fps='30000/1001')
    assert sum(schedule.due(Fraction(i*1001,30000)) is not None for i in range(3000)) == 3000
