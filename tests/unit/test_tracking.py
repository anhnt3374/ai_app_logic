from logic_simulator.tracking import Tracker


def test_unique_assignment_and_ttl():
    tracker=Tracker(ttl_ms=1000, iou_threshold=.3)
    result=tracker.update([{'bbox':[0,0,50,50]},{'bbox':[100,0,150,50]}],0)
    assert list(result)==[1,2]
    result=tracker.update([{'bbox':[2,0,52,50]}],100)
    assert list(result)==[1]
    tracker.expire(1200)
    assert tracker.tracks=={}
    assert list(tracker.update([{'bbox':[0,0,50,50]}],1200))==[3]
