from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass
class FramePacket:
    frame_id: int
    original_pts: Fraction
    time: Fraction
    pixels: Any

    @property
    def ms(self):
        return float(self.time * 1000)
