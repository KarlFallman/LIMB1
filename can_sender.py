import struct
import can


class CanSender:
    def __init__(self, channel="can0", bitrate=1_000_000):
        self.bus = can.Bus(
            interface="socketcan",
            channel=channel,
            bitrate=bitrate
        )

    def send_actuation(self, can_id, angle_deg, velocity_dps=20.0):
        data = struct.pack("<ff", float(angle_deg), float(velocity_dps))

        msg = can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=False
        )

        self.bus.send(msg)

    def close(self):
        self.bus.shutdown()