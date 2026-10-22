import sys
from pathlib import Path
from rosbags.highlevel import AnyReader
import rosbag2_py
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

src, dst = sys.argv[1], sys.argv[2]
w = rosbag2_py.SequentialWriter()
w.open(rosbag2_py.StorageOptions(uri=dst, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
w.create_topic(rosbag2_py.TopicMetadata(name='scan', type='sensor_msgs/msg/LaserScan', serialization_format='cdr'))
w.create_topic(rosbag2_py.TopicMetadata(name='odom', type='nav_msgs/msg/Odometry', serialization_format='cdr'))
def stamp(h):
    return Time(sec=int(h.stamp.sec), nanosec=int(h.stamp.nanosec))
n = 0
with AnyReader([Path(src)]) as r:
    for con, t, raw in r.messages():
        m = r.deserialize(raw, con.msgtype)
        if con.topic == 'scan':
            o = LaserScan(); o.header.stamp = stamp(m.header); o.header.frame_id = m.header.frame_id
            o.angle_min, o.angle_max, o.angle_increment = float(m.angle_min), float(m.angle_max), float(m.angle_increment)
            o.time_increment, o.scan_time = float(m.time_increment), float(m.scan_time)
            o.range_min, o.range_max = float(m.range_min), float(m.range_max)
            o.ranges = [float(x) for x in m.ranges]
        else:
            o = Odometry(); o.header.stamp = stamp(m.header); o.header.frame_id = m.header.frame_id; o.child_frame_id = m.child_frame_id
            p, q = m.pose.pose.position, m.pose.pose.orientation
            o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = float(p.x), float(p.y), float(p.z)
            o.pose.pose.orientation.x, o.pose.pose.orientation.y = float(q.x), float(q.y)
            o.pose.pose.orientation.z, o.pose.pose.orientation.w = float(q.z), float(q.w)
        w.write(con.topic, serialize_message(o), t); n += 1
print(dst, n, 'messages')
