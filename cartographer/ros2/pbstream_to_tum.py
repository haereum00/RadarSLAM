import sys, gzip, struct
sys.path.insert(0, sys.argv[3])
from cartographer.mapping.proto import serialization_pb2
MAGIC = 0x7b1d1f7b5bf501db
EPOCH_TICKS = 621355968000000000
with open(sys.argv[1], 'rb') as f:
    assert struct.unpack('<Q', f.read(8))[0] == MAGIC, 'not a pbstream'
    msgs = []
    while True:
        h = f.read(8)
        if len(h) < 8: break
        n = struct.unpack('<Q', h)[0]; msgs.append(gzip.decompress(f.read(n)))
rows = []
for raw in msgs[1:]:
    d = serialization_pb2.SerializedData(); d.ParseFromString(raw)
    if d.HasField('pose_graph'):
        for tr in d.pose_graph.trajectory:
            for nd in tr.node:
                p, q = nd.pose.translation, nd.pose.rotation
                rows.append(((nd.timestamp - EPOCH_TICKS) / 1e7, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
rows.sort()
with open(sys.argv[2], 'w') as f:
    for r in rows: f.write(' '.join(f'{v:.9f}' for v in r) + '\n')
print(sys.argv[2], len(rows), 'poses')
