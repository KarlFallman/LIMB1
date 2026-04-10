import depthai as dai
for name in ['XLinkOut','XLinkOutBridge','XLinkOutHost','XLinkIn','XLinkInBridge','XLinkInHost']:
    cls = getattr(dai.node.internal, name)
    print('---', name, cls.__module__, cls.__bases__)
    p = dai.Pipeline()
    try:
        p.create(cls)
        print('create ok')
    except Exception as e:
        print('create failed', type(e).__name__, e)
