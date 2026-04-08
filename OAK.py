import depthai as dai
import cv2

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.Camera)
cam.setSensorType(dai.CameraSensorType.COLOR)
cam.build()
#testforcommit
output = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888i)
video_queue = output.createOutputQueue(maxSize=4, blocking=False)

pipeline.start()
print("OAK-D-Lite is running. Press 'q' to quit.")

while True:
    frame = video_queue.get().getCvFrame()
    if frame is None:
        continue

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
