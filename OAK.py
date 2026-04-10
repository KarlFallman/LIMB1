import depthai as dai
import cv2
from blobconverter import from_openvino

palm_blob = from_openvino(
    xml="models/palm_detection_192x192.xml",
    bin="models/palm_detection_192x192.bin",
    shaves=4
)

pipeline = dai.Pipeline()

# Viktigt: .build() behövs i v3
cam = pipeline.create(dai.node.Camera).build()
cam.setSensorType(dai.CameraSensorType.COLOR)

# Kameraoutput i rätt storlek för modellen
cam_out = cam.requestOutput(
    (192, 192),
    type=dai.ImgFrame.Type.BGR888p
)

# NN
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(palm_blob)

# Länka kamerabilden till NN
cam_out.link(nn.input)

# Queues
video_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)
nn_out = nn.out.createOutputQueue(maxSize=4, blocking=False)

pipeline.start()
print("OAK-D Lite is running. Press 'q' to quit.")

while pipeline.isRunning():
    frame_in = video_queue.get()
    nn_data = nn_out.get()

    frame = frame_in.getCvFrame()

    if nn_data is not None:
        print("Typ:", type(nn_data))

        if hasattr(nn_data, "getAllLayerNames"):
            print("Layer names:", nn_data.getAllLayerNames())

        print("Tillgängliga attribut:", dir(nn_data))

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()