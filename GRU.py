import os
import json
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# =========================
# CONFIG
# =========================
ENGINE_PATH = "movement_gru.engine"

INPUT_SHAPE = (1, 30, 69)
OUTPUT_SHAPE = (1, 64)

DATA_FOLDER = "Data/Training" #CHANGE TO REFERENCES LATER
TEST_FOLDER = "Test/Training" #CHANGE TO FINAL LATER
TEST_FILE = "ID1test.json"


# =========================
# PREPROCESS
# =========================
def frame_to_vector(frame):
    vec = []

    for joint in ["shoulder", "elbow"]:
        vec.extend(frame.get(joint, [0.0, 0.0, 0.0]))

    hand = frame.get("hand", [])
    hand_map = {p["id"]: p for p in hand}

    for i in range(21):
        if i in hand_map:
            p = hand_map[i]
            vec.extend([p["x"], p["y"], p["depth_m"]])
        else:
            vec.extend([0.0, 0.0, 0.0])

    return np.array(vec, dtype=np.float32)


def load_sequence(path):
    with open(path, "r") as f:
        raw = json.load(f)

    seq = [frame_to_vector(frame) for frame in raw["data"]]
    return np.stack(seq)


def prepare_sequence(seq, target_len=30):
    seq = seq[:target_len]

    if len(seq) < target_len:
        pad = np.zeros(
            (target_len - len(seq), seq.shape[1]),
            dtype=np.float32
        )
        seq = np.vstack([seq, pad])

    return seq


# =========================
# TENSORRT
# =========================
class TRTInference:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(
                f.read()
            )

        self.context = self.engine.create_execution_context()

        self.input_size = int(
            np.prod(INPUT_SHAPE) * np.float32().nbytes
        )
        self.output_size = int(
            np.prod(OUTPUT_SHAPE) * np.float32().nbytes
        )

        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)

        self.stream = cuda.Stream()

    def infer(self, input_array):
        output = np.empty(OUTPUT_SHAPE, dtype=np.float32)

        cuda.memcpy_htod_async(
            self.d_input,
            input_array.astype(np.float32),
            self.stream
        )

        bindings = [int(self.d_input), int(self.d_output)]

        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=self.stream.handle
        )

        cuda.memcpy_dtoh_async(
            output,
            self.d_output,
            self.stream
        )

        self.stream.synchronize()

        return output[0]


# =========================
# EMBEDDING
# =========================
def get_embedding(trt_model, file_path):
    seq = load_sequence(file_path)
    seq = prepare_sequence(seq)

    input_data = np.expand_dims(seq, axis=0).astype(np.float32)

    return trt_model.infer(input_data)


def distance(a, b):
    return np.linalg.norm(a - b)


# =========================
# BUILD REFERENCES
# =========================
def build_references(trt_model):
    references = {}

    for file in os.listdir(DATA_FOLDER):
        if not file.endswith(".json"):
            continue

        path = os.path.join(DATA_FOLDER, file)

        with open(path, "r") as f:
            raw = json.load(f)

        user_id = raw["user_id"]

        if user_id not in references:
            references[user_id] = []

        references[user_id].append(path)

    person_embeddings = {}

    print("Building references...\n")

    for user_id, files in references.items():
        embeddings = []

        for file_path in files:
            emb = get_embedding(trt_model, file_path)
            embeddings.append(emb)

        mean_embedding = np.mean(embeddings, axis=0)
        person_embeddings[user_id] = mean_embedding

        print(
            f"User {user_id}: "
            f"{len(files)} reference files"
        )

    return person_embeddings


# =========================
# PREDICT
# =========================
def predict(trt_model):
    references = build_references(trt_model)

    test_path = os.path.join(TEST_FOLDER, TEST_FILE)

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Missing test file: {test_path}"
        )

    print(f"\nTesting: {test_path}\n")

    test_embedding = get_embedding(
        trt_model,
        test_path
    )

    best_user = None
    best_distance = float("inf")

    print("Distances:")

    for user_id, ref_embedding in references.items():
        d = distance(test_embedding, ref_embedding)

        print(f"User {user_id}: {d:.4f}")

        if d < best_distance:
            best_distance = d
            best_user = user_id

    print(f"\nPrediction: User {best_user}")


# =========================
# MAIN
# =========================
def main():
    trt_model = TRTInference(ENGINE_PATH)
    predict(trt_model)


if __name__ == "__main__":
    main()