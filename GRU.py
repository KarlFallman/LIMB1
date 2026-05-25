import os
import sys

# ==============================================================================
# Windows DLL-fix för CUDA och TensorRT (Måste ligga allra högst upp i filen!)
# ==============================================================================
# 1. Byt ut v12.4 mot din exakta CUDA-version om du har en annan (t.ex. v12.6)
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"

if os.path.exists(cuda_bin):
    if sys.version_info >= (3, 8):
        os.add_dll_directory(cuda_bin)
    os.environ["PATH"] = cuda_bin + os.path.pathsep + os.environ["PATH"]
else:
    print(f"Varning: Hittade inte CUDA-mappen på sökvägen: {cuda_bin}")
import json
import numpy as np
import tensorrt as trt

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

# =========================
# CONFIG
# =========================
ONNX_PATH = "movement_gru.onnx"    # Din ONNX-fil från träningskoden
ENGINE_PATH = "movement_gru.engine"

INPUT_SHAPE = (1, 60, 69)          
OUTPUT_SHAPE = (1, 128)          

DATA_FOLDER = "Data/References"
TEST_FOLDER = "Test/Final"
TEST_FILE = "ID11test.json"       
UNKNOWN_THRESHOLD = 0.51           


# =========================
# AUTOMATISK ENGINE-BUILDER (Ersätter trtexec!)
# =========================
def build_engine_from_onnx():
    if os.path.exists(ENGINE_PATH):
        return # Motorn finns redan, hoppa över!

    print(f"'{ENGINE_PATH}' saknas. Bygger TensorRT-motor från {ONNX_PATH}...")
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"Hittade inte din {ONNX_PATH}! Kör träningskoden först.")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Konfigurera nätverket
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    # Läs in ONNX
    with open(ONNX_PATH, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Misslyckades med att tolka ONNX-filen.")

    # Konfigurera byggaren
    config = builder.create_builder_config()
    
    # ==============================================================================
    # FIX: Sätt upp Optimization Profile för dynamisk ONNX-modell
    # ==============================================================================
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name # Hämta namnet på input-lagret (t.ex. "input" eller "onnx::gemm_0")
    
    # Vi sätter Min, Optimal och Max form till (1, 60, 69) eftersom du kör en sekvens i taget
    profile.set_shape(input_name, (1, 60, 69), (1, 60, 69), (1, 60, 69))
    config.add_optimization_profile(profile)
    # ==============================================================================

    # Bygg och spara motorn
    print("Kompilerar nätverket (detta kan ta upp till en minut)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Misslyckades med att bygga TensorRT-motorn. Kontrollera loggarna ovan.")

    with open(ENGINE_PATH, 'wb') as f:
        f.write(serialized_engine)
        
    print(f" Klart! Motorn sparades framgångsrikt som '{ENGINE_PATH}'.\n")

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

    data = raw["data"]
    seq = [frame_to_vector(frame) for frame in data]
    seq = np.stack(seq)

    num_frames = seq.shape[0]
    num_features = seq.shape[1]
    target_len = INPUT_SHAPE[1]
    
    if num_frames > 1:
        current_indices = np.linspace(0, num_frames - 1, num_frames)
        target_indices = np.linspace(0, num_frames - 1, target_len)
        resampled_chunk = np.zeros((target_len, num_features))
        for f in range(num_features):
            resampled_chunk[:, f] = np.interp(target_indices, current_indices, seq[:, f])
    else:
        resampled_chunk = np.repeat(seq, target_len, axis=0)

    centered_chunk = resampled_chunk.copy()
    for t in range(len(centered_chunk)):
        if np.all(centered_chunk[t] == 0): 
            continue
        base_x, base_y, base_z = centered_chunk[t, 0], centered_chunk[t, 1], centered_chunk[t, 2]
        for p in range(0, 69, 3):
            centered_chunk[t, p] -= base_x
            centered_chunk[t, p+1] -= base_y
            centered_chunk[t, p+2] -= base_z

    return centered_chunk


# =========================
# TENSORRT INFERENCE RUNTIME
# =========================
class TRTInference:
    def __init__(self, engine_path):
        import torch
        # 1. Starta TensorRT-logg och läs in din färdiga motor
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Spara tensor-objekten direkt på GPU (istället för bara adresser)
        self.input_tensor = torch.empty(INPUT_SHAPE, dtype=torch.float32, device="cuda")
        self.output_tensor = torch.empty(OUTPUT_SHAPE, dtype=torch.float32, device="cuda")
        
        # Spara adresserna så TensorRT hittar dem
        self.d_input = self.input_tensor.data_ptr()
        self.d_output = self.output_tensor.data_ptr()
        
        # Hämta den riktiga stream-adressen från PyTorch
        self.stream = torch.cuda.current_stream().cuda_stream

    def infer(self, input_data):
        import torch

        # 1. Konvertera input_data till en PyTorch-tensor och skicka direkt till GPU-tensorn
        with torch.no_grad():
            input_torch = torch.from_numpy(input_data).float().to("cuda")
            self.input_tensor.copy_(input_torch)

        # 2. Koppla adresserna till TensorRT
        input_name = self.engine.get_tensor_name(0)
        output_name = self.engine.get_tensor_name(1)
        self.context.set_tensor_address(input_name, self.d_input)
        self.context.set_tensor_address(output_name, self.d_output)

        # 3. Kör inferensen på grafikkortet via PyTorchs stream
        self.context.execute_async_v3(self.stream)
        
        # Tvinga GPU:n att bli klar
        torch.cuda.synchronize()

        # 4. Hämta tillbaka resultatet till CPU och gör om till en NumPy array
        return self.output_tensor.cpu().numpy()
    
    def __del__(self):
        if hasattr(self, 'd_input'): cudart.cudaFree(self.d_input)
        if hasattr(self, 'd_output'): cudart.cudaFree(self.d_output)
        if hasattr(self, 'stream'): cudart.cudaStreamDestroy(self.stream)


# =========================
# EMBEDDING & DISTANCE
# =========================
def get_embedding(trt_model, file_path):
    chunk = load_sequence(file_path)
    input_data = np.expand_dims(chunk, axis=0).astype(np.float32)
    return trt_model.infer(input_data)

def distance(a, b):
    return np.linalg.norm(a - b)


# =========================
# BUILD REFERENCES
# =========================
def build_references(trt_model):
    file_embeddings = []
    print("Building references via TensorRT Engine...\n")

    for file in os.listdir(DATA_FOLDER):
        if not file.endswith(".json"):
            continue

        path = os.path.join(DATA_FOLDER, file)
        with open(path, "r") as f:
            raw = json.load(f)

        user_id = raw["user_id"]
        emb = get_embedding(trt_model, path)
        
        file_embeddings.append({
            "user_id": user_id,
            "file_name": file,
            "embedding": emb
        })

    print(f"Loaded {len(file_embeddings)} individual reference files.\n")
    return file_embeddings


# =========================
# PREDICT
# =========================
def predict(trt_model):
    all_refs = build_references(trt_model)
    test_path = os.path.join(TEST_FOLDER, TEST_FILE)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find test file: {test_path}")

    print(f"Testing file: {test_path}\n")
    test_embedding = get_embedding(trt_model, test_path)

    scored_files = []
    for ref in all_refs:
        d = distance(test_embedding, ref["embedding"])
        scored_files.append({
            "user_id": ref["user_id"],
            "file_name": ref["file_name"],
            "distance": d
        })

    scored_files = sorted(scored_files, key=lambda x: x["distance"])

    print("Top 5 closest reference files:")
    for i, res in enumerate(scored_files[:5]):
        print(f" {i+1}. User {res['user_id']} (File: {res['file_name']}) - Distance: {res['distance']:.4f}")

    best_match = scored_files[0]
    print("-" * 40)
    if best_match["distance"] > UNKNOWN_THRESHOLD:
        print(f"Prediction: Unknown User (Closest was User {best_match['user_id']} at {best_match['distance']:.4f})")
    else:
        print(f"Prediction: User {best_match['user_id']} (Based on file {best_match['file_name']})")


# =========================
# MAIN
# =========================
def main():
    # 1. Bygg motorn automatiskt via Python (om den inte redan finns)
    build_engine_from_onnx()

    # 2. Starta inferensen
    trt_model = TRTInference(ENGINE_PATH)
    predict(trt_model)

if __name__ == "__main__":
    main()