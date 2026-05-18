#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

class ONNX_IK_Engine {
private:
    Ort::Env env;
    Ort::Session* session;
    Ort::AllocatorWithDefaultOptions allocator;

    // IMPORTANT: These names must match your ONNX model's exact layer names!
    const char* input_node_names[1] = {"input"};  
    const char* output_node_names[1] = {"output"}; 

    // Shape: 1 batch of 63 numbers (21 MediaPipe dots * X,Y,Z)
    std::vector<int64_t> input_shape = {1, 63}; 

public:
    // 1. THE CONSTRUCTOR (Runs ONCE when you turn the robot on)
    ONNX_IK_Engine(const char* model_filepath) 
        : env(ORT_LOGGING_LEVEL_WARNING, "IK_Engine") {
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1); // Keeps it running fast on the Jetson
        
        // Load the frozen AI brain into memory
        session = new Ort::Session(env, model_filepath, session_options);
        std::cout << "ONNX Model Loaded Successfully!\n";
    }

    // Clean up memory when the robot turns off
    ~ONNX_IK_Engine() {
        delete session;
    }

    // 2. THE FAST FUNCTION (Runs 30 times a second)
    std::vector<float> calculateAngles(std::vector<float>& incoming_coords) {
        
        // A. Package the C++ coordinates into an "ONNX Tensor" (the format the AI reads)
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, incoming_coords.data(), incoming_coords.size(),
            input_shape.data(), input_shape.size()
        );

        // B. Run the AI calculation!
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr}, 
            input_node_names, &input_tensor, 1, 
            output_node_names, 1
        );

        // C. Extract the answers (the angles) from the AI's output
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        size_t output_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        
        // Return the angles as a clean C++ vector
        return std::vector<float>(floatarr, floatarr + output_count);
    }
};