#include <iostream>
#include <vector>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <algorithm> // For std::clamp

#include "ONNX_IK.h" // Your AI Translator
#include "DMP1D.h"   // Your Math Smoother

// --- CONFIGURATION ---
const int PORT_RECEIVE = 5005;  // Catching data FROM Python
const int PORT_SEND = 5006;     // Sending data TO Python
const int NUM_COORDINATES = 63; // 21 MediaPipe dots * 3 (X,Y,Z)
const double DT = 0.033;        // Simulation step time (~30 FPS)

// --- HARDWARE JOINT LIMITS (From joint_limits.py) ---
// These are in RADIANS.
const double LIMITS_RAD[4][2] = {
    {0.0, 1.05},   // 0: Elbow flexion (0 to ~60 deg)
    {0.0, 1.39},   // 1: Shoulder flexion (0 to ~80 deg)
    {0.0, 0.69},   // 2: Shoulder abduction (0 to ~40 deg)
    {-0.69, 0.69}  // 3: Shoulder lateral/medial rotation (-40 to 40 deg)
};

int main() {
    // ==========================================
    // 1. TURN ON THE AI BRAIN
    // ==========================================
    std::cout << "Loading AI Brain...\n";
    // Using the exact ONNX filename from your repository
    ONNX_IK_Engine AI_Brain("movement_gru.onnx"); 
    
    // ==========================================
    // 2. TURN ON THE DMP MUSCLES (4 DOFs)
    // ==========================================
    std::cout << "Starting DMP Engines for the Arm...\n";
    DMP1D dmp_elbow(30);   
    DMP1D dmp_sh_flex(30);
    DMP1D dmp_sh_abd(30);
    DMP1D dmp_sh_rot(30);

    // Track current physical angles (start at 0.0)
    double current_angles[4] = {0.0, 0.0, 0.0, 0.0};

    // ==========================================
    // 3. BUILD THE UDP "EARS" (To receive coordinates)
    // ==========================================
    int udp_socket;
    struct sockaddr_in server_address, client_address;
    socklen_t client_len = sizeof(client_address);

    if ((udp_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        std::cerr << "Error: Could not create UDP socket!\n";
        return -1;
    }

    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(PORT_RECEIVE);

    if (bind(udp_socket, (const struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
        std::cerr << "Error: Port " << PORT_RECEIVE << " is already in use!\n";
        return -1;
    }

    // ==========================================
    // 4. PREPARE THE UDP "MOUTH" (To send to PyBullet)
    // ==========================================
    struct sockaddr_in sim_address;
    memset(&sim_address, 0, sizeof(sim_address));
    sim_address.sin_family = AF_INET;
    sim_address.sin_port = htons(PORT_SEND);
    inet_pton(AF_INET, "127.0.0.1", &sim_address.sin_addr);
    
    std::cout << "UDP Socket Open! Listening on Port " << PORT_RECEIVE << "...\n";
    std::cout << "Ready for live mimicking (independent of recognition).\n";
    std::cout << "-------------------------------------------\n";

    float incoming_buffer[NUM_COORDINATES]; 

    // ==========================================
    // 5. THE LIVE MIMICKING LOOP
    // ==========================================
    while (true) {
        // A. CATCH FILTERED DATA FROM PYTHON
        int bytes_received = recvfrom(udp_socket, (char *)incoming_buffer, sizeof(incoming_buffer), 
                                      MSG_WAITALL, (struct sockaddr *)&client_address, &client_len);

        if (bytes_received > 0) {
            std::vector<float> camera_coords(incoming_buffer, incoming_buffer + NUM_COORDINATES);

            // B. RUN THE AI INVERSE KINEMATICS
            std::vector<float> target_angles = AI_Brain.calculateAngles(camera_coords);

            // Safety check: ensure ONNX returned exactly 4 angles
            if (target_angles.size() < 4) continue;

            // C. SET DMP TARGETS
            dmp_elbow.setupMovement(current_angles[0], target_angles[0], 0.5);
            dmp_sh_flex.setupMovement(current_angles[1], target_angles[1], 0.5);
            dmp_sh_abd.setupMovement(current_angles[2], target_angles[2], 0.5);
            dmp_sh_rot.setupMovement(current_angles[3], target_angles[3], 0.5);

            // D. CALCULATE SMOOTH STEP & APPLY LIMITS
            current_angles[0] = std::clamp(dmp_elbow.step(DT), LIMITS_RAD[0][0], LIMITS_RAD[0][1]);
            current_angles[1] = std::clamp(dmp_sh_flex.step(DT), LIMITS_RAD[1][0], LIMITS_RAD[1][1]);
            current_angles[2] = std::clamp(dmp_sh_abd.step(DT), LIMITS_RAD[2][0], LIMITS_RAD[2][1]);
            current_angles[3] = std::clamp(dmp_sh_rot.step(DT), LIMITS_RAD[3][0], LIMITS_RAD[3][1]);

            // E. SEND BACK TO PYTHON SIMULATION
            // Package the 4 safe, smoothed angles
            float sim_output[4] = {
                (float)current_angles[0], 
                (float)current_angles[1], 
                (float)current_angles[2], 
                (float)current_angles[3]
            };
            
            // Fire it to Port 5006
            sendto(udp_socket, sim_output, sizeof(sim_output), 0, 
                   (struct sockaddr*)&sim_address, sizeof(sim_address));

            std::cout << "Sent to Sim -> Elbow: " << current_angles[0] 
                      << " | Sh_Flex: " << current_angles[1]
                      << " | Sh_Abd: " << current_angles[2]
                      << " | Sh_Rot: " << current_angles[3] << "\r" << std::flush;
        }
    }

    close(udp_socket);
    return 0;
}