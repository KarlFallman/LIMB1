#include <vector>
#include <cmath>

class DMP1D {
private:
    // 1. Basic physics rules
    // These numbers make sure our "invisible spring" glides smoothly 
    // to a stop without bouncing back and forth.
    double alpha_z = 25.0;
    double beta_z  = 25.0 / 4.0; 
    double alpha_x = 25.0 / 3.0; // How fast the internal clock ticks

    // 2. Live tracking (what the arm is doing right now)
    double x = 1.0;   // The clock (starts at 1.0, ticks down to 0)
    double y = 0.0;   // Current angle
    double z = 0.0;   // Current speed 

    // 3. The current mission
    double tau = 1.0; // Time limit (in seconds)
    double y0  = 0.0; // Start line
    double g   = 0.0; // The goal (where the magnet is)

    // 4. The learned memory (the human shape)
    int num_basis;
    std::vector<double> c; // Where the checkpoints are
    std::vector<double> h; // How wide the checkpoints are
    std::vector<double> w; // The actual memory (how hard to push at each checkpoint)

public:
    // Setup the engine: Space out the checkpoints evenly along the clock
    DMP1D(int num_basis_functions = 20) : num_basis(num_basis_functions) {
        c.resize(num_basis);
        h.resize(num_basis);
        w.resize(num_basis, 0.0); // Start with a blank memory

        for (int i = 0; i < num_basis; ++i) {
            c[i] = std::exp(-alpha_x * (static_cast<double>(i) / (num_basis - 1)));
            h[i] = num_basis / std::pow(c[i], 2); 
        }
    }

    // Load the trained brain (usually imported from Python)
    void setWeights(const std::vector<double>& learned_weights) {
        if (learned_weights.size() == num_basis) {
            w = learned_weights;
        }
    }

    // Get ready at the start line for a new movement
    void setupMovement(double start_pos, double goal_pos, double duration_sec) {
        y0  = start_pos;
        y   = start_pos;
        g   = goal_pos;
        z   = 0.0;
        x   = 1.0;          // Wind the clock back up to 100%
        tau = duration_sec; // Set the speed limit
    }

    // The main engine loop. Run this over and over to drive the motor.
    double step(double dt) {
        
        // Step 1: Figure out how much extra "push" we need right now 
        // to match the specific human curve we memorized.
        double sum_w_psi = 0.0;
        double sum_psi   = 0.0;

        for (int i = 0; i < num_basis; ++i) {
            double psi = std::exp(-h[i] * std::pow(x - c[i], 2));
            sum_psi   += psi;
            sum_w_psi += w[i] * psi;
        }

        double f = 0.0;
        if (sum_psi > 1e-6) { // Just making sure we don't divide by zero
            f = (sum_w_psi / sum_psi) * x * (g - y0);
        }

        // Step 2: Calculate the invisible spring pulling us toward the goal,
        // and add the custom "push" (f) from Step 1.
        double dz = alpha_z * (beta_z * (g - y) - z) + f;
        double dy = z;

        // Step 3: Tick the clock down a tiny bit
        double dx = -alpha_x * x;

        // Step 4: Apply the math to update our speed and position
        z += (dz / tau) * dt;
        y += (dy / tau) * dt;
        x += (dx / tau) * dt;

        // Hand this safe, smooth angle over to the physical motor
        return y; 
    }
    
    // Quick tools to check how fast we are going, or how much time is left
    double getVelocity() const { return z / tau; }
    double getPhase() const { return x; }
};