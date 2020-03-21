#include <iostream>
#include "../inc/sls_multi_core.h"
int main(int argc, char const *argv[]) {

    std::cout << "Hallo" << std::endl;
    multi_core::sls(5,               // # of DNFs
                    10,                // # of Updates
                    0.5,                       // Prob of rand term in H
                    0.5,                       // Prob of rand term in H
                    0.5,                        // Prob of rand literal in H
                    nullptr,                 // Data input
                    nullptr,                      // Label input
                    nullptr,              // Positive or negative for formula
                    nullptr,               // Mask for formula
                    nullptr,     // Positive or negative for formula
                    nullptr,      // Mask for formula
                    0,                // # of data vectors
                    0,              // # of Features
                    false,                 // If score calculation should be done batchwise for given clause
                    false,          // Restart if stuck in bad local minimum
                    0.0,                // Decay factor, could be zero. Up to min_prob
                    0.0,             // Not decay below this threshold
                    false              // Wether to go bigger steps in case of no sucess
    );
    return 0;
}
