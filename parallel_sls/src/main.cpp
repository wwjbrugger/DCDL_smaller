#include <iostream>
#include "../inc/sls_multi_core.h"
int main(int argc, char const *argv[]) {

    std::cout << "Hallo" << std::endl;
    features_t foo [5] = { 16, 2, 77, 40, 2 };
    bool label [5] = {true, true, false, false, true};


    multi_core::sls(1,               // # of DNFs
                    10,                // # of Updates
                    0.5,                       // Prob of rand term in H
                    0.5,                       // Prob of rand term in H
                    0.5,                        // Prob of rand literal in H
                    //nullptr,                 // Data input
                    foo,
                    //nullptr,                      // Label input
                    label,
                    //nullptr,              // Positive or negative for formula
                    foo,
                    //nullptr,               // Mask for formula
                    foo,
                    //nullptr,     // Positive or negative for formula
                    foo,
                    //nullptr,      // Mask for formula
                    foo,
                    5,                // # of data vectors
                    8,              // # of Features
                    false,                 // If score calculation should be done batchwise for given clause
                    false,          // Restart if stuck in bad local minimum
                    0.0,                // Decay factor, could be zero. Up to min_prob
                    0.0,             // Not decay below this threshold
                    false              // Wether to go bigger steps in case of no sucess
    );
    return 0;
}
