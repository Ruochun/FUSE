#include <fuse/smoke.h>
#include <iostream>
#include <stdexcept>
#include <vector>
int main() {
    try {
        if (fuse::smoke_test() != std::vector<float>{3.0f, -2.0f, 0.5f, 7.0f}) return 1;
        std::cout << "NanoVDB CUDA readback passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
