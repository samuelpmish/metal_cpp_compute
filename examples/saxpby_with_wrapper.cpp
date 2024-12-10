#include <cassert>
#include <iostream>

#include <Metal/Metal.hpp>

#include "metal/wrapper.hpp"

const char* saxpby_shader = R"(
    #include <metal_stdlib>

    using namespace metal;

    kernel void saxpby(constant float & a,
                       device const float * x,
                       constant float & b,
                       device float * y,
                       uint i [[thread_position_in_grid]]) {
        y[i] = a * x[i] + b * y[i];
    }
)";

int main() {

    // allocate and initialize data involved in the calculation
    uint32_t n = 100;
    float a = 2.0;
    float b = 3.0;
    metal::array<float> x(n);
    metal::array<float> y(n);
    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = 2;
    }

    using signature = void(
        float,
        const metal::array< float > &, 
        float,
        metal::array<float> &
    );

    // JIT compile and wrap the MTL objects in a std::function
    std::function saxpby = metal::create_kernel< signature >(saxpby_shader, "saxpby");

    // launch the kernel (asynchronous)
    metal::dim3 grid(n);
    metal::dim3 threadgroup(std::min(128u, n));
    saxpby(grid, threadgroup, a, x, b, y);

    // wait for kernels to finish
    metal::device_synchronize();

    // check answers
    for (int i = 0; i < n; i++) {
        float expected = a * i + b * 2;
        std::cout << i << ": " << y[i] << " " << expected << std::endl;
    }

}