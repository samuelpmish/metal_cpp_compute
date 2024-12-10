#include <cassert>
#include <iostream>

#include <Metal/Metal.hpp>

#include "common/math.hpp"
#include "common/timer.hpp"
#include "common/binary_io.hpp"
#include "common/threadpool.hpp"

#define DATA_DIR "/Users/sam/code/metal-cpp-cmake/data/"

#define USE_ATOMICS

BS::thread_pool pool;

void set_num_threads(uint32_t num_threads) {
  pool.reset(num_threads);
};

void parallel_for(uint32_t n, std::function<void(uint32_t)> func) {
  pool.detach_loop<uint32_t>(0, n, func);
  pool.wait();
}

void vertex_relaxation(
    std::vector< float3 > & vertices,
    const std::vector< uint4 > & elements,
    const std::vector< float > & fixed,
    float alpha,
    float step,
    int num_iterations,
    int num_threads) {

  if (num_iterations > 0) {

    set_num_threads(num_threads);

    uint32_t num_elements = elements.size();
    uint32_t num_vertices = vertices.size();
   #ifdef USE_ATOMICS 
    std::vector< std::atomic<float> > scale(num_vertices);
    std::vector< std::atomic<float> > grad(3 * num_vertices);
    for (int i = 0; i < num_vertices; i++) {
      scale[i] = 0.0f;
      grad[3*i+0] = 0.0f;
      grad[3*i+1] = 0.0f;
      grad[3*i+2] = 0.0f;
    }
   #else
    constexpr int nmutex = 4096;
    std::mutex mtx[nmutex];
    std::vector< float > scale(num_vertices, 0.0f);
    std::vector< float3 > grad(num_vertices, float3{});
   #endif

    timer stopwatch;

    stopwatch.start();

    for (uint32_t k = 0; k < num_iterations; k++) {

      // for each element with this geometry
      parallel_for(num_elements, [&](uint32_t i) {

        auto ids = elements[i];

        float3 v[4] = {vertices[ids[0]], vertices[ids[1]], vertices[ids[2]], vertices[ids[3]]};

        float3 L01 = v[1]-v[0];
        float3 L02 = v[2]-v[0];
        float3 L03 = v[3]-v[0];
        float3 L12 = v[2]-v[1];
        float3 L13 = v[3]-v[1];
        float3 L23 = v[3]-v[2];

        float Lrms = sqrt(dot(L01,L01)+dot(L02,L02)+dot(L03,L03)+dot(L12,L12)+dot(L13,L13)+dot(L23,L23)); 

        float top = det(L01, L02, L03);

        float4x3 dtop_dx = {{
          {
             v[1][2] * v[2][1] - v[1][1] * v[2][2] - v[1][2] * v[3][1] + v[2][2] * v[3][1] + v[1][1] * v[3][2] - v[2][1] * v[3][2], 
            -v[1][2] * v[2][0] + v[1][0] * v[2][2] + v[1][2] * v[3][0] - v[2][2] * v[3][0] - v[1][0] * v[3][2] + v[2][0] * v[3][2], 
             v[1][1] * v[2][0] - v[1][0] * v[2][1] - v[1][1] * v[3][0] + v[2][1] * v[3][0] + v[1][0] * v[3][1] - v[2][0] * v[3][1]
          }, {
            -v[0][2] * v[2][1] + v[0][1] * v[2][2] + v[0][2] * v[3][1] - v[2][2] * v[3][1] - v[0][1] * v[3][2] + v[2][1] * v[3][2], 
             v[0][2] * v[2][0] - v[0][0] * v[2][2] - v[0][2] * v[3][0] + v[2][2] * v[3][0] + v[0][0] * v[3][2] - v[2][0] * v[3][2], 
            -v[0][1] * v[2][0] + v[0][0] * v[2][1] + v[0][1] * v[3][0] - v[2][1] * v[3][0] - v[0][0] * v[3][1] + v[2][0] * v[3][1]
          }, {
             v[0][2] * v[1][1] - v[0][1] * v[1][2] - v[0][2] * v[3][1] + v[1][2] * v[3][1] + v[0][1] * v[3][2] - v[1][1] * v[3][2], 
            -v[0][2] * v[1][0] + v[0][0] * v[1][2] + v[0][2] * v[3][0] - v[1][2] * v[3][0] - v[0][0] * v[3][2] + v[1][0] * v[3][2], 
             v[0][1] * v[1][0] - v[0][0] * v[1][1] - v[0][1] * v[3][0] + v[1][1] * v[3][0] + v[0][0] * v[3][1] - v[1][0] * v[3][1]
          }, {
            -v[0][2] * v[1][1] + v[0][1] * v[1][2] + v[0][2] * v[2][1] - v[1][2] * v[2][1] - v[0][1] * v[2][2] + v[1][1] * v[2][2], 
             v[0][2] * v[1][0] - v[0][0] * v[1][2] - v[0][2] * v[2][0] + v[1][2] * v[2][0] + v[0][0] * v[2][2] - v[1][0] * v[2][2], 
            -v[0][1] * v[1][0] + v[0][0] * v[1][1] + v[0][1] * v[2][0] - v[1][1] * v[2][0] - v[0][0] * v[2][1] + v[1][0] * v[2][1]
          }
        }};

        float bot = Lrms * Lrms * Lrms;
        float4x3 dbot_dx = (1.5f * Lrms) * float4x3{{
          { 
            6.0f * v[0][0] - 2.0f * (v[1][0] + v[2][0] + v[3][0]), 
            6.0f * v[0][1] - 2.0f * (v[1][1] + v[2][1] + v[3][1]), 
            6.0f * v[0][2] - 2.0f * (v[1][2] + v[2][2] + v[3][2])
          }, {
            -2.0f * (v[0][0] - 3.0f * v[1][0] + v[2][0] + v[3][0]), 
            -2.0f * (v[0][1] - 3.0f * v[1][1] + v[2][1] + v[3][1]), 
            -2.0f * (v[0][2] - 3.0f * v[1][2] + v[2][2] + v[3][2])
          }, {
            -2.0f * (v[0][0] + v[1][0] - 3.0f * v[2][0] + v[3][0]), 
            -2.0f * (v[0][1] + v[1][1] - 3.0f * v[2][1] + v[3][1]), 
            -2.0f * (v[0][2] + v[1][2] - 3.0f * v[2][2] + v[3][2])
          }, {
            -2.0f * (v[0][0] + v[1][0] + v[2][0] - 3.0f * v[3][0]), 
            -2.0f * (v[0][1] + v[1][1] + v[2][1] - 3.0f * v[3][1]), 
            -2.0f * (v[0][2] + v[1][2] + v[2][2] - 3.0f * v[3][2])
          }
        }};

        constexpr float factor = 20.784609690826527522; // 12 * sqrt(3)

        float Q = factor * (top / bot);
        float expQ = expf(-alpha * Q);

        float4x3 dQdX = (factor * expQ) * (dtop_dx / bot - (top / (bot * bot)) * dbot_dx);

        for (int j = 0; j < 4; j++) {
          int id = ids[j];

         #ifdef USE_ATOMICS
          scale[id] += expQ;
          grad[3*id+0] += dQdX[j][0];
          grad[3*id+1] += dQdX[j][1];
          grad[3*id+2] += dQdX[j][2];
         #else
          int which = id % nmutex;
          mtx[which].lock();
          scale[id] += expQ;
          grad[id] += dQdX[j];
          mtx[which].unlock();
         #endif
        }

      });

      parallel_for(num_vertices, [&](uint32_t i) {

        float factor = step * fixed[i] / scale[i];

       #ifdef USE_ATOMICS
        float3 g = {grad[3*i+0], grad[3*i+1], grad[3*i+2]};
        vertices[i] += factor * g;
        grad[3*i+0] = 0.0f;
        grad[3*i+1] = 0.0f;
        grad[3*i+2] = 0.0f;
       #else
        vertices[i] += factor * g;
        grad[i] = float3{0.0f, 0.0f, 0.0f};
       #endif
        scale[i] = 0.0;
      });

    }

    stopwatch.stop();

    std::cout << "vertex relaxation time (CPU, ";
    if (num_threads == 1) {
        std::cout << " 1 thread)  ";
    } else {
        std::cout << num_threads << " threads) ";
    }
    std::cout << stopwatch.elapsed() * 1000 << "ms" << std::endl;

  }

}

const char* vertex_relaxation_src = R"(
    #include <metal_stdlib>

    using namespace metal;

    kernel void compute_dQdX(device atomic<float> * scale,
                             device atomic<float> * grad,
                             device const float * vertices,
                             device const uint4 * elements,
                             constant float & alpha,
                             uint e [[thread_position_in_grid]]) {

        uint4 ids = elements[e];

        float3 v[4];
        for (int j = 0; j < 4; j++) {
            v[j][0] = vertices[3*ids[j]+0];
            v[j][1] = vertices[3*ids[j]+1];
            v[j][2] = vertices[3*ids[j]+2];
        }

        float3 L01 = v[1]-v[0];
        float3 L02 = v[2]-v[0];
        float3 L03 = v[3]-v[0];
        float3 L12 = v[2]-v[1];
        float3 L13 = v[3]-v[1];
        float3 L23 = v[3]-v[2];

        float Lrms = sqrt(dot(L01,L01)+dot(L02,L02)+dot(L03,L03)+dot(L12,L12)+dot(L13,L13)+dot(L23,L23)); 

        float top = determinant(float3x3{L01, L02, L03});

        float4x3 dtop_dx = {{
          {
             v[1][2] * v[2][1] - v[1][1] * v[2][2] - v[1][2] * v[3][1] + v[2][2] * v[3][1] + v[1][1] * v[3][2] - v[2][1] * v[3][2], 
            -v[1][2] * v[2][0] + v[1][0] * v[2][2] + v[1][2] * v[3][0] - v[2][2] * v[3][0] - v[1][0] * v[3][2] + v[2][0] * v[3][2], 
             v[1][1] * v[2][0] - v[1][0] * v[2][1] - v[1][1] * v[3][0] + v[2][1] * v[3][0] + v[1][0] * v[3][1] - v[2][0] * v[3][1]
          }, {
            -v[0][2] * v[2][1] + v[0][1] * v[2][2] + v[0][2] * v[3][1] - v[2][2] * v[3][1] - v[0][1] * v[3][2] + v[2][1] * v[3][2], 
             v[0][2] * v[2][0] - v[0][0] * v[2][2] - v[0][2] * v[3][0] + v[2][2] * v[3][0] + v[0][0] * v[3][2] - v[2][0] * v[3][2], 
            -v[0][1] * v[2][0] + v[0][0] * v[2][1] + v[0][1] * v[3][0] - v[2][1] * v[3][0] - v[0][0] * v[3][1] + v[2][0] * v[3][1]
          }, {
             v[0][2] * v[1][1] - v[0][1] * v[1][2] - v[0][2] * v[3][1] + v[1][2] * v[3][1] + v[0][1] * v[3][2] - v[1][1] * v[3][2], 
            -v[0][2] * v[1][0] + v[0][0] * v[1][2] + v[0][2] * v[3][0] - v[1][2] * v[3][0] - v[0][0] * v[3][2] + v[1][0] * v[3][2], 
             v[0][1] * v[1][0] - v[0][0] * v[1][1] - v[0][1] * v[3][0] + v[1][1] * v[3][0] + v[0][0] * v[3][1] - v[1][0] * v[3][1]
          }, {
            -v[0][2] * v[1][1] + v[0][1] * v[1][2] + v[0][2] * v[2][1] - v[1][2] * v[2][1] - v[0][1] * v[2][2] + v[1][1] * v[2][2], 
             v[0][2] * v[1][0] - v[0][0] * v[1][2] - v[0][2] * v[2][0] + v[1][2] * v[2][0] + v[0][0] * v[2][2] - v[1][0] * v[2][2], 
            -v[0][1] * v[1][0] + v[0][0] * v[1][1] + v[0][1] * v[2][0] - v[1][1] * v[2][0] - v[0][0] * v[2][1] + v[1][0] * v[2][1]
          }
        }};

        float bot = Lrms * Lrms * Lrms;
        float4x3 dbot_dx = (1.5f * Lrms) * float4x3{{
          { 
            6.0f * v[0][0] - 2.0f * (v[1][0] + v[2][0] + v[3][0]), 
            6.0f * v[0][1] - 2.0f * (v[1][1] + v[2][1] + v[3][1]), 
            6.0f * v[0][2] - 2.0f * (v[1][2] + v[2][2] + v[3][2])
          }, {
            -2.0f * (v[0][0] - 3.0f * v[1][0] + v[2][0] + v[3][0]), 
            -2.0f * (v[0][1] - 3.0f * v[1][1] + v[2][1] + v[3][1]), 
            -2.0f * (v[0][2] - 3.0f * v[1][2] + v[2][2] + v[3][2])
          }, {
            -2.0f * (v[0][0] + v[1][0] - 3.0f * v[2][0] + v[3][0]), 
            -2.0f * (v[0][1] + v[1][1] - 3.0f * v[2][1] + v[3][1]), 
            -2.0f * (v[0][2] + v[1][2] - 3.0f * v[2][2] + v[3][2])
          }, {
            -2.0f * (v[0][0] + v[1][0] + v[2][0] - 3.0f * v[3][0]), 
            -2.0f * (v[0][1] + v[1][1] + v[2][1] - 3.0f * v[3][1]), 
            -2.0f * (v[0][2] + v[1][2] + v[2][2] - 3.0f * v[3][2])
          }
        }};

        constexpr float factor = 20.784609690826527522; // 12 * sqrt(3)

        float Q = factor * (top / bot);
        float expQ = exp(-alpha * Q);

        float4x3 dQdX = (factor * expQ) * (dtop_dx / bot - (top / (bot * bot)) * dbot_dx);

        for (int j = 0; j < 4; j++) {
          int id = ids[j];
          atomic_fetch_add_explicit(scale + id, expQ, memory_order_relaxed);
          atomic_fetch_add_explicit(grad + 3*id+0, dQdX[j][0], memory_order_relaxed);
          atomic_fetch_add_explicit(grad + 3*id+1, dQdX[j][1], memory_order_relaxed);
          atomic_fetch_add_explicit(grad + 3*id+2, dQdX[j][2], memory_order_relaxed);
        }
    }

    kernel void update_vertices(device float * vertices,
                                device float * scale,
                                device float * grad,
                                device const float * fixed,
                                constant float & step,
                                uint i [[thread_position_in_grid]]) {

        float factor = step * fixed[i] / scale[i];

        vertices[3*i+0] += factor * grad[3*i+0];
        vertices[3*i+1] += factor * grad[3*i+1];
        vertices[3*i+2] += factor * grad[3*i+2];

        scale[i] = 0.0;
        grad[3*i+0] = 0.0f;
        grad[3*i+1] = 0.0f;
        grad[3*i+2] = 0.0f;

    }

)";

void vertex_relaxation_metal(std::vector< float3 > & vertices,
                             const std::vector< uint4 > & elements,
                             const std::vector< float > & fixed,
                             float alpha,
                             float step,
                             int num_iterations) {

    MTL::Device * pDevice = MTL::CreateSystemDefaultDevice();

    NS::Error * pError = nullptr;

    MTL::Library* pComputeLibrary = pDevice->newLibrary(NS::String::string(vertex_relaxation_src, NS::UTF8StringEncoding), nullptr, &pError );
    if (!pComputeLibrary) {
        printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    MTL::Function * pComputedQdXFn = pComputeLibrary->newFunction(NS::String::string("compute_dQdX", NS::UTF8StringEncoding));

    MTL::ComputePipelineState * pComputedQdXPSO = pDevice->newComputePipelineState(pComputedQdXFn, &pError);
    if (!pComputedQdXPSO) {
        printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    MTL::Function * pUpdateVerticesFn = pComputeLibrary->newFunction(NS::String::string("update_vertices", NS::UTF8StringEncoding));

    MTL::ComputePipelineState * pUpdateVerticesPSO = pDevice->newComputePipelineState(pUpdateVerticesFn, &pError);
    if (!pUpdateVerticesPSO) {
        printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    uint32_t num_vertices = vertices.size();
    uint32_t num_elements = elements.size();

    MTL::Buffer * vertices_buffer = pDevice->newBuffer(3 * num_vertices * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer * elements_buffer = pDevice->newBuffer(num_elements * sizeof(uint4), MTL::ResourceStorageModeShared);
    MTL::Buffer * fixed_buffer    = pDevice->newBuffer(num_vertices * sizeof(float), MTL::ResourceStorageModeShared);

    MTL::Buffer * scale_buffer = pDevice->newBuffer(num_vertices * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer * grad_buffer = pDevice->newBuffer(3 * num_vertices * sizeof(float), MTL::ResourceStorageModeShared);

    float3 * vertices_ptr = (float3*)vertices_buffer->contents();
    float * fixed_ptr = (float*)fixed_buffer->contents();
    float * grad_ptr = (float*)grad_buffer->contents();
    float * scale_ptr = (float*)scale_buffer->contents();
    for (int i = 0; i < num_vertices; i++) {
        vertices_ptr[i] = vertices[i];
        fixed_ptr[i] = fixed[i];
        grad_ptr[3*i+0] = 0.0f;
        grad_ptr[3*i+1] = 0.0f;
        grad_ptr[3*i+2] = 0.0f;
        scale_ptr[i] = 0.0f;
    }

    uint4 * elements_ptr = (uint4*)elements_buffer->contents();
    for (int i = 0; i < num_elements; i++) {
        elements_ptr[i] = elements[i];
    }

    MTL::CommandQueue * pQueue = pDevice->newCommandQueue();

    // Encode the pipeline state object and its parameters.
    MTL::Size element_grid(num_elements, 1, 1);
    MTL::Size element_threadgroup(std::min(128u, num_elements), 1, 1);

    MTL::Size vertex_grid(num_vertices, 1, 1);
    MTL::Size vertex_threadgroup(std::min(128u, num_vertices), 1, 1);

    timer stopwatch;

    stopwatch.start();

    MTL::CommandBuffer * pCommandBuffer = pQueue->commandBuffer();
    if (!pCommandBuffer) {
        printf("%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    // Start a compute pass.
    MTL::ComputeCommandEncoder * pComputeEncoder = pCommandBuffer->computeCommandEncoder();
    if (!pComputeEncoder) {
        printf("%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    for (int k = 0; k < num_iterations; k++) {

        //  kernel void compute_dQdX(device atomic<float> * scale,
        //                           device atomic<float> * grad,
        //                           device const float * vertices,
        //                           device const uint4 * elements,
        //                           constant float & alpha,
        //                           uint e [[thread_position_in_grid]]) {
        pComputeEncoder->setComputePipelineState(pComputedQdXPSO);
        pComputeEncoder->setBuffer(scale_buffer, 0, 0);
        pComputeEncoder->setBuffer(grad_buffer, 0, 1);
        pComputeEncoder->setBuffer(vertices_buffer, 0, 2);
        pComputeEncoder->setBuffer(elements_buffer, 0, 3);
        pComputeEncoder->setBytes(&alpha, sizeof(float), 4);
        pComputeEncoder->dispatchThreads(element_grid, element_threadgroup);

        // kernel void update_vertices(device float * vertices,
        //                             device float * scale,
        //                             device float * grad,
        //                             device const float * fixed,
        //                             constant float & step,
        //                             uint i [[thread_position_in_grid]]) {
        pComputeEncoder->setComputePipelineState(pUpdateVerticesPSO);
        pComputeEncoder->setBuffer(vertices_buffer, 0, 0);
        pComputeEncoder->setBuffer(scale_buffer, 0, 1);
        pComputeEncoder->setBuffer(grad_buffer, 0, 2);
        pComputeEncoder->setBuffer(fixed_buffer, 0, 3);
        pComputeEncoder->setBytes(&step, sizeof(float), 4);
        pComputeEncoder->dispatchThreads(vertex_grid, vertex_threadgroup);

    }

    // End the compute pass.
    pComputeEncoder->endEncoding();

    // Execute the command.
    pCommandBuffer->commit();

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    pCommandBuffer->waitUntilCompleted();

    stopwatch.stop();

    std::cout << "vertex relaxation time (GPU)              " << stopwatch.elapsed() * 1000 << "ms" << std::endl;

    for (int i = 0; i < num_vertices; i++) {
        vertices[i] = vertices_ptr[i];
    }

    pDevice->release();

    pComputedQdXFn->release();
    pComputedQdXPSO->release();
    pUpdateVerticesFn->release();
    pUpdateVerticesPSO->release();
    pComputeLibrary->release();

}


int main() {

    std::vector< float3 > vertices = read_binary< float3 >(DATA_DIR"ball_coords.bin");
    std::vector< uint4 > elements = read_binary< uint4 >(DATA_DIR"ball_connectivity.bin");
    std::vector< uint32_t > bdr_vertex_ids = read_binary< uint32_t >(DATA_DIR"ball_bdr_vertices.bin");

    std::vector< float > fixed(vertices.size(), 1.0f);
    for (uint32_t i : bdr_vertex_ids) {
        fixed[i] = 0.0f;
    }

    std::vector< float3 > original_vertices = vertices;

    // run the problem single-threaded
    float step = 0.05f;
    float alpha = 6.0f;
    int num_iterations = 25;
    vertex_relaxation(vertices, elements, fixed, alpha, step, num_iterations, 1);
    std::vector< float3 > answer1 = vertices;

    // run the problem multi-threaded
    vertices = original_vertices;
    int num_threads = std::thread::hardware_concurrency();
    vertex_relaxation(vertices, elements, fixed, alpha, step, num_iterations, num_threads);
    std::vector< float3 > answer2 = vertices;

    // run the problem on an Apple silicon GPU
    vertices = original_vertices;
    vertex_relaxation_metal(vertices, elements, fixed, alpha, step, num_iterations);
    std::vector< float3 > answer3 = vertices;

    #if 0
    for (int i = 10000; i < 10100; i++) {
        std::cout << i << ": " << std::endl;
        std::cout << "    " << original_vertices[i] << std::endl;
        std::cout << "    " << answer1[i] << std::endl;
        std::cout << "    " << answer2[i] << std::endl;
        std::cout << "    " << answer3[i] << std::endl;
    }
    #endif

}