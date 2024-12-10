#include <cassert>
#include <iostream>

#include <Metal/Metal.hpp>

int main() {

    MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();

    const char* saxpby_src = R"(
        #include <metal_stdlib>

        using namespace metal;

        kernel void saxpby(constant float & a,
                           device const float * x,
                           constant float & b,
                           device float * y,
                           uint i [[thread_position_in_grid]])
        {
            y[i] = a * x[i] + b * y[i];
        }
    )";

    NS::Error * pError = nullptr;

    MTL::Library* pComputeLibrary = pDevice->newLibrary(NS::String::string(saxpby_src, NS::UTF8StringEncoding), nullptr, &pError );
    if ( !pComputeLibrary )
    {
        printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    MTL::Function * pSaxpbyFn = pComputeLibrary->newFunction(NS::String::string("saxpby", NS::UTF8StringEncoding) );

    MTL::ComputePipelineState * pComputePSO = pDevice->newComputePipelineState( pSaxpbyFn, &pError );
    if ( !pComputePSO )
    {
        printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    uint32_t n = 100;
    uint32_t bufferSize = n * sizeof(float);

    MTL::Buffer * x = pDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer * y = pDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);

    float a = 2.0;
    float b = 3.0;
    float * x_ptr = (float*)x->contents();
    float * y_ptr = (float*)y->contents();
    for (int i = 0; i < n; i++) {
        x_ptr[i] = i;
        y_ptr[i] = 2;
    }

    MTL::CommandQueue * pQueue = pDevice->newCommandQueue();

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

    // Encode the pipeline state object and its parameters.
    pComputeEncoder->setComputePipelineState(pComputePSO);
    pComputeEncoder->setBytes(&a, sizeof(float), 0);
    pComputeEncoder->setBuffer(x, 0, 1);
    pComputeEncoder->setBytes(&b, sizeof(float), 2);
    pComputeEncoder->setBuffer(y, 0, 3);

    MTL::Size gridSize(n, 1, 1);
    MTL::Size threadgroupSize(std::min(128u, n), 1, 1);

    // Encode the compute command.
    pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);

    // End the compute pass.
    pComputeEncoder->endEncoding();

    // Execute the command.
    pCommandBuffer->commit();

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    pCommandBuffer->waitUntilCompleted();

    // check answers
    for (int i = 0; i < n; i++) {
        float expected = a * i + b * 2;
        std::cout << i << ": " << y_ptr[i] << " " << expected << std::endl;
    }

    pDevice->release();

    pSaxpbyFn->release();
    pComputeLibrary->release();

}