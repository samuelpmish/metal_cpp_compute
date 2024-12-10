#include "metal/wrapper.hpp"

namespace metal {

NS::Error * _error;
MTL::Device * _device;
MTL::CommandQueue * _queue;
MTL::CommandBuffer * _command_buffer;

void _initialize() {
    _device = MTL::CreateSystemDefaultDevice();
    _queue = _device->newCommandQueue();
    _command_buffer = _queue->commandBuffer();
    if (!_command_buffer) {
        printf("%s", _error->localizedDescription()->utf8String());
        assert(false);
    }
}

MTL::Device * device() {
    if (_device == nullptr) {
        _initialize();
    }

    return _device;
};

MTL::CommandQueue * queue() {
    return _queue;
};

MTL::CommandBuffer * command_buffer() {
    return _command_buffer;
};

void device_synchronize() {
    _command_buffer->waitUntilCompleted();
};

}