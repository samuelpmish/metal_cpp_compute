#pragma once

#include <memory>
#include <iostream> // TODO
#include <functional>

#include <Metal/Metal.hpp>

namespace metal {

  MTL::Device * device();
  MTL::CommandQueue * queue();
  MTL::CommandBuffer * command_buffer();

  void device_synchronize();

  struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    dim3(uint32_t nx, uint32_t ny = 1, uint32_t nz = 1) : x(nx), y(ny), z(ny) {}
  };

  template < typename T >
  struct array {
    array(uint32_t size) {
        sz = size;
        buffer = device()->newBuffer(size * sizeof(T), MTL::ResourceStorageModeShared);
        ptr = (T*)buffer->contents();
    }

    ~array() {
        buffer->release();
        sz = 0;
        ptr = nullptr;
    }

    T & operator[](uint32_t i) { return ptr[i]; }
    const T & operator[](uint32_t i) const { return ptr[i]; }

    T * ptr;
    MTL::Buffer * buffer;
    uint32_t sz;
  };

  template < typename T >
  struct KernelArgumentType {
    static constexpr bool is_valid = false;
  };

  #define REGISTER_VALID_TYPE(T) \
  template <> struct KernelArgumentType< T > { static constexpr bool is_valid = true; };

  REGISTER_VALID_TYPE(int32_t);
  REGISTER_VALID_TYPE(uint32_t);
  REGISTER_VALID_TYPE(float);

  REGISTER_VALID_TYPE(array< int > &);
  REGISTER_VALID_TYPE(array< uint32_t > &);
  REGISTER_VALID_TYPE(array< float > &);

  REGISTER_VALID_TYPE(const array< int32_t > &);
  REGISTER_VALID_TYPE(const array< uint32_t > &);
  REGISTER_VALID_TYPE(const array< float > &);

  #undef VALID_TYPE

  template < typename T >
  struct FunctionSignature{
    static constexpr bool is_valid_kernel_signature = false;
  };

  template < typename return_type, typename ... arg_types >
  struct FunctionSignature< return_type(arg_types ...) >{
    static constexpr bool is_valid_kernel_signature = 
        std::is_same< return_type, void >::value && (KernelArgumentType<arg_types>::is_valid && ...);
  };

  namespace impl {

    template < typename T > struct is_scalar_type : std::false_type {};
    template <> struct is_scalar_type< int32_t > : std::true_type {};
    template <> struct is_scalar_type< uint32_t > : std::true_type {};
    template <> struct is_scalar_type< float > : std::true_type {};

    template < typename T > struct is_array_type : std::false_type {};
    template < typename T > struct is_array_type< metal::array<T> & > : std::true_type {};
    template < typename T > struct is_array_type< const metal::array<T> & > : std::true_type {};

    template < typename arg_type >
    void encode_argument(MTL::ComputeCommandEncoder * encoder, arg_type arg, uint32_t arg_index) {
        if constexpr (is_scalar_type<arg_type>::value) {
            encoder->setBytes(&arg, sizeof(arg_type), arg_index);
            //std::cout << "encoding uniform at index " << arg_index << std::endl;
            //std::cout << "arg value: " << arg << std::endl;
        }

        if constexpr (is_array_type<arg_type>::value) {
            encoder->setBuffer(arg.buffer, 0, arg_index);
            //std::cout << "encoding buffer at index " << arg_index << std::endl;
            //std::cout << "buffer values: " << arg[0] << " " << arg[1] << " " << arg[2] << std::endl;
        }
    }

    template < typename ... arg_types >
    auto create_kernel(FunctionSignature<void(arg_types ...)>, std::string shader_src, std::string kernel_name) {

      NS::Error * error = nullptr;

      MTL::Library* compute_library = device()->newLibrary(NS::String::string(shader_src.c_str(), NS::UTF8StringEncoding), nullptr, &error);
      if (!compute_library) {
        printf("%s", error->localizedDescription()->utf8String());
        assert(false);
      }

      MTL::Function * kernel_fn = compute_library->newFunction(NS::String::string(kernel_name.c_str(), NS::UTF8StringEncoding) );

      MTL::ComputePipelineState * compute_pso = device()->newComputePipelineState(kernel_fn, &error);
      if (!compute_pso) {
        printf("%s", error->localizedDescription()->utf8String());
        assert(false);
      }

      auto deleter = [](auto * ptr){ ptr->release(); };

      return std::function< void(dim3, dim3, arg_types ...) >([
        _compute_library = std::shared_ptr< MTL::Library >(compute_library, deleter),
        _kernel_fn = std::shared_ptr< MTL::Function >(kernel_fn, deleter),
        _compute_pso = std::shared_ptr< MTL::ComputePipelineState >(compute_pso, deleter)
      ](dim3 grid, dim3 threadgroup, arg_types ... args) {

        // Start a compute pass.
        MTL::ComputeCommandEncoder * encoder = command_buffer()->computeCommandEncoder();
        encoder->setComputePipelineState(_compute_pso.get());

        // Encode the compute shader arguments
        uint32_t index = 0;
        (encode_argument<arg_types>(encoder, args, index++), ...);

        // Encode the compute command.
        encoder->dispatchThreads(
            MTL::Size(grid.x, grid.y, grid.z), 
            MTL::Size(threadgroup.x, threadgroup.y, threadgroup.z)
        );

        // End the compute pass.
        encoder->endEncoding();

        // Execute the command.
        command_buffer()->commit();

      });
    };

  }

  template < typename T >
  auto create_kernel(std::string shader_src, std::string kernel_name) {
    using signature = FunctionSignature<T>;
    static_assert(signature::is_valid_kernel_signature, "invalid kernel signature");
    return impl::create_kernel(signature{}, shader_src, kernel_name);
  };

}