#include <iostream>
#include <vector>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "dnnl_sycl.hpp"
#include "example_utils.hpp"

using namespace dnnl; 

int main(int argc, char** argv) { 
    // engine 
    dnnl::engine engine(engine::kind::gpu, 0);

    // stream
    dnnl::stream stream(engine);

    // tensor dimension
    int n = 1, c = 1, h = 1, w = 7;
    int size = n * c * h * w;

    // host allocation
    std::vector<float> src(size); 
    std::vector<float> dst(size); 
    
    // tensor initialization
    for (int i = 0; i < size; i++) { src[i] = float(i); }
    for (int i = 0; i < size; i++) { dst[i] = 0.f; }

    // create memory descriptor
    auto src_d = memory::desc({n,c,h,w}, memory::data_type::f32, memory::format_tag::nhwc);
    auto dst_d = memory::desc({n,c,h,w}, memory::data_type::f32, memory::format_tag::nhwc);
    
    // create memory object 
    auto src_mem = sycl_interop::make_memory(src_d, engine, sycl_interop::memory_kind::buffer);
    auto dst_mem = sycl_interop::make_memory(dst_d, engine, sycl_interop::memory_kind::buffer);

    // write data to object's handle (host -> device) 
    write_to_dnnl_memory(src.data(), src_mem);

    // primitive descriptor
    auto sigmoid = eltwise_forward(
        eltwise_forward::primitive_desc( 
            eltwise_forward::desc(  
                prop_kind::forward, 
                algorithm::eltwise_logistic, 
                src_d, 
                1.f, 
                0.f), 
            engine
        )
    ); 

    // primitive execution
    sigmoid.execute(
        stream, 
        { 
            {DNNL_ARG_SRC, src_mem}, 
            {DNNL_ARG_DST, dst_mem}
        }
    ); 

    // Wait for the computation to finalize.
    stream.wait();

    // Read data from memory object's handle (device -> host) 
    read_from_dnnl_memory(dst.data(), dst_mem);

    std::cout << "src tensor: ";
    for (int i=0; i < 7; i++) { std::cout << " " << src[i]; }
    std::cout << std::endl;

    std::cout << "dst tensor: ";
    for (int i=0; i < 7; i++) { std::cout << " " << dst[i]; }
    std::cout << std::endl;

    return 0; 
} 
