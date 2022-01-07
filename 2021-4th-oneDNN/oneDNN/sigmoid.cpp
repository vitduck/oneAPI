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
    dnnl::stream stream(engine);

    // tensor dimension
    const dnnl::memory::dim N = 1, C = 1, H = 1, W = 7; 

    // tensor allocation (host) 
    std::vector<float> src(N*C*H*W);  
    std::vector<float> dst(N*C*H*W); 
    
    // tensor initialization
    for (int i = 0; i < src.size(); i++) { src[i] = float(i); }

    // create memory descriptor (NCHW format)
    auto src_d = memory::desc({N,C,H,W}, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_d = memory::desc({N,C,H,W}, memory::data_type::f32, memory::format_tag::nchw);
    
    // create memory object 
    auto src_mem = sycl_interop::make_memory(src_d, engine, sycl_interop::memory_kind::buffer);
    auto dst_mem = sycl_interop::make_memory(dst_d, engine, sycl_interop::memory_kind::buffer);

    // write data to device's memory
    write_to_dnnl_memory(src.data(), src_mem);

    // operation descriptor
    auto eltwise_d = eltwise_forward::desc(prop_kind::forward, algorithm::eltwise_logistic, src_d, 1.f, 0.f); 

    // sigmoid descriptor 
    auto eltwise_p = eltwise_forward::primitive_desc(eltwise_d, engine); 
    
    // relu stream
    auto sigmoid = eltwise_forward(eltwise_p); 

    // relu execution
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
