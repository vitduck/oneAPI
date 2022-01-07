#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h" 
#include "dnnl_sycl.hpp"
#include "example_utils.hpp"

using namespace::dnnl; 

int main(int argc, char** argv) { 
    // engine 
    dnnl::engine engine(engine::kind::gpu, 0);
    dnnl::stream stream(engine);

    // convert src image to matrix format 
    cv::Mat src = cv::imread("./Lenna.png", cv::IMREAD_COLOR);

    // convert src matrix to FP32 with 3 channels
    src.convertTo(src, CV_32FC3); 

    // normalize color range to (0,1)
    cv::normalize(src, src, 0, 1, cv::NORM_MINMAX); 

    // debug: write mat to xml file
    // cv::FileStorage file("src.xml", cv::FileStorage::WRITE);
    // file << "SRC" << src; 

    // image dimension 
    const dnnl::memory::dim channel = src.channels(); 
    const dnnl::memory::dim height  = src.rows; 
    const dnnl::memory::dim width   = src.cols; 
    
    // image size 
    auto size = channel * height * width * sizeof(float); 

    // dst vector allocation 
    float *dst = (float*) malloc(size); 
    cv::Mat edge(height, width, CV_32FC3, dst); 

    // bias tensor: 1D tensor with OC dimension 
    // https://docs.oneapi.io/versions/latest/onednn/dev_guide_convolution.html
    // initialize bias to 0 (unused) 
    std::vector<float> bias(channel); 
    for (int i = 0; i < bias.size(); i++) bias[i] = 0.f;
    
    // edge detection kernel 
    const float laplacian[3][3] = { {1, 1, 1}, {1, -8, 1}, {1, 1, 1} };

    // 4d conv kernel 
    float filter[3][3][3][3];

    // copy laplacian to 4d tensor
    for (int i = 0; i < 3; i++) 
        for (int j = 0; j < 3; j++) 
            for (int k = 0; k < 3; k++) 
                for (int m = 0; m < 3; m++) 
                    filter[i][j][k][m] = laplacian[k][m];

    // create memory descriptor
    auto src_d    = memory::desc({1,channel,height,width}, memory::data_type::f32, memory::format_tag::nhwc);
    auto dst_d    = memory::desc({1,channel,height,width}, memory::data_type::f32, memory::format_tag::nhwc);
    auto filter_d = memory::desc({3,3,3,3}, memory::data_type::f32, memory::format_tag::oihw); 
    auto bias_d   = memory::desc({3}, memory::data_type::f32, memory::format_tag::x);

    // create memory object 
    auto src_mem    = sycl_interop::make_memory(src_d, engine, sycl_interop::memory_kind::buffer);
    auto dst_mem    = sycl_interop::make_memory(src_d, engine, sycl_interop::memory_kind::buffer);
    auto bias_mem   = sycl_interop::make_memory(bias_d, engine, sycl_interop::memory_kind::buffer);
    auto filter_mem = sycl_interop::make_memory(filter_d, engine, sycl_interop::memory_kind::buffer);
    
    // host -> device transfer
    write_to_dnnl_memory(src.ptr<float>(0), src_mem);
    write_to_dnnl_memory(filter, filter_mem);
    write_to_dnnl_memory(bias.data(), bias_mem);

    // create convolution_forward description
    auto conv_d = convolution_forward::desc(prop_kind::forward, algorithm::convolution_direct, src_d, filter_d, bias_d, dst_d, {1,1}, {1,1}, {1,1}); 
    auto conv_p = convolution_forward::primitive_desc(conv_d, engine); 
    auto conv   = convolution_forward(conv_p); 

    // perform convolution 
    conv.execute(
        stream, 
        {
            {DNNL_ARG_SRC,     src_mem},
            {DNNL_ARG_WEIGHTS, filter_mem},
            {DNNL_ARG_BIAS,    bias_mem}, 
            {DNNL_ARG_DST,     dst_mem}
        }
    ); 

    stream.wait(); 

    // device -> host 
    read_from_dnnl_memory(dst, dst_mem);

    // avoid negative pixels ?! 
    cv::threshold(edge, edge, 0, 0, cv::THRESH_TOZERO);
   
    // renormalize to RBG range 
    cv::normalize(edge, edge, 0.0, 255.0, cv::NORM_MINMAX);

    // convert back to 8bit (RGB) format 
    edge.convertTo(edge, CV_8UC3);

    // write to png file 
    cv::imwrite("detection.png", edge);

    return 0; 
}
