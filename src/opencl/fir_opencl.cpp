#define CL_TARGET_OPENCL_VERSION 120

#include "paths.h"

#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

const char *kKernel = R"CLC(
__kernel void fir_resident(__global const float *hist, __global const float *neu,
                           __global float *hist_out, __global float *y, __global const float *h,
                           int buf, int fir, __local float *sx) {
    const int ch = get_group_id(0);
    const int src = get_group_id(1);
    const int tid = get_local_id(0);
    const int nthreads = get_local_size(0);
    const int hist_n = fir - 1;
    for (int i = tid; i < hist_n; i += nthreads) {
        sx[i] = hist[src * hist_n + i];
    }
    for (int i = tid; i < buf; i += nthreads) {
        sx[hist_n + i] = neu[src * buf + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __global const float *hc = h + (src * 2 + ch) * fir;
    __global float *yc = y + (src * 2 + ch) * buf;
    for (int j = tid; j < buf; j += nthreads) {
        float acc = 0.0f;
        for (int k = 0; k < fir; ++k) {
            acc += sx[j + k] * hc[k];
        }
        yc[j] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ch == 0) {
        for (int i = tid; i < hist_n; i += nthreads) {
            hist_out[src * hist_n + i] = sx[buf + i];
        }
    }
}
)CLC";

void set_message(char *message, int message_bytes, const char *text) {
    if (message && message_bytes > 0) {
        std::snprintf(message, static_cast<size_t>(message_bytes), "%s", text);
    }
}

}  // namespace

int opencl_probe(char *message, int message_bytes) {
    cl_uint platforms = 0;
    const cl_int err = clGetPlatformIDs(0, nullptr, &platforms);
    if (err != CL_SUCCESS || platforms == 0) {
        char text[160];
        std::snprintf(text, sizeof(text), "平台不可用（clGetPlatformIDs 返回 %d，未安装厂商 ICD）",
                      static_cast<int>(err));
        set_message(message, message_bytes, text);
        return 1;
    }
    set_message(message, message_bytes, "OpenCL 平台可用");
    return 0;
}

int opencl_resident_measure(int buf, int fir, int sources, const float *x, const float *h_flip,
                            int total_blocks, int warmup, int measured, float *y_out, double *us_e2e,
                            double *us_kernel, double *us_copy, long *bytes_per_block) {
    const int hist = fir - 1;
    const int nx = hist + total_blocks * buf;
    if (warmup + measured > total_blocks) {
        return 1;
    }
    const size_t new_floats = static_cast<size_t>(sources) * static_cast<size_t>(buf);
    const size_t y_floats = static_cast<size_t>(sources) * 2u * static_cast<size_t>(buf);
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) {
        return 2;
    }
    cl_platform_id platform = nullptr;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device = nullptr;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS &&
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr) != CL_SUCCESS) {
        return 2;
    }
    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        return 1;
    }
    cl_command_queue queue =
        clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        return 1;
    }
    const size_t src_len = std::strlen(kKernel);
    cl_program program = clCreateProgramWithSource(context, 1, &kKernel, &src_len, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_bytes = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_bytes);
        std::string log(log_bytes, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_bytes, log.data(), nullptr);
        std::fprintf(stderr, "OpenCL build:\n%s\n", log.c_str());
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "fir_resident", &err);
    cl_mem d_hist[2] = {
        clCreateBuffer(context, CL_MEM_READ_WRITE, static_cast<size_t>(sources) * hist * sizeof(float), nullptr,
                       &err),
        clCreateBuffer(context, CL_MEM_READ_WRITE, static_cast<size_t>(sources) * hist * sizeof(float), nullptr,
                       &err)};
    cl_mem d_new = clCreateBuffer(context, CL_MEM_READ_ONLY, new_floats * sizeof(float), nullptr, &err);
    cl_mem d_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, y_floats * sizeof(float), nullptr, &err);
    cl_mem d_h = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                static_cast<size_t>(sources) * 2u * static_cast<size_t>(fir) * sizeof(float),
                                const_cast<float *>(h_flip), &err);
    float *pack = static_cast<float *>(std::malloc(static_cast<size_t>(total_blocks) * new_floats * sizeof(float)));
    if (!pack || err != CL_SUCCESS) {
        return 1;
    }
    for (int b = 0; b < total_blocks; ++b) {
        for (int s = 0; s < sources; ++s) {
            const float *src = x + static_cast<size_t>(s) * nx + hist + static_cast<size_t>(b) * buf;
            std::memcpy(pack + (static_cast<size_t>(b) * sources + s) * buf, src,
                        static_cast<size_t>(buf) * sizeof(float));
        }
    }
    const cl_uint zero = 0;
    clEnqueueFillBuffer(queue, d_hist[0], &zero, sizeof(zero), 0,
                        static_cast<size_t>(sources) * hist * sizeof(float), 0, nullptr, nullptr);
    clEnqueueFillBuffer(queue, d_hist[1], &zero, sizeof(zero), 0,
                        static_cast<size_t>(sources) * hist * sizeof(float), 0, nullptr, nullptr);

    const cl_int buf_i = buf;
    const cl_int fir_i = fir;
    const size_t local_mem = static_cast<size_t>(hist + buf) * sizeof(float);
    const size_t local[2] = {256, 1};
    const size_t global[2] = {512, static_cast<size_t>(sources)};

    for (int b = 0; b < total_blocks; ++b) {
        const int mi = b - warmup;
        const int rec = b >= warmup;
        cl_event write_ev = nullptr;
        cl_event kernel_ev = nullptr;
        cl_event read_ev = nullptr;
        clEnqueueWriteBuffer(queue, d_new, CL_FALSE, 0, new_floats * sizeof(float),
                             pack + static_cast<size_t>(b) * new_floats, 0, nullptr, &write_ev);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_hist[b % 2]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_new);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_hist[(b + 1) % 2]);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_y);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_h);
        clSetKernelArg(kernel, 5, sizeof(cl_int), &buf_i);
        clSetKernelArg(kernel, 6, sizeof(cl_int), &fir_i);
        clSetKernelArg(kernel, 7, local_mem, nullptr);
        clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 1, &write_ev, &kernel_ev);
        if (rec) {
            clEnqueueReadBuffer(queue, d_y, CL_FALSE, 0, y_floats * sizeof(float),
                                y_out + static_cast<size_t>(mi) * y_floats, 1, &kernel_ev, &read_ev);
            clFinish(queue);
            cl_ulong t0 = 0;
            cl_ulong t1 = 0;
            cl_ulong k0 = 0;
            cl_ulong k1 = 0;
            cl_ulong w1 = 0;
            clGetEventProfilingInfo(write_ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
            clGetEventProfilingInfo(write_ev, CL_PROFILING_COMMAND_END, sizeof(w1), &w1, nullptr);
            clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_START, sizeof(k0), &k0, nullptr);
            clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_END, sizeof(k1), &k1, nullptr);
            clGetEventProfilingInfo(read_ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
            cl_ulong r0 = 0;
            clGetEventProfilingInfo(read_ev, CL_PROFILING_COMMAND_START, sizeof(r0), &r0, nullptr);
            us_e2e[mi] = static_cast<double>(t1 - t0) / 1000.0;
            us_kernel[mi] = static_cast<double>(k1 - k0) / 1000.0;
            us_copy[mi] = static_cast<double>((w1 - t0) + (t1 - r0)) / 1000.0;
            clReleaseEvent(read_ev);
        } else {
            clFinish(queue);
        }
        clReleaseEvent(write_ev);
        clReleaseEvent(kernel_ev);
    }
    *bytes_per_block =
        static_cast<long>(sources) * static_cast<long>(buf + 2 * buf) * static_cast<long>(sizeof(float));
    std::free(pack);
    clReleaseMemObject(d_hist[0]);
    clReleaseMemObject(d_hist[1]);
    clReleaseMemObject(d_new);
    clReleaseMemObject(d_y);
    clReleaseMemObject(d_h);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
