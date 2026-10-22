#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/mman.h>

int main(void) {
    const size_t bytes = 64 * 4096;
    void *page = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    cudaError_t init_err;
    cudaError_t reg_err;

    if (page == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    init_err = cudaFree(0);
    if (init_err != cudaSuccess) {
        printf("cudaHostRegister: FAIL (cuda init: %s)\n", cudaGetErrorString(init_err));
        munmap(page, bytes);
        return 0;
    }

    reg_err = cudaHostRegister(page, bytes, cudaHostRegisterDefault);
    if (reg_err == cudaSuccess) {
        printf("cudaHostRegister: SUCCESS\n");
        cudaHostUnregister(page);
    } else {
        printf("cudaHostRegister: FAIL (%s)\n", cudaGetErrorString(reg_err));
    }
    munmap(page, bytes);
    return 0;
}
