//
//  main.c
//  VectorAdditionOpenCL
//
//  Created by Can Firtina on 24/07/15.
//  Copyright (c) 2015 Can Firtina. All rights reserved.
//

#include <stdio.h>
#include <OpenCL/OpenCL.h>

// Simple compute kernel which computes the addition of a two vector
const char *kernelSource = "\n" \
"__kernel void vectorAddition(                                          \n" \
"   __global float *vectorA,                                            \n" \
"   __global float *vectorB,                                            \n" \
"   __global float *vectorC,                                            \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       vectorC[i] = vectorA[i] + vectorB[i];                           \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

#define NUM_OF_VALUES 100000

int main( int argc, const char * argv[]) {
    
    cl_int clerr = CL_SUCCESS;
    
    //create context that includes gpu devices in it
    cl_context clContext = clCreateContextFromType( NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &clerr);
    if( clerr != CL_SUCCESS) {
        
        printf("Error during clCreateContextFromType\n");
        return EXIT_FAILURE;
    }
    
    //we will get the array of devices, but first we need to get the size (in bytes) of it
    size_t devicesSize;
    clerr = clGetContextInfo( clContext, CL_CONTEXT_DEVICES, 0, NULL, &devicesSize);
    if( clerr != CL_SUCCESS) {
        
        printf("Error during clGetContextInfo to get deviesSize\n");
        return EXIT_FAILURE;
    }
    
    //now we say that here is the array with devicesSize (in bytes). Fill it with devices
    cl_device_id *clDevices = (cl_device_id *)malloc(devicesSize);
    clerr = clGetContextInfo( clContext, CL_CONTEXT_DEVICES, devicesSize, clDevices, NULL);
    if( clerr != CL_SUCCESS) {
        
        printf("Error during clGetContextInfo to fill clDevices\n");
        return EXIT_FAILURE;
    }
    
    cl_uint numOfDevices;
    clerr = clGetContextInfo( clContext, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &numOfDevices, NULL);
    
    cl_uint maxComputeUnits = 0;
    cl_uint maxClockFrequency = 0;
    cl_device_id gpuId = clDevices[0];
    cl_uint id = 0;
    
    for( int curDevice = 0; curDevice < numOfDevices; curDevice++){
        
        cl_uint computeUnits;
        cl_uint clockFrequency;
        clerr = clGetDeviceInfo( clDevices[curDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits, NULL);
        clerr |= clGetDeviceInfo( clDevices[curDevice], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &clockFrequency, NULL);
        
        if( clerr != CL_SUCCESS) {
            
            printf("Error during clGetDeviceInfo\n");
            return EXIT_FAILURE;
        }
        
        if( maxComputeUnits * maxClockFrequency < computeUnits * clockFrequency){
            
            maxComputeUnits = computeUnits;
            maxClockFrequency = clockFrequency;
            gpuId = clDevices[curDevice];
            id = curDevice;
        }
    }
    
    unsigned int numOfValues = NUM_OF_VALUES;
    float vectorA[NUM_OF_VALUES];
    float vectorB[NUM_OF_VALUES];
    float vectorC[NUM_OF_VALUES];
    
    for(int i = 0; i < NUM_OF_VALUES; i++){
        
        vectorA[i] = rand() / (float)RAND_MAX;
        vectorB[i] = rand() / (float)RAND_MAX;
    }
    
    cl_command_queue clCommandQueue = clCreateCommandQueue( clContext, gpuId, 0, &clerr);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clCommandQueues\n");
        return EXIT_FAILURE;
    }
    
    cl_program clProgram = clCreateProgramWithSource( clContext, 1, (const char **)&kernelSource, NULL, &clerr);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clCreateProgramWithSource\n");
        return EXIT_FAILURE;
    }
    
    //now the program object has been created with the source code specified for the devices
    //associated with the clProgram, which is also associated with clContext.
    clerr = clBuildProgram( clProgram, 1, &gpuId, NULL, NULL, NULL);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clBuildProgram\n");
        return EXIT_FAILURE;
    }
    
    //Here program compiled. Now it needs to be executed. So we will create executable (kernel) for
    //that now... Worth to note that when you give a name that is different than the function name
    //that you have written for the kernel, clerr is not set CL_SUCCESS.
    cl_kernel clKernel = clCreateKernel( clProgram, "vectorAddition", &clerr);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clCreateKernel\n");
        return EXIT_FAILURE;
    }
    
    cl_mem d_vectorA, d_vectorB, d_vectorC;
    size_t dataSize = NUM_OF_VALUES*sizeof(float);
    
    d_vectorA = clCreateBuffer( clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, vectorA, NULL);
    d_vectorB = clCreateBuffer( clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, vectorB, NULL);
    d_vectorC = clCreateBuffer( clContext, CL_MEM_WRITE_ONLY, dataSize, NULL, NULL);
    
    if( !d_vectorA || !d_vectorB || !d_vectorC){
        
        printf("Error during clCreateBuffer\n");
        return EXIT_FAILURE;
    }
    
    clerr = 0;
    clerr = clSetKernelArg( clKernel, 0, sizeof(cl_mem), (void *)&d_vectorA);
    clerr &= clSetKernelArg( clKernel, 1, sizeof(cl_mem), (void *)&d_vectorB);
    clerr &= clSetKernelArg( clKernel, 2, sizeof(cl_mem), (void *)&d_vectorC);
    clerr &= clSetKernelArg( clKernel, 3, sizeof(unsigned int), (void *)&numOfValues);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clSetKernelArg\n");
        return EXIT_FAILURE;
    }
    
    size_t localWorkGroupSize;
    clerr = clGetKernelWorkGroupInfo( clKernel, gpuId, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(size_t), &localWorkGroupSize, NULL);
    printf("info: local work group size for device %d is %zu\n", id, localWorkGroupSize);
    if( clerr != CL_SUCCESS){
        
        printf("Error during clGetKernelWorkGroupInfo for device id %d\n", id);
        return EXIT_FAILURE;
    }
    
    //the only constraint for the global_work_size is that it must be a multiple of the
    //local_work_size (for each dimension).
    size_t globalWorkItems = (numOfValues/localWorkGroupSize + 1)*localWorkGroupSize;
    clerr = clEnqueueNDRangeKernel( clCommandQueue, clKernel, 1, NULL, &globalWorkItems,
                                   &localWorkGroupSize, 0, NULL, NULL);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clEnqueueNDRangeKernel\n");
        return EXIT_FAILURE;
    }
    
    clFinish(clCommandQueue);
    
    clerr = clEnqueueReadBuffer( clCommandQueue, d_vectorC, CL_TRUE, 0,
                                sizeof(float)*numOfValues, vectorC, 0, NULL, NULL);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clEnqueueReadBuffer\n");
        return EXIT_FAILURE;
    }
    
    // Validate our results
    int correct = 0;
    for(int i = 0; i < numOfValues; i++){
        if(vectorC[i] == vectorA[i] + vectorB[i])
            correct++;
        else
            printf("%d %f %f\n", i, vectorC[i], vectorA[i] + vectorB[i]);
    }
    
    // Print a brief summary detailing the results
    printf("Computed '%d/%d' correct values!\n", correct, numOfValues);
    
    // Shutdown and cleanup
    clReleaseMemObject(d_vectorA);
    clReleaseMemObject(d_vectorB);
    clReleaseMemObject(d_vectorC);
    clReleaseProgram(clProgram);
    clReleaseKernel(clKernel);
    clReleaseCommandQueue(clCommandQueue);
    clReleaseContext(clContext);
    
    return 0;
}
