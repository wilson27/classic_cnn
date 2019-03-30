#ifndef _MAXPOOL_LAYER_H
#define _MAXPOOL_LAYER_H

// Limits
#define MAX_BATCH_POOL 10
#define MAX_KERNEL_SIZE_POOL 2
#define MAX_INPUT_DIMS_POOL 50
#define MAX_OUTPUT_DIMS_POOL 50
#define MAX_INPUT_WIDTH_POOL 24
#define MAX_INPUT_HEIGHT_POOL 24
#define MAX_OUTPUT_WIDTH_POOL 12
#define MAX_OUTPUT_HEIGHT_POOL 12
#define MAX_MAXPOOL_INPUT MAX_INPUT_DIMS_POOL*MAX_INPUT_WIDTH_POOL*MAX_INPUT_HEIGHT_POOL 
#define MAX_MAXPOOL_OUTPUT MAX_OUTPUT_DIMS_POOL*MAX_OUTPUT_WIDTH_POOL*MAX_OUTPUT_HEIGHT_POOL 
#define MAX_WEIGHT_SIZE_POOL MAX_OUTPUT_DIMS_POOL*MAX_INPUT_DIMS_POOL*MAX_KERNEL_SIZE_POOL*MAX_KERNEL_SIZE_POOL

void maxpool_layer(float * mem,            // global memory pointer
                int input_offset,       // offset of inputs
                int output_offset,      // offset of outputs
                const int b,            // batch size
                const int od,           // output dimensions
                const int ox,           // output width
                const int oy,           // output height
                const int id,           // input dimensions
                const int ix,           // input width
                const int iy,           // input height
                const int s,            // stride
                const int k);           // kernel size
void hw_maxpool_layer(int target,             // control register target
                   float * mem,            // global memory pointer
                   int input_offset,       // offset of inputs
                   int output_offset,      // offset of outputs
                   const int b,            // batch size
                   const int od,           // output dimensions
                   const int ox,           // output width
                   const int oy,           // output height
                   const int id,           // input dimensions
                   const int ix,           // input width
                   const int iy,           // input height
                   const int s,            // stride
                   const int k);           // kernel size
#endif
