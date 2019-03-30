#ifndef _CONV_LAYER_H
#define _CONV_LAYER_H

// Limits
#define MAX_BATCH_CONV 10
#define MAX_KERNEL_SIZE_CONV 5
#define MAX_INPUT_DIMS_CONV 20
#define MAX_OUTPUT_DIMS_CONV 50
#define MAX_INPUT_WIDTH_CONV 30
#define MAX_INPUT_HEIGHT_CONV 30
#define MAX_OUTPUT_WIDTH_CONV 24
#define MAX_OUTPUT_HEIGHT_CONV 24
#define MAX_CONV_INPUT MAX_INPUT_DIMS_CONV*MAX_INPUT_WIDTH_CONV*MAX_INPUT_HEIGHT_CONV 
#define MAX_CONV_OUTPUT MAX_OUTPUT_DIMS_CONV*MAX_OUTPUT_WIDTH_CONV*MAX_OUTPUT_HEIGHT_CONV 
#define MAX_WEIGHT_SIZE_CONV MAX_OUTPUT_DIMS_CONV*MAX_INPUT_DIMS_CONV*MAX_KERNEL_SIZE_CONV*MAX_KERNEL_SIZE_CONV

void conv_layer(float * mem,            // global memory pointer
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
void hw_conv_layer(int target,             // control register target
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
