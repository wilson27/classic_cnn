#include <algorithm>
#include <float.h>
#include "maxpool_layer.h"
#include <iostream>

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
                const int k)            // kernel size
{

// Global memory interface
#pragma HLS INTERFACE m_axi port=mem depth=2147483648
// Bind all control ports to a single bundle
#pragma HLS INTERFACE s_axilite port=b bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=od bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=ox bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=oy bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=id bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=ix bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=iy bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=s bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=k bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=input_offset
#pragma HLS INTERFACE s_axilite port=output_offset
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
 
  int num_weights = id*od*k*k;
  int num_biases = od;
  int num_input = b*id*ix*iy;
  int num_output = b*od*ox*oy;
  float temp; //hold the next value
 
  // Batch
  for (int b_=0; b_< b; b_++)
  {
    // Output Dimensions (Feature Maps)
    for (int o_d = 0; o_d < od; o_d++)
    {
      // Output Y Dimension
      for (int o_y = 0; o_y < oy; o_y++)
      {
        // Output X Dimension
        for (int o_x = 0; o_x < ox; o_x++)
        {
	  float output_element = -9000;
          // Weighted Sum:

          // Input Dimensions (Feature Maps)
          //for (int i_d = 0; i_d < id; i_d++)
          //{
		int i_d = o_d;
            // Input Y Dimension
            for (int i_y = o_y*s; i_y < o_y*s+k; i_y++)
            {
              // Input X Dimension
              for (int i_x = o_x*s; i_x < o_x*s+k; i_x++)
              {
#pragma HLS PIPELINE
		 temp = mem[input_offset/sizeof(float) + b_*id*ix*iy + i_d*ix*iy + i_y*ix + i_x];
		 output_element = std::max(temp, output_element);
              }
            }
          // Write output
          mem[output_offset/sizeof(float) + b_*od*ox*oy + o_d*ox*oy + o_y*ox + o_x] = output_element;
          //}


        }
      }
    }
  }
}

