
//https://cnugteren.github.io/tutorial/pages/page2.html
__kernel void mxMul(__global const float* A, __global const float* B, __global const int* mxSizes, __global float* output) 
{
    const int M = mxSizes[0];
    const int N = mxSizes[1];
    const int K = mxSizes[2];

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; ++k) {
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    }
 
    // Store the result
    C[globalCol*M + globalRow] = acc;
}
