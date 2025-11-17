#include "ocs_kf.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace ocs {

// full 7x7 KF (batched)
__device__ __forceinline__ float& Pidx(float* P7x7, int r, int c){ return P7x7[r*7 + c]; }
__device__ __forceinline__ float  Pget(const float* P7x7, int r, int c){ return P7x7[r*7 + c]; }

__global__ void kf_predict_kernel(float* x, float* P, int T, float dt, float q_pos, float q_size){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=T) return;
    float* xi = x + i*7; float* Pi = P + i*49;
    // x = F x ; F applies: cx+=vx*dt, cy+=vy*dt, s+=vs*dt, r unchanged
    xi[0] += xi[4]*dt; // cx
    xi[1] += xi[5]*dt; // cy
    xi[2] += xi[6]*dt; // s
    // r keep; velocities unchanged
    // P' = P + N P + P N^T + N P N^T + Q; N has (0,4)=(1,5)=(2,6)=dt
    float d = dt;
    // Make a copy of P to start Pnew
    float Pnew[49];
    #pragma unroll
    for (int r=0;r<7;++r){
        #pragma unroll
        for (int c=0;c<7;++c){ Pnew[r*7+c] = Pget(Pi,r,c); }
    }
    // For pair (p=0,v=4), (p=1,v=5), (p=2,v=6)
    int p_idx[3] = {0,1,2};
    int v_idx[3] = {4,5,6};
    #pragma unroll
    for (int k=0;k<3;++k){
        int p = p_idx[k], v = v_idx[k];
        // row update: Pnew[p,*] += d * P[v,*]
        for (int c=0;c<7;++c){ Pnew[p*7 + c] += d * Pget(Pi, v, c); }
        // col update: Pnew[*,p] += d * P[*,v]
        for (int r=0;r<7;++r){ Pnew[r*7 + p] += d * Pget(Pi, r, v); }
        // NPN^T term on (p,p)
        Pnew[p*7 + p] += d*d * Pget(Pi, v, v);
    }
    // Add Q (diag)
    Pnew[0*7+0] += q_pos;
    Pnew[1*7+1] += q_pos;
    Pnew[2*7+2] += q_size;
    Pnew[3*7+3] += q_size*0.25f;
    Pnew[4*7+4] += q_pos*0.5f;
    Pnew[5*7+5] += q_pos*0.5f;
    Pnew[6*7+6] += q_size*0.5f;
    // write back
    #pragma unroll
    for (int r=0;r<7;++r){
        #pragma unroll
        for (int c=0;c<7;++c){ Pidx(Pi,r,c) = Pnew[r*7+c]; }
    }
}

__device__ bool inv4x4(const float A[16], float invOut[16]){
    // Gauss-Jordan elimination for 4x4
    float m[16]; float inv[16] = {0};
    #pragma unroll
    for(int i=0;i<16;++i) m[i]=A[i];
    inv[0]=1; inv[5]=1; inv[10]=1; inv[15]=1;
    for(int col=0; col<4; ++col){
        // find pivot
        int piv = col; float maxv = fabsf(m[col*4+col]);
        for(int r=col+1;r<4;++r){ float v=fabsf(m[r*4+col]); if(v>maxv){maxv=v; piv=r;} }
        if (maxv < 1e-8f) return false;
        // swap rows
        if (piv!=col){ for(int c=0;c<4;++c){ float t=m[col*4+c]; m[col*4+c]=m[piv*4+c]; m[piv*4+c]=t; float ti=inv[col*4+c]; inv[col*4+c]=inv[piv*4+c]; inv[piv*4+c]=ti; } }
        // normalize
        float diag = m[col*4+col]; float invdiag = 1.0f/diag;
        for(int c=0;c<4;++c){ m[col*4+c]*=invdiag; inv[col*4+c]*=invdiag; }
        // eliminate
        for(int r=0;r<4;++r){ if(r==col) continue; float f=m[r*4+col]; if (f==0) continue; for(int c=0;c<4;++c){ m[r*4+c]-=f*m[col*4+c]; inv[r*4+c]-=f*inv[col*4+c]; } }
    }
    #pragma unroll
    for(int i=0;i<16;++i) invOut[i]=inv[i];
    return true;
}

__global__ void kf_update_kernel(float* x, float* P, const float* z, const int* m, int T,
                                 float r_pos, float r_size){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=T) return;
    if (!m[i]) return; // only matched tracks update
    float* xi = x + i*7; float* Pi = P + i*49; const float* zi = z + i*4;
    // H selects indices 0,1,2,3
    int idx4[4] = {0,1,2,3};
    float Rdiag[4] = { r_pos, r_pos, r_size, r_size*0.25f };
    // build S = H P H^T + R (4x4)
    float S[16];
    for (int a=0;a<4;++a){
        for (int b=0;b<4;++b){
            float v = Pget(Pi, idx4[a], idx4[b]);
            if (a==b) v += Rdiag[a];
            S[a*4+b] = v;
        }
    }
    // invert S
    float Sinv[16]; bool ok = inv4x4(S, Sinv);
    if (!ok){ // add jitter and retry once
        for (int d=0; d<4; ++d) S[d*4+d] += 1e-3f;
        if (!inv4x4(S, Sinv)) return;
    }
    // PHt = P H^T -> 7x4, PHt[r,c] = P[r, idx4[c]]
    float PHt[28];
    for (int r=0;r<7;++r){ for (int c=0;c<4;++c){ PHt[r*4+c] = Pget(Pi, r, idx4[c]); } }
    // K = PHt * Sinv -> 7x4
    float K[28];
    for (int r=0;r<7;++r){ for (int c=0;c<4;++c){ float sum=0.f; for (int t0=0;t0<4;++t0){ sum += PHt[r*4+t0] * Sinv[t0*4+c]; } K[r*4+c]=sum; } }
    // y = z - Hx
    float y[4]; for (int a=0;a<4;++a){ y[a] = zi[a] - xi[idx4[a]]; }
    // x = x + K y
    for (int r=0;r<7;++r){ float add=0.f; for (int c=0;c<4;++c) add += K[r*4+c]*y[c]; xi[r]+=add; }
    // P = P - K * (H P)
    float Pnew[49];
    for (int r=0;r<7;++r){
        for (int c=0;c<7;++c){
            // compute (K * (H P))[r,c] = sum_{a=0..3} K[r,a] * P[idx4[a], c]
            float sum=0.f; for (int a=0;a<4;++a){ sum += K[r*4+a] * Pget(Pi, idx4[a], c); }
            Pnew[r*7+c] = Pget(Pi, r, c) - sum;
        }
    }
    // symmetrize: P = 0.5*(P + P^T)
    for (int r=0;r<7;++r){ for (int c=r+1;c<7;++c){ float v = 0.5f*(Pnew[r*7+c] + Pnew[c*7+r]); Pnew[r*7+c]=v; Pnew[c*7+r]=v; } }
    for (int r=0;r<7;++r){ for (int c=0;c<7;++c){ Pidx(Pi,r,c) = Pnew[r*7+c]; } }
    // simple velocity momentum adjust towards innovation on pos/size
    xi[4] = xi[4]*0.9f + 0.1f*y[0];
    xi[5] = xi[5]*0.9f + 0.1f*y[1];
    xi[6] = xi[6]*0.9f + 0.1f*y[2];
}

void kf_predict(float* x, float* P, int T, float dt, float q_pos, float q_size, cudaStream_t stream){
    int tb=256, gb=(T+tb-1)/tb; kf_predict_kernel<<<gb,tb,0,stream>>>(x,P,T,dt,q_pos,q_size);
}

void kf_update(float* x, float* P, const float* z, const int* matched_mask, int T,
               float r_pos, float r_size, cudaStream_t stream){
    int tb=256, gb=(T+tb-1)/tb; kf_update_kernel<<<gb,tb,0,stream>>>(x,P,z,matched_mask,T,r_pos,r_size);
}

} // namespace ocs
