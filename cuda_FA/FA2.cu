#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel_v2(const float* Q,const float* K,const float* V,const int N,const int d,
        const int Tc,const int Tr,const int Bc,const int Br,const float softmax_scale,
        float* l,float* m,float* O){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int qkv_offset = (bx*gridDim.y*N*d)+(by*N*d);
    int lm_offset = (bx*gridDim.y*N)+(by*N);

    extern __shared__ float sram[];
    const int tile_size_qo = Br*d;//size of Qi,Oi
    const int tile_size_kv = Bc*d;//size of Kj,Vj
    float* Qi = sram;
    float* Oi = &sram[tile_size_qo];
    float* Kj = &sram[tile_size_qo*2];
    float* Vj = &sram[tile_size_qo*2+tile_size_kv];
    float* S = &sram[tile_size_qo*2+tile_size_kv*2];
    
    for(int i=0;i<Tr;i++){
        //load Oi,Qi to SRAM
        for(int x=0;x<d;x++){
            Qi[(tx*d)+x] = Q[qkv_offset +(tile_size_qo*i)+(tx*d)+x];
            Oi[(tx*d)+x] = O[qkv_offset +(tile_size_qo*i)+(tx*d)+x];
        }
        __syncthreads();
        float row_m_prev = m[lm_offset+(Br*i)+tx];
        float row_l_prev = l[lm_offset+(Br*i)+tx];
        float row_m_new,row_l_new;

        for(int j=0;j<Tc;j++){
            //load Kj,Vj to SRAM
            for(int x=0;x<d;x++){
                Kj[(tx*d)+x] = K[qkv_offset+(tile_size_kv*j)+(tx*d)+x];
                Vj[(tx*d)+x] = V[qkv_offset+(tile_size_kv*j)+(tx*d)+x];
            }

            float row_m = -INFINITY;
            for(int y=0;y<Bc;y++){
                float sum =0;
                for(int x=0;x<d;x++){
                    sum+=Qi[(tx*d)+x]*Kj[(y*d)+x];
                }
                sum *= softmax_scale;
                S[(Bc*tx)+y] = sum;
                if(sum > row_m) row_m=sum;
            }
            //max mi
            row_m_new = max(row_m_prev,row_m);
            //P=exp(S-row_m),row_l = rowsum(P)
            float row_l = 0;
            for(int y=0;y<Bc;y++){
                S[(Bc*tx)+y]=__expf(S[(Bc*tx)+y]-row_m_new);
                row_l += S[(Bc*tx)+y];
            }
            //Compute new l
            row_l_prev = (__expf(row_m_prev - row_m_new)*row_l_prev)+row_l;

            //write O,l,m to HBM
            for(int x=0;x<d;x++){
                float pv = 0;
                for(int y=0;y<Bc;y++){
                    pv+=S[(Bc*tx)+y]*Vj[(y*d)+x];
                }
                Oi[(tx*d)+x]=
                    ((__expf(row_m_prev-row_m_new)*Oi[(tx*d)+x])+pv);
            }
            row_m_prev = row_m_new;
        }
        for(int x=0;x<d;x++){
            O[qkv_offset+(tile_size_qo*i)+(tx*d)+x]=Oi[(tx*d)+x]/row_l_prev;
        }
        __syncthreads();
    }
}

