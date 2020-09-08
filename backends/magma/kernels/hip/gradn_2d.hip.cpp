// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed.h>
#include "hip/hip_runtime.h"
#include <magma_v2.h>
#include "../common/magma_common_device.h"
#include "../common/grad_device.h"

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int NCOMP, int P, int Q, int MAXPQ>
static __global__ void
magma_gradn_2d_kernel(
    const T *dinterp1d, const T *dgrad1d, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, const int dstrdU,  
          T *dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem)
{
    HIP_DYNAMIC_SHARED( CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    T rU[1][NCOMP][P] = { make_zero<T>() };  // here DIMU = 1, but might be different for a fused operator
    T rV[1][NCOMP][Q] = { make_zero<T>() };  // here DIMV = 1, but might be different for a fused operator
    T rTmp = make_zero<T>();

    // shift global memory pointers by elem stride
    dU += elem_id * estrdU;
    dV += elem_id * estrdV;

    // assign shared memory pointers
    T* sTinterp = (T*)(shared_data);
    T* sTgrad   = sTinterp + P*Q;
    T* sTmp     = sTgrad   + P*Q;
    sTmp       += ty * (P * MAXPQ);

    // read T
    if (ty == 0) {
        dread_T_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp);
        dread_T_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad);
    }

    // No need to read V ( required only in transposed grad )
    const T beta = make_zero<T>();

    /* read U (idim = 0 for dU, iDIM = 0 for rU) -- 
       there is a sync at the end of this function */
    readU_2d<T, P, 1, NCOMP, P, 0>
    (dU + (0*dstrdU), cstrdU, rU, sTmp, tx);

    /* first call (iDIM = 0, iDIMU = 0, iDIMV = 0) -- 
       output from rV[0][][] into dV (idim = 0) */
    magma_grad_2d_device<T, 1, 1, NCOMP, P, Q, P, Q, 0, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp); 
    /* there is a sync at the end of magma_grad_2d_device */
    writeV_2d<T, Q, 1, NCOMP, Q, 0>
    (dV+(0*dstrdV), cstrdV, rV, tx);

    /* second call (iDIM = 1, iDIMU = 0, iDIMV = 0) -- 
    output from rV[0][][] into dV (idim = 1) */
    magma_grad_2d_device<T, 1, 1, NCOMP, P, Q, P, Q, 1, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_2d_device */
    writeV_2d<T, Q, 1, NCOMP, Q, 0>
    (dV+(1*dstrdV), cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int NCOMP, int P, int Q>
static magma_int_t 
magma_gradn_2d_kernel_driver(  
                const T *dinterp1d, const T *dgrad1d, magma_trans_t transT,
                const T *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU, 
                      T *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV, 
                magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t shmem_max, nthreads_max;
    const int MAXPQ = maxpq(P,Q);

    magma_int_t nthreads = MAXPQ; 
    magma_int_t ntcol = (maxthreads < nthreads) ? 1 : (maxthreads / nthreads);
    magma_int_t shmem  = 0;
    shmem += sizeof(T) * 2*P*Q;  // for sTinterp and sTgrad
    shmem += sizeof(T) * ntcol * (P*MAXPQ);  // for reforming rU we need PxP, and for the intermediate output we need PxQ    

    hipDeviceGetAttribute (&nthreads_max, hipDeviceAttributeMaxThreadsPerBlock, device);
    hipDeviceGetAttribute (&shmem_max, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
   
    if ( (nthreads*ntcol) > nthreads_max || shmem > shmem_max ) {
        return 1;    // launch failed
    }
    else { 
        magma_int_t nblocks = (nelem + ntcol-1) / ntcol;
        dim3 threads(nthreads, ntcol, 1);
        dim3 grid(nblocks, 1, 1);
        // IMPORTANT: we instantiate with DIM=1 instead of DIM=2 because the kernel handles one dimension at a time
        // We should instantiate with DIM >= 1 when we fuse the whole operator, because of the q-function
        hipLaunchKernelGGL(HIP_KERNEL_NAME(magma_gradn_2d_kernel<T,NCOMP,P,Q,MAXPQ>), dim3(grid), dim3(threads), shmem, magma_queue_get_hip_stream(queue), dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem);
        return (hipPeekAtLastError() == hipSuccess) ? 0 : 1;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
static magma_int_t 
magma_gradn_2d_ncomp(
                magma_int_t ncomp,
                const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU,
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV,
                magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (ncomp) {
        case 1: 
          launch_failed = magma_gradn_2d_kernel_driver<CeedScalar,1,P,Q>
          (dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case 2: 
          launch_failed = magma_gradn_2d_kernel_driver<CeedScalar,2,P,Q>
          (dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case 3: 
          launch_failed = magma_gradn_2d_kernel_driver<CeedScalar,3,P,Q>
          (dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magma_gradn_2d_ncomp_q(
                magma_int_t Q, magma_int_t ncomp,
                const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU,
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV,
                magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (Q) {
        case  1: 
          launch_failed = magma_gradn_2d_ncomp<P, 1>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  2: 
          launch_failed = magma_gradn_2d_ncomp<P, 2>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  3: 
          launch_failed = magma_gradn_2d_ncomp<P, 3>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  4: 
          launch_failed = magma_gradn_2d_ncomp<P, 4>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  5: 
          launch_failed = magma_gradn_2d_ncomp<P, 5>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  6: 
          launch_failed = magma_gradn_2d_ncomp<P, 6>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  7: 
          launch_failed = magma_gradn_2d_ncomp<P, 7>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  8: 
          launch_failed = magma_gradn_2d_ncomp<P, 8>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  9: 
          launch_failed = magma_gradn_2d_ncomp<P, 9>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case 10: 
          launch_failed = magma_gradn_2d_ncomp<P,10>
          (ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}


//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_gradn_2d_ncomp_q_p(
                magma_int_t P, magma_int_t Q, magma_int_t ncomp,
                const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
               const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU,
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV,
                magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (P) {
        case  1: 
          launch_failed = magma_gradn_2d_ncomp_q< 1>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  2: 
          launch_failed = magma_gradn_2d_ncomp_q< 2>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  3: 
          launch_failed = magma_gradn_2d_ncomp_q< 3>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  4: 
          launch_failed = magma_gradn_2d_ncomp_q< 4>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  5: 
          launch_failed = magma_gradn_2d_ncomp_q< 5>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  6: 
          launch_failed = magma_gradn_2d_ncomp_q< 6>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  7: 
          launch_failed = magma_gradn_2d_ncomp_q< 7>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  8: 
          launch_failed = magma_gradn_2d_ncomp_q< 8>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case  9: 
          launch_failed = magma_gradn_2d_ncomp_q< 9>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        case 10: 
          launch_failed = magma_gradn_2d_ncomp_q<10>
          (Q, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV, dstrdV, nelem, maxthreads, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}



//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_gradn_2d( 
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,  
    const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode, 
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU,
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV,
    magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    launch_failed = magma_gradn_2d_ncomp_q_p(
                        P, Q, ncomp, 
                        dinterp1d, dgrad1d, transT, 
                        dU, estrdU, cstrdU, dstrdU, 
                        dV, estrdV, cstrdV, dstrdV, 
                        nelem, maxthreads, queue);

    return launch_failed;
}
