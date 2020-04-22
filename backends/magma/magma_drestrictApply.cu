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
#include <magma.h>
#include "atomics.cuh"

//////////////////////////////////////////////////////////////////////////////////////////
// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from L-vector (du) to E-vector (dv):
//
// dv(i, e, c) = du( offsets(i, e) + compstride * c)  
static __global__ void 
magma_readDofsOffset_kernel(const int NCOMP, const int compstride,
                            const int esize, const int nelem, int *offsets, 
                            const double *du, double *dv)
{
  const int  pid = threadIdx.x;
  const int elem = blockIdx.x;
 
  for (CeedInt i = pid; i < esize; i += blockDim.x) {
        const CeedInt ind = offsets ? offsets[i + elem * esize] : i + elem * esize;
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            dv[i+elem*esize+comp*esize*nelem] = du[ind + compstride * comp];
        }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from L-vector (du) to E-vector (dv), with strides provided 
//  to describe the L-vector layout
//
// dv(i, e, c) = du( i * strides[0] + c * strides[1] + e * strides[2] )  
static __global__ void 
magma_readDofsStrided_kernel(const int NCOMP, const int esize, const int nelem,
                             const int *strides, const double *du, double *dv)
{
  const int  pid = threadIdx.x;
  const int elem = blockIdx.x;
 
  for (CeedInt i = pid; i < esize; i += blockDim.x) {
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            dv[i+elem*esize+comp*esize*nelem] = du[i * strides[0] + 
                                                   comp * strides[1] + 
                                                   elem * strides[2]];
        }
  }
}


// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from E-vector (du) to L-vector (dv):
//
// dv(offsets(i, e) + compstride * c) = du(i, e, c)
static __global__ void 
magma_writeDofsOffset_kernel(const int NCOMP, const int compstride,
                             const int esize, const int nelem, int *offsets, 
                             const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    for (CeedInt i = pid; i < esize; i += blockDim.x) {
        const CeedInt ind = offsets ? offsets[i + elem * esize] : i + elem * esize;
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            magmablas_datomic_add(dv + (ind + compstride * comp),
                                  du[i+elem*esize+comp*esize*nelem]);
        }
    }
}

// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from E-vector (du) to L-vector (dv), with strides provided 
//  to describe the L-vector layout
//
// dv( i * strides[0] + c * strides[1] + e * strides[2] ) = du(i, e, c) 
static __global__ void 
magma_writeDofsStrided_kernel(const int NCOMP, const int esize, const int nelem,
                              const int *strides, const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    for (CeedInt i = pid; i < esize; i += blockDim.x) {
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            magmablas_datomic_add(dv + (i * strides[0] + comp * strides[1] + 
                                        elem * strides[2]),
                                  du[i+elem*esize+comp*esize*nelem]);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

// ReadDofs to device memory
// du is L-vector, size lsize
// dv is E-vector, size nelem * esize * NCOMP
extern "C" void
magma_readDofsOffset(const magma_int_t NCOMP, const magma_int_t compstride,
                     const magma_int_t esize, const magma_int_t nelem,
                     magma_int_t *offsets, const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_readDofsOffset_kernel<<<grid, threads, 0, NULL>>>(NCOMP, compstride,
      esize, nelem, offsets, du, dv);
}

// ReadDofs to device memory, strided description for L-vector
// du is L-vector, size lsize
// dv is E-vector, size nelem * esize * NCOMP
extern "C" void
magma_readDofsStrided(const magma_int_t NCOMP, const magma_int_t esize,
                      const magma_int_t nelem, const int *strides,
                      const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_readDofsStrided_kernel<<<grid, threads, 0, NULL>>>(NCOMP, esize,
      nelem, strides, du, dv);
}

// WriteDofs from device memory
// du is E-vector, size nelem * esize * NCOMP
// dv is L-vector, size lsize 
extern "C" void
magma_writeDofsOffset(const magma_int_t NCOMP, const magma_int_t compstride,
                      const magma_int_t esize, const magma_int_t nelem,
                      magma_int_t *offsets, const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_writeDofsOffset_kernel<<<grid, threads, 0, NULL>>>(NCOMP, compstride,
      esize, nelem, offsets, du, dv);
}

// WriteDofs from device memory, strided description for L-vector
// du is E-vector, size nelem * esize * NCOMP
// dv is L-vector, size lsize
extern "C" void
magma_writeDofsStrided(const magma_int_t NCOMP, const magma_int_t esize,
                       const magma_int_t nelem, const int *strides,
                       const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_writeDofsStrided_kernel<<<grid, threads, 0, NULL>>>(NCOMP, esize,
      nelem, strides, du, dv);
}
