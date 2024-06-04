// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Stiffness operator for Navier-Stokes example using PETSc
#include <ceed.h>
#include <math.h>
#include "utils.h"
// *****************************************************************************
// This QFunction applies the stiffness matrix to a single field. Support for
// additional fields to come.
//
// Inputs:
//   du      - Input vector at quadrature points
//   q_data - Quadrature weights
//
// Output:
//   dv - Output vector at quadrature points
//
// *****************************************************************************
CEED_QFUNCTION_HELPER int Stiff_N(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, const CeedInt N, const CeedInt dim) {
  const CeedScalar(*du)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)        = in[1];
  CeedScalar(*dv)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    switch (dim) {
        case 2:
            CeedScalar wdetJ, dXdx[2][2];
            QdataUnpack_2D(Q, i, q_data, &wdetJ, dXdx);

            //compute qd = wdetJ . dXdx . dXdx^T
            const CeedScalar a00 = dXdx[0][0];
            const CeedScalar a01 = dXdx[0][1];
            const CeedScalar a10 = dXdx[1][0];
            const CeedScalar a11 = dXdx[1][1];

            // qd: 0 2
            //     2 1
            CeedScalar qd[3];
            qd[0] = wdetJ * (a00*a00 + a01*a01);
            qd[1] = wdetJ * (a10*a10 + a11*a11);
            qd[2] = wdetJ * (a00*a10 + a01*a11);
            
            for(CeedInt j = 0; j < N; j++){
                const CeedScalar du0 = du[0][j][i];
                const CeedScalar du1 = du[1][j][i];
                dv[0][j][i] = qd[0]*du0 + qd[2]*du1;
                dv[1][j][i] = qd[2]*du0 + qd[1]*du1;       
            }
            break;
        case 3:
            CeedScalar wdetJ, dXdx3[3][3];
            QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx3);

            const CeedScalar a00 = dXdx3[0][0];
            const CeedScalar a01 = dXdx3[0][1];
            const CeedScalar a02 = dXdx3[0][2];
            const CeedScalar a10 = dXdx3[1][0];
            const CeedScalar a11 = dXdx3[1][1];
            const CeedScalar a12 = dXdx3[1][2];
            const CeedScalar a20 = dXdx3[2][0];
            const CeedScalar a21 = dXdx3[2][1];
            const CeedScalar a22 = dXdx3[2][2];
            // Stored in Voigt convention
            // 0 5 4
            // 5 1 3
            // 4 3 2
            // CeedScalar qd[6];
            // qd[0] = wdetJ * (a00*a00 + a01*a01 + a02*a02);
            // qd[1] = wdetJ * (a10*a10 + a11*a11 + a12*a12);
            // qd[2] = wdetJ * (a20*a20 + a21*a21 + a22*a22);
            // qd[3] = wdetJ * (a10*a20 + a11*a21 + a12*a22);
            // qd[4] = wdetJ * (a00*a20 + a01*a21 + a02*a22);
            // qd[5] = wdetJ * (a00*a10 + a01*a11 + a02*a12);
            
            const CeedScalar dXdxdXdxT[3][3] = {
                {(a00*a00 + a01*a01 + a02*a02), (a00*a10 + a01*a11 + a02*a12), (a00*a20 + a01*a21 + a02*a22)},
                {(a00*a10 + a01*a11 + a02*a12), (a10*a10 + a11*a11 + a12*a12), (a10*a20 + a11*a21 + a12*a22)},
                {(a00*a20 + a01*a21 + a02*a22), (a10*a20 + a11*a21 + a12*a22), (a20*a20 + a21*a21 + a22*a22)}
            };

            for (CeedInt j = 0; j < N; j++){
                for (CeedInt k = 0; k < dim; k++){ 
                    dv[k][j][i] = wdetJ * (du[0][j][i] * dXdxdXdxT[0][k] + du[1][j][i] * dXdxdXdxT[1][k] + du[2][j][i] * dXdxdXdxT[2][k]);
                }
            }
            // Matrid matrix - This is a function that might do the same
           // MatMatN(dXdx3,dXdx3,N,CEED_NOTRANSPOSE, CEED_TRANSPOSE,dXdxdXdxT)
            break;
    }
  }
  return 0;
}

CEED_QFUNCTION(Stiff_1_2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Stiff_N(ctx, Q, in, out, 1, 2); }

CEED_QFUNCTION(Stiff_1_3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Stiff_N(ctx, Q, in, out, 1, 3); }

//CEED_QFUNCTION(Stiff_5)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Stiff_N(ctx, Q, in, out, 5, dim); }
//
//CEED_QFUNCTION(Stiff_7)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Stiff_N(ctx, Q, in, out, 7, dim); }
//
//CEED_QFUNCTION(Stiff_9)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Stiff_N(ctx, Q, in, out, 9, dim); }
//
//CEED_QFUNCTION(Stiff_22)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return Stiff_N(ctx, Q, in, out, 22, dim); }
//