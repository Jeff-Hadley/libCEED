// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/grid_anisotropy_tensor.h"

#include <petscdmplex.h>

#include "../navierstokes.h"



PetscErrorCode DataCompSetupApply(Ceed ceed, User user, CeedData ceed_data, CeedElemRestriction *elem_restr_mass,
                                                        CeedVector *mass_vector) {
  DM mass;
  CeedInt N;
  CeedOperator         op_mass, op_stiff;
  CeedQFunction        qf_mass, qf_stiff;
  CeedBasis            basis_mass; //both mass and stiffness can use the same basis
  CeedInt              q_data_size;
  MPI_Comm             comm = PetscObjectComm((PetscObject)user->dm);
  DMLabel              domain_label = NULL;
  PetscInt             label_value = 0, height = 0, dm_field = 0; 

  PetscFunctionBeginUser;

  // -- Create DM for Mass Matrix for Data compression
  PetscCall(DMClone(user->dm, &mass)); //Will need to create a DM clone for us to keep all the BC's that would get taken out. Only keeps topo info, get's rid of basis and restrictions for dof's 
  PetscCall(PetscObjectSetName((PetscObject)mass, "Data Comp Mass Matrix"));

  // -- Setup DM
    PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, 1, &N,
                                 mass));

  // -- Get Pre-requisite things
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, mass, domain_label, label_value, height, dm_field, elem_restr_mass));
  PetscCall(CreateBasisFromPlex(ceed, mass, domain_label, label_value, height, dm_field, &basis_mass));

  // -- Build Mass Operator - op_mass
  PetscCall(CreateMassQFunction(ceed, N, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", *elem_restr_mass, basis_mass, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", *elem_restr_mass, basis_mass, CEED_VECTOR_ACTIVE));

  // -- Build Stiffness Operator - op_stiff
  PetscCall(CreateStiffQFunction(ceed, N, dim, q_data_size, &qf_stiff));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_stiff, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "du", *elem_restr_mass, basis_mass, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "dv", *elem_restr_mass, basis_mass, CEED_VECTOR_ACTIVE));


  {  // -- Create MatCEEDs to then be assembled
    Mat mat_mass, mat_stiff;

    PetscCall(MatCeedCreate(mass, mass, op_mass, NULL, &mat_mass));
    PetscCall(MatCeedCreate(mass, mass, op_stiff, NULL, &mat_stiff));

    // this assembles the mat_xxxx from above
    Mat mat_assembled_mass, mat_assembled_stiff;
    
    PetscCall(MatCeedCreateMatCOO(mat_mass, &mat_assembled_mass));
    PetscCall(MatCeedAssembleCOO(mat_mass, mat_assembled_mass));
   
    PetscCall(MatCeedCreateMatCOO(mat_stiff, &mat_assembled_stiff));
    PetscCall(MatCeedAssembleCOO(mat_stiff, mat_assembled_stiff));
    
    // Call to destroy matrices, but want these to be used before getting destroyed I think.
    PetscCall(MatDestroy(&mat_mass));
    PetscCall(MatDestroy(&mat_stiff));
  }

  // -- Cleanup
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_mass));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stiff));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_stiff));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_stiff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

