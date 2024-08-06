// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//#include "../qfunctions/grid_anisotropy_tensor.h"

#include <ceed.h>
#include <petscdmplex.h>

#include "../navierstokes.h"
#include <petscksp.h>
#include <petscvec.h>
#include <petscviewer.h>

// Sets up the mass and stiffness matrices of a poisson problem associated to the mesh the actual problem is being solved on.
// Assembles into global mass and stiff matrices. 

PetscErrorCode DataCompSetupApply(Ceed ceed, User user, CeedData ceed_data, CeedInt dim) {
  DM mass;
  CeedInt N = 1;
  CeedOperator         op_mass, op_stiff;
  CeedElemRestriction  elem_restr_mass;
  CeedQFunction        qf_mass, qf_stiff;
  CeedBasis            basis_mass; //both mass and stiffness can use the same basis
  CeedInt              q_data_size;
  // MPI_Comm             comm = PetscObjectComm((PetscObject)user->dm);
  DMLabel              domain_label = NULL;
  PetscInt             label_value = 0, height = 0, dm_field = 0;

  PetscFunctionBeginUser;


  PetscCall(PetscNew(&user->data_comp));

  // -- Create DM for Mass Matrix for Data compression
  PetscCall(DMClone(user->dm, &mass)); //Will need to create a DM clone for us to keep all the BC's nodes that would toherwise get taken out (KEEP all nodes, even if Dirichlet). Only keeps topo info, get's rid of basis and restrictions for dof's. 
  PetscCall(PetscObjectSetName((PetscObject)mass, "Data Comp Mass Matrix"));

  // -- Setup DM
    PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, user->app_ctx->degree, 1, user->app_ctx->q_extra, 1, &N,
                                 mass));

  // -- Get Pre-requisite things
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, mass, domain_label, label_value, height, dm_field, &elem_restr_mass));
  PetscCall(CreateBasisFromPlex(ceed, mass, domain_label, label_value, height, dm_field, &basis_mass));

  // -- Build Mass Operator - op_mass
  PetscCall(CreateMassQFunction(ceed, N, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", elem_restr_mass, basis_mass, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", elem_restr_mass, basis_mass, CEED_VECTOR_ACTIVE));

  // -- Build Stiffness Operator - op_stiff
  PetscCall(CreateStiffQFunction(ceed, N, dim, q_data_size, &qf_stiff));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_stiff, NULL, NULL, &op_stiff));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stiff, "du", elem_restr_mass, basis_mass, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stiff, "qdata", ceed_data->elem_restr_qd_i, 
  CEED_BASIS_NONE, ceed_data->q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stiff, "dv", elem_restr_mass, basis_mass, CEED_VECTOR_ACTIVE));


  {  // -- Create MatCEEDs to then be assembled
    Mat mat_mass, mat_stiff;

    PetscCall(MatCeedCreate(mass, mass, op_mass, NULL, &mat_mass));
    PetscCall(MatCeedCreate(mass, mass, op_stiff, NULL, &mat_stiff));

    // This assembles the mat_xxxx from above into mat_assembled_xxxx
    
    //Mat mat_assembled_mass, mat_assembled_stiff;
    
    PetscCall(MatCeedCreateMatCOO(mat_mass, &user->data_comp->assembled_mass));
    PetscCall(MatCeedAssembleCOO(mat_mass, user->data_comp->assembled_mass));
   
    PetscCall(MatCeedCreateMatCOO(mat_stiff, &user->data_comp->assembled_stiff));
    PetscCall(MatCeedAssembleCOO(mat_stiff, user->data_comp->assembled_stiff));
    
    // Call to destroy local matrices
    PetscCall(MatDestroy(&mat_mass));
    PetscCall(MatDestroy(&mat_stiff));
  }
  // -- Cleanup
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_mass));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stiff));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_stiff));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode DataCompExtractProlongation(User user){
  
  printf("Calling Hypre Functions \n");
  
  PC pcHypre;
  Vec x, b;
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(user->comm, &user->data_comp->kspHypre));
  PetscCall(KSPSetType(user->data_comp->kspHypre, KSPRICHARDSON));
  PetscCall(KSPGetPC(user->data_comp->kspHypre, &pcHypre));
  PetscCall(PCSetType(pcHypre, PCHYPRE));
  PetscCall(PCHYPRESetType(pcHypre, "boomeramg"));
  PetscCall(PCSetOptionsPrefix(pcHypre, "data_comp_")); //yaml file will have options for data compression under 'data_comp:'
  PetscCall(PCSetFromOptions(pcHypre));
  PetscCall(PCSetOperators(pcHypre, user->data_comp->assembled_stiff, user->data_comp->assembled_stiff));
  PetscCall(PCSetUp(pcHypre));
  
  PetscCall(MatCreateVecs(user->data_comp->assembled_stiff, &x, &b));
  PetscCall(PCApply(pcHypre, x, b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(PCView(pcHypre, NULL));
  PetscCall(PCGetInterpolations(pcHypre, &user->data_comp->num_levels, &user->data_comp->ProlongationOps));
  
  printf("Num levels: %d\n", user->data_comp->num_levels);
  printf("Finished calling the Hypre functions \n");

  printf("Calling the MatView functions \n");
  PetscViewer viewer;
  PetscCall(PetscViewerCreate(user->comm, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_BINARY_MATLAB));
  #define filenametemplate "Prolongation_%d.dat"
  
  char file_name[20];
  for(int i = 0; i < (int)user->data_comp->num_levels-1; i++){
    sprintf(file_name, filenametemplate, i+1);
    printf("%s \n", file_name);
    PetscCall(PetscViewerBinaryOpen(user->comm, file_name, FILE_MODE_WRITE, &viewer));
    PetscCall(MatView(user->data_comp->ProlongationOps[i], viewer));
  }
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatViewFromOptions(user->data_comp->assembled_mass, NULL, "-mat_view_ass_mass"));
  PetscCall(MatViewFromOptions(user->data_comp->assembled_stiff, NULL, "-mat_view_ass_stiff"));
  printf("Finished calling the MatView functions\n");
  
  PetscFunctionReturn(PETSC_SUCCESS); 
}


PetscErrorCode DataCompDestroy(DataCompression data_comp){
  //PetscErrorCode NodalProjectionDataDestroy(NodalProjectionData context) {
  PetscFunctionBeginUser;
  if (data_comp == NULL) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatDestroy(&data_comp->assembled_mass));
  PetscCall(MatDestroy(&data_comp->assembled_stiff));
  PetscCall(KSPDestroy(&data_comp->kspHypre));
  for(CeedInt i = 0; i < (CeedInt)data_comp->num_levels-1; i++){
     PetscCall(MatDestroy(&data_comp->ProlongationOps[i])); 
  } 
  PetscCall(PetscFree(data_comp->ProlongationOps));
  PetscCall(PetscFree(data_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
