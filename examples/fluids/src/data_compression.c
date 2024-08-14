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
#include "mat-ceed.h"
#include "mpi.h"
#include "petsc-ceed.h"
#include "petscbt.h"
#include "petscerror.h"
#include "petscis.h"
#include "petscmat.h"
#include "petscsys.h"
#include "petscsystypes.h"
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


//PetscErrorCode DataCompMultilevelDecomp(DataCompression data){
//  PetscFunctionBeginUser;
//  
//  PetscFunctionReturn(PETSC_SUCCESS); 
//}

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
  PetscCall(PCGetCFMarkers(pcHypre, &user->data_comp->n_per_level, &user->data_comp->CFMarkers));
  PetscFunctionReturn(PETSC_SUCCESS); 
}
 
PetscErrorCode DataCompExportMats(User user) { 
  PetscFunctionBeginUser;
  printf("Num levels: %d\n", user->data_comp->num_levels);
  printf("Finished calling the Hypre functions \n");

  printf("Calling the Viewer functions \n");
  PetscViewer viewer1;
  PetscCall(PetscViewerCreate(user->comm, &viewer1));
  PetscCall(PetscViewerPushFormat(viewer1, PETSC_VIEWER_BINARY_MATLAB));
  
  PetscViewer viewer2;
  PetscCall(PetscViewerCreate(user->comm, &viewer2));
  PetscCall(PetscViewerPushFormat(viewer2, PETSC_VIEWER_ASCII_MATLAB));
  
  #define Prolongationtemplate "Prolongation_%d.dat"
  #define CFMarkertemplate "CFMarkers_%d.dat"
  #define Pfloortemplate "Pfloor_%d.dat"
  #define LocToGlobtemplate "LtoG_%d.dat"
  
  char file_name[20];
  for(int i = 0; i < (int)user->data_comp->num_levels-1; i++){
    sprintf(file_name, Prolongationtemplate, i+1);
    printf("%s \n", file_name);
    PetscCall(PetscViewerBinaryOpen(user->comm, file_name, FILE_MODE_WRITE, &viewer1));
    PetscCall(MatView(user->data_comp->ProlongationOps[i], viewer1));

    sprintf(file_name, Pfloortemplate, i+1);
    printf("%s \n", file_name);
    PetscCall(PetscViewerBinaryOpen(user->comm, file_name, FILE_MODE_WRITE, &viewer1));
    PetscCall(MatView(user->data_comp->Pfloor[i], viewer1));
  }

  for(int i = 0; i < (int)user->data_comp->num_levels-1; i++){
    sprintf(file_name, CFMarkertemplate, i+1);
    printf("%s \n", file_name);
    PetscCall(PetscViewerASCIIOpen(user->comm, file_name, &viewer2));
    //printf("nodes on level %d: %d \n", i, user->data_comp->n_per_level[i]);
    PetscCall(PetscBTView(user->data_comp->n_per_level[i],user->data_comp->CFMarkers[i], viewer2));
  }

 // for(int i = 0; i < (int)user->data_comp->num_levels; i++){
 //   sprintf(file_name, LocToGlobtemplate, i+1);
 //   printf("%s \n", file_name);
 //   PetscCall(PetscViewerBinaryOpen(user->comm, file_name, FILE_MODE_WRITE, &viewer1));
 //   PetscCall(VecView(user->data_comp->LocToGlob[i], viewer1));
 // }

  PetscCall(PetscViewerDestroy(&viewer1));
  PetscCall(PetscViewerDestroy(&viewer2));


  PetscCall(MatViewFromOptions(user->data_comp->assembled_mass, NULL, "-mat_view_ass_mass"));
  PetscCall(MatViewFromOptions(user->data_comp->assembled_stiff, NULL, "-mat_view_ass_stiff"));
  printf("Finished calling the Viewer functions\n");
  
  PetscFunctionReturn(PETSC_SUCCESS); 
}

//PetscErrorCode DataCompProlongFloor(PetscBT *CFmarkers[], , PetscInt *n_per_level[], Mat *Pfloor[]){
PetscErrorCode DataCompProlongFloor(MPI_Comm comm, DataCompression data_comp){
  PetscInt nfine, ncoarse, nnz;
  Mat *Pfloortemp;
  PetscScalar val = 1;
  PetscFunctionBeginUser;

  PetscCall(PetscMalloc1(data_comp->num_levels-1, &Pfloortemp));
  for(PetscInt i = 0; i < data_comp->num_levels-1; i++){
    nnz = 0;
    PetscCall(MatGetSize(data_comp->ProlongationOps[i], &nfine, &ncoarse));
    PetscCall(MatDuplicate(data_comp->ProlongationOps[i], MAT_DO_NOT_COPY_VALUES, &Pfloortemp[i])); 
    //PetscCall(MatCreate(comm, &Pfloortemp[i]));
    //PetscCall(MatSetSizes(Pfloortemp[i], nfine, ncoarse, ));
    printf("+++ Level %d +++\n nfine: %d\n ncoarse: %d\n", i, nfine, ncoarse);
    //PetscCall(MatView(Pfloortemp[i], PETSC_VIEWER_STDOUT_WORLD));
    //PetscCall(PetscBTView(nfine, data_comp->CFMarkers[i], PETSC_VIEWER_STDOUT_WORLD));
    for(PetscInt j = 0; j < nfine; j++){
      if(PetscBTLookup(data_comp->CFMarkers[i],j)){
        //printf("Inserting Value into Pfloor\n");
        PetscCall(MatSetValues(Pfloortemp[i], 1, &j, 1, &nnz, &val, INSERT_VALUES));
        nnz++;
      }
    }
    MatAssemblyBegin(Pfloortemp[i], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Pfloortemp[i], MAT_FINAL_ASSEMBLY);
  }
  data_comp->Pfloor = Pfloortemp;
  PetscFunctionReturn(PETSC_SUCCESS); 
}

PetscErrorCode DataCompGetIndexSets(MPI_Comm comm, DataCompression data_comp){
  //Vec *LtoGtemp;
  IS  *LtoGtempIS;
  IS  *OnlyFineGlobtemp;
  IS *OnlyFineOnLeveltemp;
  IS  *CtoFtemp;
  PetscInt nfine, ncoarse, end, nnz; //, vec_size
  PetscInt *temp_idx_c;
  PetscInt *temp_idx_f;
  IS tempIS;
  PetscFunctionBeginUser;
  //PetscCall(PetscMalloc1(data_comp->num_levels, &LtoGtemp));   
  PetscCall(PetscMalloc1(data_comp->num_levels, &LtoGtempIS));
  PetscCall(PetscMalloc1(data_comp->num_levels-1, &OnlyFineGlobtemp));  
  PetscCall(PetscMalloc1(data_comp->num_levels-1, &OnlyFineOnLeveltemp)); 
  PetscCall(PetscMalloc1(data_comp->num_levels-1, &CtoFtemp)); 

  end = data_comp->num_levels-1;
  printf("end index value: %d\n", end);
  for(PetscInt i = 0; i < data_comp->num_levels; i++){
  
    if(i == 0){ //Need a finest index set, 0 to nfinest-1 index numbers
      PetscCall(MatGetSize(data_comp->Pfloor[end-1], &nfine, &ncoarse));
      printf("nfinest = %d\n", nfine);
      PetscCall(PetscMalloc1(nfine, &temp_idx_f));
      for(PetscInt i = 0; i < nfine; i++){
        temp_idx_f[i] = i;
      }
      PetscCall(ISCreateGeneral(comm, nfine, temp_idx_f, PETSC_COPY_VALUES, &LtoGtempIS[end-i]));
      PetscCall(PetscFree(temp_idx_f));
      
    }
    else{
      PetscCall(MatGetSize(data_comp->Pfloor[end-i], &nfine, &ncoarse));
      PetscCall(PetscMalloc1(ncoarse, &temp_idx_c));
      PetscCall(PetscMalloc1(nfine, &temp_idx_f));
      nnz = 0;
      for(PetscInt j = 0; j < nfine; j++){
        temp_idx_f[j] = j;
        if(PetscBTLookup(data_comp->CFMarkers[end-i],j)){
          temp_idx_c[nnz] = j;
          nnz++;
        }
      }
      PetscCall(ISCreateGeneral(comm, ncoarse, temp_idx_c, PETSC_COPY_VALUES, &CtoFtemp[end-i])); //CoarsetoFine on level relative
      PetscCall(ISCreateGeneral(comm, nfine, temp_idx_f, PETSC_COPY_VALUES, &tempIS));
      PetscCall(ISDifference(tempIS, CtoFtemp[end-i], &OnlyFineOnLeveltemp[end-i])); //Fine only nodes, on level index number
      PetscCall(PetscFree(temp_idx_c));
      PetscCall(PetscFree(temp_idx_f));
      PetscCall(ISDestroy(&tempIS));

      // Grabbing the 'finest' index numbers corresponding to current on level coarse nodes and storing. 
      PetscCall(ISCreateSubIS(LtoGtempIS[end-i+1], CtoFtemp[end-i], &LtoGtempIS[end-i])); //finest index numbers of all on level nodes
      PetscCall(ISDifference(LtoGtempIS[end-i+1], LtoGtempIS[end-i], &OnlyFineGlobtemp[end-i])); //finest index numbers of 'only fine nodes' on current level. 

    }
  }

  data_comp->LocToGlobIS = LtoGtempIS;
  data_comp->CoarsetoFineIS = CtoFtemp;
  data_comp->OnlyFineGlobIS = OnlyFineGlobtemp;
  data_comp->OnlyFineOnLevelIS = OnlyFineOnLeveltemp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCompDecompose(MPI_Comm comm, DataCompression data_comp, Vec x){
  Mat Mass_f, Mass_c, Msub, Psub;
  Vec x_c_i, x_f_i, x_int_i, Malpha, f, z;
  IS tempIS;
  PetscInt *temp_idx;
  PetscInt nfine, ncoarse;
  KSP ksp;

  PetscFunctionBeginUser;
  
  //Grab Finest level mass matrix to use
  PetscCall(MatConvert(data_comp->assembled_mass, MATSAME, MAT_INITIAL_MATRIX, &Mass_f));

  // Loop over levels, creating projections and determining multilevel coefficients
  for(CeedInt i = data_comp->num_levels-1; i > -1; i--){
        
    // Get size of fine and coarse domains for current level
    PetscCall(MatGetSize(data_comp->ProlongationOps[i-1], &nfine, &ncoarse)); 
    
    //Create Coarse level Mass Matrix from fine level and Prolongation Ops
    PetscCall(MatPtAP(Mass_f, data_comp->ProlongationOps[i-1], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mass_c));

     //Grab Coarse node values sub vector
    PetscCall(VecGetSubVector(x, data_comp->LocToGlobIS[i-1], &x_c_i));

    { //Get Psub - rows of P corresponding to 'only fine' nodes, and all columns.
     PetscCall(PetscMalloc1(ncoarse, &temp_idx));
     for(PetscInt i = 0; i < ncoarse; i++){
       temp_idx[i] = i;
     }
     PetscCall(ISCreateGeneral(comm, ncoarse, temp_idx, PETSC_COPY_VALUES, &tempIS));
     PetscCall(PetscFree(temp_idx)); //tempIS is 0:ncoarse-1 so that we grab all columns.
     PetscCall(MatCreateSubMatrix(data_comp->ProlongationOps[i-1], data_comp->OnlyFineOnLevelIS[i-1], tempIS, MAT_INITIAL_MATRIX, &Psub));
     PetscCall(ISDestroy(&tempIS));
    }

    // Calculate delta_u values on 'fine only' nodes
    PetscCall(VecCreate(comm, &x_int_i));
    PetscCall(MatMult(Psub,x_c_i,x_int_i)); //x_int_i has the interpolated values on 'fine only' nodes
    PetscCall(VecGetSubVector(x, data_comp->OnlyFineGlobIS[i-1], &x_f_i)); //'fine only' node values currently on x
    PetscCall(VecAXPY(x_f_i, -1, x_int_i)); //x_f_i now contains the delta_u on fine only nodes
    PetscCall(VecDestroy(&x_int_i));
    
    {//Get Msub - all rows of Mass_f, and 'only fine' node columns
      PetscCall(PetscMalloc1(nfine, &temp_idx));
      for(PetscInt i = 0; i < nfine; i++){
        temp_idx[i] = i;
      }
      PetscCall(ISCreateGeneral(comm, nfine, temp_idx, PETSC_COPY_VALUES, &tempIS));
      PetscCall(PetscFree(temp_idx)); //tempIS is 0:nfine-1 so that we grab all rows of mass matrix for respective fine only nodes.
    PetscCall(MatCreateSubMatrix(Mass_f, tempIS, data_comp->OnlyFineOnLevelIS[i-1], MAT_INITIAL_MATRIX, &Msub));
    }

    // calc M_f * alpha using Msub matrix to save on flops
    PetscCall(MatMult(Msub, x_f_i, Malpha)); //now have M*alpha for forcing vector
    PetscCall(MatDestroy(&Msub));
    
    //Store delta_u values on original x vector
    PetscCall(VecRestoreSubVector(x, data_comp->OnlyFineGlobIS[i-1], &x_f_i));
    
    //Calc f = P' * M_f * alpha
    PetscCall(MatMultTranspose(data_comp->ProlongationOps[i-1], Malpha, f));
    
    // Destroy Vecs no longer needed
    PetscCall(VecDestroy(&x_f_i));
    PetscCall(VecDestroy(&Malpha));

    //Solve Mass_c z = f for z
    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetOperators(ksp, Mass_c, Mass_c));
    PetscCall(KSPSolve(ksp, f, z)); // Calculate correction factor z
    PetscCall(VecDestroy(&f));
    PetscCall(KSPDestroy(&ksp));

    //Adding correction factor to original solution x
    PetscCall(VecAXPY(x_c_i, 1, z)); //x(coarse nodes) + z | adding the correction factors
    PetscCall(VecRestoreSubVector(x, data_comp->LocToGlobIS[i-1], &x_c_i));
    
    // Destroy Vecs no longer needed
    PetscCall(VecDestroy(&x_c_i));
    PetscCall(VecDestroy(&z));

    //Swap coarse to fine Mass for next level
    PetscCall(MatDestroy(&Mass_f));
    Mass_f = Mass_c;
    Mass_c = NULL;
    
  }  
  // x now contains multilevel coefficients
  PetscCall(MatDestroy(&Mass_f));
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
     PetscCall(MatDestroy(&data_comp->Pfloor[i]));
     PetscCall(PetscBTDestroy(&data_comp->CFMarkers[i]));
     PetscCall(ISDestroy(&data_comp->OnlyFineGlobIS[i]));
     PetscCall(ISDestroy(&data_comp->OnlyFineOnLevelIS[i]));
     PetscCall(ISDestroy(&data_comp->CoarsetoFineIS[i]));
  } 
  for(CeedInt i = 0; i < (CeedInt)data_comp->num_levels; i++){
    PetscCall(ISDestroy(&data_comp->LocToGlobIS[i]));
  }
  PetscCall(PetscFree(data_comp->ProlongationOps));
  PetscCall(PetscFree(data_comp->Pfloor));
  PetscCall(PetscFree(data_comp->CFMarkers));
  PetscCall(PetscFree(data_comp->LocToGlobIS));
  PetscCall(PetscFree(data_comp->OnlyFineGlobIS));
  PetscCall(PetscFree(data_comp->OnlyFineOnLevelIS));
  PetscCall(PetscFree(data_comp->CoarsetoFineIS));
  PetscCall(PetscFree(data_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
