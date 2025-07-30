/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2009 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Program based on step-37
 *
 * Authors: Katharina Kormann, Martin Kronbichler, Uppsala University,
 * 2009-2012, updated to MPI version with parallel vectors in 2016
 */


#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>


namespace multigrid
{
  using namespace dealii;



  template <int dim, typename number>
  class LaplaceOperator
    : public MatrixFreeOperators::
        Base<dim, LinearAlgebra::distributed::Vector<number>>
  {
  public:
    using value_type = number;

    LaplaceOperator() = default;

    virtual void
    compute_diagonal() override;

    void
    vmult(LinearAlgebra::distributed::Vector<number>       &dst,
          const LinearAlgebra::distributed::Vector<number> &src) const;

    void
    compute_matrix();

    const SparseMatrix<number> &
    get_matrix()
    {
      return matrix;
    }

  private:
    virtual void
    apply_add(
      LinearAlgebra::distributed::Vector<number>       &dst,
      const LinearAlgebra::distributed::Vector<number> &src) const override;

    void
    local_apply(const MatrixFree<dim, number>                    &data,
                LinearAlgebra::distributed::Vector<number>       &dst,
                const LinearAlgebra::distributed::Vector<number> &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_compute_diagonal(
      FEEvaluation<dim, -1, 0, 1, number> &integrator) const;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<number> matrix;
  };



  template <int dim, typename number>
  void
  LaplaceOperator<dim, number>::local_apply(
    const MatrixFree<dim, number>                    &data,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(EvaluationFlags::gradients);
        for (const unsigned int q : phi.quadrature_point_indices())
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
  }



  template <int dim, typename number>
  void
  LaplaceOperator<dim, number>::vmult(
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src, true);
    for (const unsigned int index : this->data->get_constrained_dofs())
      dst.local_element(index) = src.local_element(index);
  }



  template <int dim, typename number>
  void
  LaplaceOperator<dim, number>::apply_add(
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src, false);
  }



  template <int dim, typename number>
  void
  LaplaceOperator<dim, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);

    MatrixFreeTools::compute_diagonal(*this->data,
                                      inverse_diagonal,
                                      &LaplaceOperator::local_compute_diagonal,
                                      this);
    for (const unsigned int index : this->data->get_constrained_dofs())
      inverse_diagonal.local_element(index) = 1.;

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) =
          1. / inverse_diagonal.local_element(i);
      }
  }



  template <int dim, typename number>
  void
  LaplaceOperator<dim, number>::local_compute_diagonal(
    FEEvaluation<dim, -1, 0, 1, number> &phi) const
  {
    phi.evaluate(EvaluationFlags::gradients);
    for (const unsigned int q : phi.quadrature_point_indices())
      phi.submit_gradient(phi.get_gradient(q), q);
    phi.integrate(EvaluationFlags::gradients);
  }



  template <int dim, typename number>
  void
  LaplaceOperator<dim, number>::compute_matrix()
  {
    const DoFHandler<dim> &dof_handler = this->data->get_dof_handler();
    const unsigned int     level       = this->data->get_mg_level();
    if (level == numbers::invalid_unsigned_int)
      {
        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler,
                                        dsp,
                                        this->data->get_affine_constraints(),
                                        false);
        sparsity_pattern.copy_from(dsp);
      }
    else
      {
        DynamicSparsityPattern dsp(dof_handler.n_dofs(level),
                                   dof_handler.n_dofs(level));
        MGTools::make_sparsity_pattern(
          dof_handler, dsp, level, this->data->get_affine_constraints(), false);
        sparsity_pattern.copy_from(dsp);
      }
    matrix.reinit(sparsity_pattern);
    MatrixFreeTools::compute_matrix(*this->data,
                                    this->data->get_affine_constraints(),
                                    matrix,
                                    &LaplaceOperator::local_compute_diagonal,
                                    this);
    for (unsigned int i : this->data->get_constrained_dofs())
      matrix(i, i) = 1.;
  }



  template <typename number>
  class BlockJacobiPreconditioner
  {
  public:
    void
    initialize(
      const SparseMatrix<number>                              &sparse_matrix,
      const std::vector<std::vector<types::global_dof_index>> &dof_indices,
      const double                                             omega)
    {
      SparseMatrixTools::restrict_to_full_matrices(
        sparse_matrix,
        sparse_matrix.get_sparsity_pattern(),
        dof_indices,
        blocks);
      this->dof_indices = dof_indices;
      this->omega       = omega;

      for (auto &block : blocks)
        if (block.m() > 0 && block.n() > 0)
          {
            block.gauss_jordan();
          }
    }

    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      Vector<double> vector_src, vector_dst;

      for (unsigned int c = 0; c < blocks.size(); ++c)
        {
          const unsigned int dofs_per_block = dof_indices[c].size();
          if (dofs_per_block > 0)
            {
              vector_src.reinit(dofs_per_block);
              vector_dst.reinit(dofs_per_block);
              for (unsigned int i = 0; i < dofs_per_block; ++i)
                vector_src(i) = src(dof_indices[c][i]);
              blocks[c].vmult(vector_dst, vector_src);
              for (unsigned int i = 0; i < dofs_per_block; ++i)
                dst(dof_indices[c][i]) = omega * vector_dst(i);
            }
        }
    }

  private:
    std::vector<FullMatrix<number>>                   blocks;
    std::vector<std::vector<types::global_dof_index>> dof_indices;
    double                                            omega;
  };



  template <int dim>
  void
  run_test(const unsigned int fe_degree, const unsigned int n_refinements)
  {
    // 1. grid and dofs
    parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(n_refinements);
    GridTools::transform(
      [](const Point<dim> &p) {
        const double skew_factor = 3.;
        Point<dim> ptmp = p;
        ptmp[1] = 0.5 + 0.5 * (std::tanh(skew_factor * (2 * p[1] - 1.)) / std::tanh(skew_factor));
        return ptmp;
      },
      triangulation);


    const MappingQ1<dim> mapping;
    const FE_Q<dim>      fe(fe_degree);
    DoFHandler<dim>      dof_handler(triangulation);

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    AffineConstraints<double> constraints;
    constraints.clear();
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       DoFTools::extract_locally_relevant_dofs(dof_handler));
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();

    // 2. main matrix-free operator
    using VectorType = LinearAlgebra::distributed::Vector<double>;
    using MatrixType = LaplaceOperator<dim, double>;
    MatrixType system_matrix;
    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values);
      std::shared_ptr<MatrixFree<dim, double>> mf_storage(
        new MatrixFree<dim, double>());
      mf_storage->reinit(mapping,
                         dof_handler,
                         constraints,
                         QGauss<1>(fe.degree + 1),
                         additional_data);
      system_matrix.initialize(mf_storage);
    }

    // 3. solution vectors
    VectorType solution, rhs, tmp;

    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(rhs);
    system_matrix.initialize_dof_vector(tmp);

    // 4. right-hand side for a Poisson problem
    {
      FEEvaluation<dim, -1> phi(*system_matrix.get_matrix_free());
      for (unsigned int cell = 0;
           cell < system_matrix.get_matrix_free()->n_cell_batches();
           ++cell)
        {
          phi.reinit(cell);
          for (const unsigned int q : phi.quadrature_point_indices())
            phi.submit_value(make_vectorized_array<double>(1.0), q);
          phi.integrate(EvaluationFlags::values);
          phi.distribute_local_to_global(rhs);
        }
      rhs.compress(VectorOperation::add);
    }

    system_matrix.vmult(solution, rhs);
    system_matrix.compute_matrix();
    system_matrix.get_matrix().vmult(tmp, rhs);
    tmp -= solution;
    std::cout << "Error mat-vec: " << tmp.l2_norm() << std::endl;
    solution = 0;

    // 5. multigrid operators and its boundary conditions
    const unsigned int nlevels = triangulation.n_global_levels();

    MGConstrainedDoFs                  mg_constrained_dofs;
    const std::set<types::boundary_id> dirichlet_boundary_ids = {0};
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                       dirichlet_boundary_ids);

    MGLevelObject<MatrixType> mg_matrices(0, nlevels - 1);

    for (unsigned int level = 0; level < nlevels; ++level)
      {
        AffineConstraints<double> level_constraints(
          dof_handler.locally_owned_mg_dofs(level),
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));
        for (const types::global_dof_index dof_index :
             mg_constrained_dofs.get_boundary_indices(level))
          level_constraints.constrain_dof_to_zero(dof_index);
        level_constraints.close();

        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, double>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_gradients | update_JxW_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, double>> mf_storage_level =
          std::make_shared<MatrixFree<dim, double>>();
        mf_storage_level->reinit(mapping,
                                 dof_handler,
                                 level_constraints,
                                 QGauss<1>(fe.degree + 1),
                                 additional_data);

        mg_matrices[level].initialize(mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
        mg_matrices[level].compute_matrix();
      }

    // 6. multigrid level smoother
    using SmootherType =
      PreconditionChebyshev<MatrixType,
                            VectorType,
                            BlockJacobiPreconditioner<double>>;
    mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range     = 12.;
            smoother_data[level].degree              = 3;
            smoother_data[level].eig_cg_n_iterations = 10;
          }
        else
          {
            // set up parameters for direct solver
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree          = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }
        mg_matrices[level].compute_diagonal();
        // smoother_data[level].preconditioner =
        //   mg_matrices[level].get_matrix_diagonal_inverse();

        std::vector<std::vector<types::global_dof_index>> dof_indices(
          triangulation.n_cells(level));
        std::vector<unsigned char> dof_touched(dof_handler.n_dofs(level), 0);
        std::vector<types::global_dof_index> cell_dof_indices(fe.dofs_per_cell);
        for (const auto &cell : dof_handler.cell_iterators_on_level(level))
          {
            cell->get_mg_dof_indices(cell_dof_indices);

            // put all dof indices to zero-th child (i.e., get 4 cells into
            // one)
            auto &dof = dof_indices[cell->level() > 0 ?
                                      cell->parent()->child(0)->index() :
                                      cell->index()];
            for (types::global_dof_index i : cell_dof_indices)
              if (dof_touched[i] == 0)
                {
                  dof.push_back(i);
                  dof_touched[i] = 1;
                }
            std::sort(dof.begin(), dof.end());
            dof.erase(std::unique(dof.begin(), dof.end()), dof.end());
          }
        smoother_data[level].preconditioner =
          std::make_shared<BlockJacobiPreconditioner<double>>();
        smoother_data[level].preconditioner->initialize(
          mg_matrices[level].get_matrix(), dof_indices, 1.0);
      }
    mg_smoother.initialize(mg_matrices, smoother_data);

    // 7. coarse grid operator (= solver)
    MGCoarseGridApplySmoother<VectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    // 8. multigrid transfer operator
    MGTransferMatrixFree<dim, double> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    mg::Matrix<VectorType> mg_matrix(mg_matrices);
    Multigrid<VectorType>  mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);

    PreconditionMG<dim, VectorType, MGTransferMatrixFree<dim, double>>
      preconditioner(dof_handler, mg, mg_transfer);

    // 9. solve linear system
    SolverControl        solver_control(100, 1e-9 * rhs.l2_norm());
    SolverCG<VectorType> cg(solver_control);

    Timer time;
    for (unsigned int i = 0; i < 2; ++i)
      {
        solution = 0;
        time.restart();
        cg.solve(system_matrix, solution, rhs, preconditioner);

        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Time solve size " << dof_handler.n_dofs() << " ("
                    << solver_control.last_step() << " iterations)"
                    << (solver_control.last_step() < 10 ? "  " : " ")
                    << "(CPU/wall) " << time.wall_time() << "s\n";
      }

    // 10. check smoothing properties of multigrid method and write them to
    // output file
    std::array<VectorType, 11> aux_vectors;
    for (auto &vec : aux_vectors)
      system_matrix.initialize_dof_vector(vec);

    for (unsigned int i = 0; i < aux_vectors[0].locally_owned_size(); ++i)
      aux_vectors[0].local_element(i) = (double)rand() / RAND_MAX;
    system_matrix.vmult(aux_vectors[1], aux_vectors[0]);
    aux_vectors[1] *= -1.0;

    // multigrid operator
    preconditioner.vmult(aux_vectors[2], aux_vectors[1]);
    aux_vectors[2] += aux_vectors[0];

    // chosen multigrid smoother (chebyshev, 5 iterations)
    mg_smoother.apply(nlevels - 1, aux_vectors[3], aux_vectors[1]);
    aux_vectors[3] += aux_vectors[0];

    // relaxation smoother with point Jacobi (no Chebyshev)
    PreconditionJacobi<MatrixType> jacobi_smoother;
    system_matrix.compute_diagonal();
    jacobi_smoother.initialize(
      system_matrix,
      typename PreconditionJacobi<MatrixType>::AdditionalData(0.5));
    jacobi_smoother.vmult(aux_vectors[4], aux_vectors[1]);
    aux_vectors[4] += aux_vectors[0];

    jacobi_smoother.vmult(aux_vectors[5], aux_vectors[1]);
    for (unsigned int i = 0; i < 4; ++i)
      {
        system_matrix.vmult(aux_vectors[7], aux_vectors[5]);
        aux_vectors[7] -= aux_vectors[1];
        jacobi_smoother.vmult(aux_vectors[7], aux_vectors[7]);
        aux_vectors[5] -= aux_vectors[7];
      }
    aux_vectors[5] += aux_vectors[0];

    jacobi_smoother.vmult(aux_vectors[6], aux_vectors[1]);
    for (unsigned int i = 0; i < 9; ++i)
      {
        system_matrix.vmult(aux_vectors[7], aux_vectors[6]);
        aux_vectors[7] -= aux_vectors[1];
        jacobi_smoother.vmult(aux_vectors[7], aux_vectors[7]);
        aux_vectors[6] -= aux_vectors[7];
      }
    aux_vectors[6] += aux_vectors[0];

    BlockJacobiPreconditioner<double>                 block_jacobi;
    std::vector<std::vector<types::global_dof_index>> dof_indices(
      triangulation.n_active_cells());
    std::vector<unsigned char>           dof_touched(dof_handler.n_dofs(), 0);
    std::vector<types::global_dof_index> cell_dof_indices(fe.dofs_per_cell);
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell->get_dof_indices(cell_dof_indices);
        for (types::global_dof_index i : cell_dof_indices)
          if (dof_touched[i] == 0)
            {
              dof_indices[cell->active_cell_index()].push_back(i);
              dof_touched[i] = 1;
            }
      }
    block_jacobi.initialize(system_matrix.get_matrix(), dof_indices, 0.8);

    block_jacobi.vmult(aux_vectors[8], aux_vectors[1]);
    aux_vectors[8] += aux_vectors[0];

    block_jacobi.vmult(aux_vectors[9], aux_vectors[1]);
    for (unsigned int i = 0; i < 4; ++i)
      {
        system_matrix.vmult(aux_vectors[7], aux_vectors[9]);
        aux_vectors[7] -= aux_vectors[1];
        block_jacobi.vmult(aux_vectors[7], aux_vectors[7]);
        aux_vectors[9] -= aux_vectors[7];
      }
    aux_vectors[9] += aux_vectors[0];

    block_jacobi.vmult(aux_vectors[10], aux_vectors[1]);
    for (unsigned int i = 0; i < 9; ++i)
      {
        system_matrix.vmult(aux_vectors[7], aux_vectors[10]);
        aux_vectors[7] -= aux_vectors[1];
        block_jacobi.vmult(aux_vectors[7], aux_vectors[7]);
        aux_vectors[10] -= aux_vectors[7];
      }
    aux_vectors[10] += aux_vectors[0];

    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    flags.compression_level        = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);

    solution.update_ghost_values();
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(aux_vectors[0], "smooth_initial");
    data_out.add_data_vector(aux_vectors[2], "smooth_mg");
    data_out.add_data_vector(aux_vectors[3], "smooth_chebyshev_5");
    data_out.add_data_vector(aux_vectors[4], "smooth_jacobi_1");
    data_out.add_data_vector(aux_vectors[5], "smooth_jacobi_5");
    data_out.add_data_vector(aux_vectors[6], "smooth_jacobi_10");
    data_out.add_data_vector(aux_vectors[8], "smooth_block_jacobi_1");
    data_out.add_data_vector(aux_vectors[9], "smooth_block_jacobi_5");
    data_out.add_data_vector(aux_vectors[10], "smooth_block_jacobi_10");
    data_out.build_patches(mapping, fe.degree);

    data_out.write_vtu_in_parallel("solution-" +
                                     std::to_string(dof_handler.n_dofs()) +
                                     ".vtu",
                                   MPI_COMM_WORLD);
  }
} // namespace multigrid



int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      for (unsigned int i = 3; i < 8; ++i)
        multigrid::run_test<2>(3, i);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
