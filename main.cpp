// =============================================================================
// test_poisson.cpp
//
// ── Test 1: Manufactured solution with Mixed Dirichlet/Neumann ──────────────
//   PDE:   Δu = 6              (constant source)
//   Exact: u(x) = x² + y² + z²
//   Outer BC: Dirichlet u = |x|² 
//   Inner BC:
//      if y <= 0: Dirichlet u = |x|² + 10.0f
//      if y >  0: Neumann   du/dn = h(x)
// =============================================================================

#include "src/tiny_bvh.h"
#include "src/WoStGeometryBackend.hpp"
#include "src/CubeOuterBoundary.hpp"
#include "src/WoStKernel.hpp"
#include "src/utils.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
#  include <omp.h>
#endif

using namespace wost;

int main(){
    std::string objfile = "./spot/spot_triangulated.obj";
    unsigned int numSamples = 100000; 
    float L = 1.0f;

    WoStGeometryBackend interior(objfile);
    CubeOuterBoundary exterior(-L, L);
    WoStKernel kernel(interior, exterior);
    
    #ifdef _OPENMP
    int num_threads = 32;
    printf("OpenMP threads: %d\n", num_threads);
    #endif

    {
        printf("\n=== Test 1: Manufactured Solution (Mixed Neumann/Dirichlet) ===\n");

        // ── 边界条件与源项设置 (针对精确解 u = x² + y² + z² 进行定量验证) ──

        // ✅ Bug 3a 修正：内部 Dirichlet 边界去掉不正确的偏移量
        auto g_inner = [](const BoundaryPoint& bp) -> float {
            return dot3(bp.position, bp.position) + 10; 
        };

        // 外部立方体边界 Dirichlet 条件
        auto g_outer = [](const BoundaryPoint& bp) -> float {
            return dot3(bp.position, bp.position);
        };

        // 混合边界划分：例如将 spot 模型的上半部分（y > 0）设为 Neumann 边界
        auto is_inner_neumann = [](const BoundaryPoint& bp) -> bool {
            return bp.position.y > 0.0f;
        };

        // ✅ Bug 3b 修正：根据精确解梯度与外法线方向，精确计算 Neumann 导数值
        // 💡 说明：bp.normal 是网格的向外法线（即指向求解域内部），
        // 求解域的真正外法线为 n_out = -bp.normal。
        // 因此 ∂u/∂n_out = ∇u · (-bp.normal) = -2 * p · bp.normal
        auto h_inner = [](const BoundaryPoint& bp) -> float {
            // return -2.0f * dot3(bp.position, bp.normal);
            return 0.0;
        };
        
        // Source Term f(x) = 6
        auto f = [](const vec3& x) -> float {
            (void)x;
            return 6.0f;
        };
        
        WoStParams params;
        params.numSamples = 256;
        params.maxSteps = 512;
        params.eps = 1e-4f;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int valid_count = 0;
        std::vector<PointSolution> pointcloud;
        pointcloud.reserve(numSamples);
        
        #pragma omp parallel
        {
            #ifdef _OPENMP
            FastRNG thread_rng(omp_get_thread_num() + static_cast<int>(time(nullptr)));
            #else
            FastRNG thread_rng;
            #endif
            
            std::vector<PointSolution> local_results;
            #ifdef _OPENMP
            local_results.reserve(numSamples / omp_get_num_threads() + 1);
            #endif
            
            #pragma omp for schedule(dynamic, 64) nowait
            for (uint32_t idx = 0; idx < numSamples; ++idx) {
                float x = thread_rng.randFloat() * 2.0f * L - L;
                float y = thread_rng.randFloat() * 2.0f * L - L;
                float z = thread_rng.randFloat() * 2.0f * L - L;    
                vec3 point = {x, y, z};
                        
                if (kernel.InDomain(point)) {
                    // 更新调用：传入 is_inner_neumann 和 h_inner
                    WalkResult result = kernel.SolvePoisson(
                        point, g_inner, is_inner_neumann, h_inner, g_outer, f, params
                    );
                    
                    PointSolution ps;
                    ps.pos = point;
                    ps.value = result.value;
                    ps.stdErr = result.stdErr;
                    ps.meanSteps = result.meanSteps;
                    ps.exact = dot3(point, point);
                    local_results.push_back(ps);
                }
            }
            
            #pragma omp critical
            {
                pointcloud.insert(pointcloud.end(), 
                                 std::make_move_iterator(local_results.begin()),
                                 std::make_move_iterator(local_results.end()));
                valid_count += local_results.size();
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        printf("Valid points: %d / %d\n", valid_count, numSamples);
        printf("Computation time: %.2f seconds\n", elapsed.count());
        printf("Average samples per point: %d\n", numSamples);
        
        if (WriteVTKPointCloud("test1_mixed_boundary_pointcloud.vtk", pointcloud, true)) {
            printf("✓ Point cloud written to test1_mixed_boundary_pointcloud.vtk\n");
        } else {
            printf("✗ Failed to write point cloud\n");
        }
        
        float max_error = 0.0f;
        float total_error = 0.0f;
        for (const auto& ps : pointcloud) {
            float error = std::abs(ps.value - ps.exact);
            max_error = std::max(max_error, error);
            total_error += error;
        }
        printf("Max absolute error: %.6f\n", max_error);
        printf("Mean absolute error: %.6f\n", total_error / pointcloud.size());
    }
    return 0;
}