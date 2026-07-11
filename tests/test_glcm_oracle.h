#pragma once

#include <gtest/gtest.h>
#include <typeinfo>
#include <vector>

#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/feature_settings.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"

using namespace Nyxus;

// Regression guard for the GLCM "offset = 0 default" defect (found 2026-06).
//
// The production path (CLI and Python featurize) builds GLCM settings via
// Environment::compile_feature_settings(). That function used to initialise only the COMMON
// settings and left the GLCM-specific ones (GLCM_OFFSET / GLCM_GREYDEPTH / GLCM_NUMANG)
// zero-initialised. With GLCM_OFFSET == 0 the co-occurrence shift is dx = dy = 0, so every
// pixel co-occurs with itself -> a purely diagonal matrix -> CONTRAST = 0, CORRELATION = 1 for
// any image. The existing test_glcm.h cases never caught this because they hard-code
// GLCM_OFFSET = 1 (and run on a fully-masked phantom). This test exercises the real default.
inline void test_glcm_bug_offset_default_is_one()
{
    Environment e;
    e.set_coarse_gray_depth(64);
    e.compile_feature_settings();

    const Fsettings& s = e.get_feature_settings(typeid(GLCMFeature));

    // co-occurrence distance must default to 1 (IBSI delta = 1), NOT 0
    ASSERT_EQ(STNGS_GLCM_OFFSET(s), 1);
    // the GLCM-specific grey depth and angle count must be initialised, not left at 0
    ASSERT_GT(STNGS_GLCM_GREYDEPTH(s), 0);
    ASSERT_GT(STNGS_GLCM_NUMANG(s), 0);
}

// ---------------------------------------------------------------------------
// PyRadiomics oracle for the angle-averaged GLCM _AVE family (C++ path).
//
// The C++ twin of tests/python/test_glcm_oracle.py::test_glcm_ave_pyradiomics_oracle: it runs
// GLCMFeature directly (no Python layer) on the same dense 8x8 fixture and pins the same 20
// _AVE features to the same PyRadiomics reference values, so ./runAllTests independently verifies
// against the third-party oracle.
//
// Fixture: img(x,y) = ((row + 2*col) % 8) + 1 over a solid 8x8 ROI (row=y-1, col=x-1); every grey
// level 1..8 occurs, so PyRadiomics does not re-index levels. Settings come from the production
// Environment path (coarse_gray_depth=8, ibsi=True) -> symmetric matrix + identity binning, which
// reproduces PyRadiomics (symmetricalGLCM=True, binWidth=1, distances=[1], force2D, weightingNorm
// =None); verified value-for-value against the nyxus wheel and the radiomics/pyradiomics Docker.
//
// GLCM_ENTROPY_AVE matches only because f_entropy normalizes the co-occurrence counts by sum_p
// (otherwise it sums raw counts and returns a large-negative, meaningless value).

static const NyxusPixel glcm_dense_intensity[] = {
    {1, 1, 1}, {2, 1, 3}, {3, 1, 5}, {4, 1, 7}, {5, 1, 1}, {6, 1, 3}, {7, 1, 5}, {8, 1, 7},
    {1, 2, 2}, {2, 2, 4}, {3, 2, 6}, {4, 2, 8}, {5, 2, 2}, {6, 2, 4}, {7, 2, 6}, {8, 2, 8},
    {1, 3, 3}, {2, 3, 5}, {3, 3, 7}, {4, 3, 1}, {5, 3, 3}, {6, 3, 5}, {7, 3, 7}, {8, 3, 1},
    {1, 4, 4}, {2, 4, 6}, {3, 4, 8}, {4, 4, 2}, {5, 4, 4}, {6, 4, 6}, {7, 4, 8}, {8, 4, 2},
    {1, 5, 5}, {2, 5, 7}, {3, 5, 1}, {4, 5, 3}, {5, 5, 5}, {6, 5, 7}, {7, 5, 1}, {8, 5, 3},
    {1, 6, 6}, {2, 6, 8}, {3, 6, 2}, {4, 6, 4}, {5, 6, 6}, {6, 6, 8}, {7, 6, 2}, {8, 6, 4},
    {1, 7, 7}, {2, 7, 1}, {3, 7, 3}, {4, 7, 5}, {5, 7, 7}, {6, 7, 1}, {7, 7, 3}, {8, 7, 5},
    {1, 8, 8}, {2, 8, 2}, {3, 8, 4}, {4, 8, 6}, {5, 8, 8}, {6, 8, 2}, {7, 8, 4}, {8, 8, 6}
};
static const NyxusPixel glcm_dense_mask[] = {
    {1, 1, 1}, {2, 1, 1}, {3, 1, 1}, {4, 1, 1}, {5, 1, 1}, {6, 1, 1}, {7, 1, 1}, {8, 1, 1},
    {1, 2, 1}, {2, 2, 1}, {3, 2, 1}, {4, 2, 1}, {5, 2, 1}, {6, 2, 1}, {7, 2, 1}, {8, 2, 1},
    {1, 3, 1}, {2, 3, 1}, {3, 3, 1}, {4, 3, 1}, {5, 3, 1}, {6, 3, 1}, {7, 3, 1}, {8, 3, 1},
    {1, 4, 1}, {2, 4, 1}, {3, 4, 1}, {4, 4, 1}, {5, 4, 1}, {6, 4, 1}, {7, 4, 1}, {8, 4, 1},
    {1, 5, 1}, {2, 5, 1}, {3, 5, 1}, {4, 5, 1}, {5, 5, 1}, {6, 5, 1}, {7, 5, 1}, {8, 5, 1},
    {1, 6, 1}, {2, 6, 1}, {3, 6, 1}, {4, 6, 1}, {5, 6, 1}, {6, 6, 1}, {7, 6, 1}, {8, 6, 1},
    {1, 7, 1}, {2, 7, 1}, {3, 7, 1}, {4, 7, 1}, {5, 7, 1}, {6, 7, 1}, {7, 7, 1}, {8, 7, 1},
    {1, 8, 1}, {2, 8, 1}, {3, 8, 1}, {4, 8, 1}, {5, 8, 1}, {6, 8, 1}, {7, 8, 1}, {8, 8, 1}
};

inline void test_glcm_ave_pyradiomics_oracle_cpp()
{
    // Production settings for coarse_gray_depth=8, IBSI on (== the Python test config).
    Environment e;
    e.set_ibsi_compliance(true);
    e.set_coarse_gray_depth(8);
    e.compile_feature_settings();
    const Fsettings& s = e.get_feature_settings(typeid(GLCMFeature));

    LR roidata;
    GLCMFeature f;
    GLCMFeature::angles = { 0, 45, 90, 135 };
    load_masked_test_roi_data(roidata, glcm_dense_intensity, glcm_dense_mask,
                              sizeof(glcm_dense_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata, s));
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // PyRadiomics v3.0.1 goldens (radiomics/pyradiomics Docker; symmetricalGLCM=True, binWidth=1,
    // distances=[1], force2D=True, force2Ddimension=0, weightingNorm=None) on the identical fixture.
    struct GV { Nyxus::Feature2D fc; double gold; };
    const std::vector<GV> goldens = {
        { Feature2D::GLCM_ASM_AVE,         0.06353472511 },  // JointEnergy
        { Feature2D::GLCM_CLUPROM_AVE,     263.2873392   },  // ClusterProminence
        { Feature2D::GLCM_CLUSHADE_AVE,    0.01754371053 },  // ClusterShade
        { Feature2D::GLCM_CLUTEND_AVE,     10.89889629   },  // ClusterTendency
        { Feature2D::GLCM_CORRELATION_AVE, 0.03216471499 },  // Correlation
        { Feature2D::GLCM_DIFAVE_AVE,      2.558673469   },  // DifferenceAverage
        { Feature2D::GLCM_DIFENTRO_AVE,    0.7107003608  },  // DifferenceEntropy
        { Feature2D::GLCM_DIFVAR_AVE,      2.945413369   },  // DifferenceVariance
        { Feature2D::GLCM_ENTROPY_AVE,     3.987900382   },  // JointEntropy (== JE, post fix)
        { Feature2D::GLCM_JE_AVE,          3.987900382   },  // JointEntropy
        { Feature2D::GLCM_ID_AVE,          0.3528380102  },  // Id
        { Feature2D::GLCM_IDM_AVE,         0.2785376978  },  // Idm
        { Feature2D::GLCM_IV_AVE,          0.5086337868  },  // InverseVariance
        { Feature2D::GLCM_JAVE_AVE,        4.510204082   },  // JointAverage
        { Feature2D::GLCM_INFOMEAS1_AVE,  -0.6705029852  },  // Imc1
        { Feature2D::GLCM_INFOMEAS2_AVE,   0.9910039918  },  // Imc2
        { Feature2D::GLCM_VARIANCE_AVE,    5.279571012   },  // SumSquares
        { Feature2D::GLCM_JMAX_AVE,        0.06919642857 },  // MaximumProbability
        { Feature2D::GLCM_SUMENTROPY_AVE,  2.559663993   },  // SumEntropy
        { Feature2D::GLCM_ACOR_AVE,        20.5127551    },  // Autocorrelation
    };
    for (const auto& g : goldens)
        ASSERT_TRUE(agrees_gt(roidata.fvals[(int)g.fc][0], g.gold, 100.0))   // 1% band
            << "GLCM _AVE feature code " << (int)g.fc;
}
