#pragma once

#include <string>
#include <vector>
#include "features/aabb.h"
#include "features/contour.h"
#include "features/f_convex_hull.h"
#include "features/image_matrix.h"
#include "features/pixel.h"
#include "featureset.h"

// Label record - structure aggregating label's cached data and calculated features
#define DFLT0 -0.0	// default unassigned value
#define DFLT0i -0	// default unassigned value
struct LR
{
	int label;
	std::string segFname, intFname;

	bool roi_disabled = false;

	// Helper objects
	std::vector <Pixel2> raw_pixels;
	PixIntens aux_min, aux_max;
	AABB aabb;
	ConvexHull convHull;
	Contour contour;

	// Replaced with a faster version (class TrivialHistogram)	std::shared_ptr<Histo> aux_Histogram;

	StatsInt
		aux_PrevCount,
		aux_PrevIntens;

	// Zernike calculator may put an arbitrary number of Z_a^b terms 
	// but we output only 'NUM_ZERNIKE_COEFFS_2_OUTPUT' of them 
	static const short aux_ZERNIKE2D_ORDER = 9, aux_ZERNIKE2D_NUM_COEFS = 30;	// z00, z11, z20, z22, z31, z33, z40, z42, z44, ... ,z97, z99 - 30 items altogether 
	std::vector<std::vector<StatsReal>> fvals;
	std::vector<StatsReal> getFeatureValues(AvailableFeatures af) { return fvals[af]; }

	StatsReal aux_M2,
		aux_M3,
		aux_M4,
		aux_variance;

	std::vector<int> aux_neighboring_labels;

	ImageMatrix aux_image_matrix;	// Needed by Contour, Erosions, GLCM, GLRLM, GLSZM, GLDM, NGTDM, Radial distribution(via Contour), Gabor, Moments, ROI radius(via Contour)

	void init_aabb(StatsInt x, StatsInt y);
	void update_aabb(StatsInt x, StatsInt y);

	double getValue (AvailableFeatures f);
	void reduce_pixel_intensity_features();
	bool intensitiesAllZero();
	void reduce_edge_intensity_features();
};

