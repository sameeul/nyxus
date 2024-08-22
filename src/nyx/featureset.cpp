#include <map>
#include <string>
#include "featureset.h"
#include "roi_cache.h"

using namespace Nyxus;

namespace Nyxus
{
	std::map <std::string, Feature2D> UserFacingFeatureNames =
	{
		//=== 2D features

		// Pixel intensity stats:
		{ "COV", Feature2D::COV },
		{ "COVERED_IMAGE_INTENSITY_RANGE", Feature2D::COVERED_IMAGE_INTENSITY_RANGE },
		{ "ENERGY", Feature2D::ENERGY },
		{ "ENTROPY", Feature2D::ENTROPY },
		{ "EXCESS_KURTOSIS", Feature2D::EXCESS_KURTOSIS },
		{ "HYPERFLATNESS", Feature2D::HYPERFLATNESS },
		{ "HYPERSKEWNESS", Feature2D::HYPERSKEWNESS },
		{ "INTEGRATED_INTENSITY", Feature2D::INTEGRATED_INTENSITY },
		{ "INTERQUARTILE_RANGE", Feature2D::INTERQUARTILE_RANGE },
		{ "KURTOSIS", Feature2D::KURTOSIS },
		{ "MASS_DISPLACEMENT", Feature2D::MASS_DISPLACEMENT },
		{ "MAX", Feature2D::MAX },
		{ "MEAN", Feature2D::MEAN },
		{ "MEAN_ABSOLUTE_DEVIATION", Feature2D::MEAN_ABSOLUTE_DEVIATION },
		{ "MEDIAN", Feature2D::MEDIAN },
		{ "MEDIAN_ABSOLUTE_DEVIATION", Feature2D::MEDIAN_ABSOLUTE_DEVIATION },
		{ "MIN", Feature2D::MIN },
		{ "MODE", Feature2D::MODE },
		{ "P01", Feature2D::P01 },
		{ "P10", Feature2D::P10 },
		{ "P25", Feature2D::P25 },
		{ "P75", Feature2D::P75 },
		{ "P90", Feature2D::P90 },
		{ "P99", Feature2D::P99 },
		{ "QCOD", Feature2D::QCOD },
		{ "RANGE", Feature2D::RANGE },
		{ "ROBUST_MEAN", Feature2D::ROBUST_MEAN },
		{ "ROBUST_MEAN_ABSOLUTE_DEVIATION", Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION },
		{ "ROOT_MEAN_SQUARED", Feature2D::ROOT_MEAN_SQUARED },
		{ "SKEWNESS", Feature2D::SKEWNESS },
		{ "STANDARD_DEVIATION", Feature2D::STANDARD_DEVIATION },
		{ "STANDARD_DEVIATION_BIASED", Feature2D::STANDARD_DEVIATION_BIASED },
		{ "STANDARD_ERROR", Feature2D::STANDARD_ERROR },
		{ "UNIFORMITY", Feature2D::UNIFORMITY },
		{ "UNIFORMITY_PIU", Feature2D::UNIFORMITY_PIU },
		{ "VARIANCE", Feature2D::VARIANCE },
		{ "VARIANCE_BIASED", Feature2D::VARIANCE_BIASED },
		{ "WEIGHTED_CENTROID_X", Feature2D::WEIGHTED_CENTROID_X },
		{ "WEIGHTED_CENTROID_Y", Feature2D::WEIGHTED_CENTROID_Y },

		// Morphology:
		{ "AREA_PIXELS_COUNT", Feature2D::AREA_PIXELS_COUNT },
		{ "AREA_UM2", Feature2D::AREA_UM2 },
		{ "CENTROID_X", Feature2D::CENTROID_X },
		{ "CENTROID_Y", Feature2D::CENTROID_Y },
		{ "COMPACTNESS", Feature2D::COMPACTNESS },
		{ "BBOX_YMIN", Feature2D::BBOX_YMIN },
		{ "BBOX_XMIN", Feature2D::BBOX_XMIN },
		{ "BBOX_HEIGHT", Feature2D::BBOX_HEIGHT },
		{ "BBOX_WIDTH", Feature2D::BBOX_WIDTH },
		{ "DIAMETER_EQUAL_AREA", Feature2D::DIAMETER_EQUAL_AREA },

		// Ellipse fitting:
		{ "MAJOR_AXIS_LENGTH", Feature2D::MAJOR_AXIS_LENGTH },
		{ "MINOR_AXIS_LENGTH", Feature2D::MINOR_AXIS_LENGTH },
		{ "ECCENTRICITY", Feature2D::ECCENTRICITY},
		{ "ELONGATION", Feature2D::ELONGATION},
		{ "ORIENTATION", Feature2D::ORIENTATION},
		{ "ROUNDNESS", Feature2D::ROUNDNESS},

		// Neighbor features:
		{ "NUM_NEIGHBORS", Feature2D::NUM_NEIGHBORS},
		{ "PERCENT_TOUCHING", Feature2D::PERCENT_TOUCHING},
		{ "CLOSEST_NEIGHBOR1_DIST", Feature2D::CLOSEST_NEIGHBOR1_DIST},
		{ "CLOSEST_NEIGHBOR1_ANG", Feature2D::CLOSEST_NEIGHBOR1_ANG},
		{ "CLOSEST_NEIGHBOR2_DIST", Feature2D::CLOSEST_NEIGHBOR2_DIST},
		{ "CLOSEST_NEIGHBOR2_ANG", Feature2D::CLOSEST_NEIGHBOR2_ANG},
		{ "ANG_BW_NEIGHBORS_MEAN", Feature2D::ANG_BW_NEIGHBORS_MEAN},
		{ "ANG_BW_NEIGHBORS_STDDEV", Feature2D::ANG_BW_NEIGHBORS_STDDEV},
		{ "ANG_BW_NEIGHBORS_MODE", Feature2D::ANG_BW_NEIGHBORS_MODE},

		{ "EXTENT", Feature2D::EXTENT},
		{ "ASPECT_RATIO", Feature2D::ASPECT_RATIO},

		{ "DIAMETER_EQUAL_PERIMETER", Feature2D::DIAMETER_EQUAL_PERIMETER },
		{ "EDGE_MEAN_INTENSITY", Feature2D::EDGE_MEAN_INTENSITY },
		{ "EDGE_STDDEV_INTENSITY", Feature2D::EDGE_STDDEV_INTENSITY},
		{ "EDGE_MAX_INTENSITY", Feature2D::EDGE_MAX_INTENSITY},
		{ "EDGE_MIN_INTENSITY", Feature2D::EDGE_MIN_INTENSITY},

		{ "CONVEX_HULL_AREA", Feature2D::CONVEX_HULL_AREA},
		{ "SOLIDITY", Feature2D::SOLIDITY},

		{ "PERIMETER", Feature2D::PERIMETER},
		{ "EDGE_MEAN_INTENSITY", Feature2D::PERIMETER},
		{ "EDGE_STDDEV_INTENSITY", Feature2D::PERIMETER},
		{ "EDGE_MAX_INTENSITY", Feature2D::PERIMETER},
		{ "EDGE_MIN_INTENSITY", Feature2D::PERIMETER},
		{ "EDGE_INTEGRATED_INTENSITY", Feature2D::EDGE_INTEGRATED_INTENSITY},

		{ "CIRCULARITY", Feature2D::CIRCULARITY},	// alternatively called 'form factor'

		{ "EROSIONS_2_VANISH", Feature2D::EROSIONS_2_VANISH},
		{ "EROSIONS_2_VANISH_COMPLEMENT", Feature2D::EROSIONS_2_VANISH_COMPLEMENT},

		{ "FRACT_DIM_BOXCOUNT", Feature2D::FRACT_DIM_BOXCOUNT},
		{ "FRACT_DIM_PERIMETER", Feature2D::FRACT_DIM_PERIMETER},

		{ "EXTREMA_P1_X", Feature2D::EXTREMA_P1_X},
		{ "EXTREMA_P1_Y", Feature2D::EXTREMA_P1_Y},
		{ "EXTREMA_P2_X", Feature2D::EXTREMA_P2_X},
		{ "EXTREMA_P2_Y", Feature2D::EXTREMA_P2_Y},
		{ "EXTREMA_P3_X", Feature2D::EXTREMA_P3_X},
		{ "EXTREMA_P3_Y", Feature2D::EXTREMA_P3_Y},
		{ "EXTREMA_P4_X", Feature2D::EXTREMA_P4_X},
		{ "EXTREMA_P4_Y", Feature2D::EXTREMA_P4_Y},
		{ "EXTREMA_P5_X", Feature2D::EXTREMA_P5_X},
		{ "EXTREMA_P5_Y", Feature2D::EXTREMA_P5_Y},
		{ "EXTREMA_P6_X", Feature2D::EXTREMA_P6_X},
		{ "EXTREMA_P6_Y", Feature2D::EXTREMA_P6_Y},
		{ "EXTREMA_P7_X", Feature2D::EXTREMA_P7_X},
		{ "EXTREMA_P7_Y", Feature2D::EXTREMA_P7_Y},
		{ "EXTREMA_P8_X", Feature2D::EXTREMA_P8_X},
		{ "EXTREMA_P8_Y", Feature2D::EXTREMA_P8_Y},

		// Caliper features:
		{ "MIN_FERET_ANGLE", Feature2D::MIN_FERET_ANGLE},
		{ "MAX_FERET_ANGLE", Feature2D::MAX_FERET_ANGLE},

		{ "STAT_FERET_DIAM_MIN", Feature2D::STAT_FERET_DIAM_MIN},
		{ "STAT_FERET_DIAM_MAX", Feature2D::STAT_FERET_DIAM_MAX},
		{ "STAT_FERET_DIAM_MEAN", Feature2D::STAT_FERET_DIAM_MEAN},
		{ "STAT_FERET_DIAM_MEDIAN", Feature2D::STAT_FERET_DIAM_MEDIAN},
		{ "STAT_FERET_DIAM_STDDEV", Feature2D::STAT_FERET_DIAM_STDDEV},
		{ "STAT_FERET_DIAM_MODE", Feature2D::STAT_FERET_DIAM_MODE},

		{ "STAT_MARTIN_DIAM_MIN", Feature2D::STAT_MARTIN_DIAM_MIN},
		{ "STAT_MARTIN_DIAM_MAX", Feature2D::STAT_MARTIN_DIAM_MAX},
		{ "STAT_MARTIN_DIAM_MEAN", Feature2D::STAT_MARTIN_DIAM_MEAN},
		{ "STAT_MARTIN_DIAM_MEDIAN", Feature2D::STAT_MARTIN_DIAM_MEDIAN},
		{ "STAT_MARTIN_DIAM_STDDEV", Feature2D::STAT_MARTIN_DIAM_STDDEV},
		{ "STAT_MARTIN_DIAM_MODE", Feature2D::STAT_MARTIN_DIAM_MODE},

		{ "STAT_NASSENSTEIN_DIAM_MIN", Feature2D::STAT_NASSENSTEIN_DIAM_MIN},
		{ "STAT_NASSENSTEIN_DIAM_MAX", Feature2D::STAT_NASSENSTEIN_DIAM_MAX},
		{ "STAT_NASSENSTEIN_DIAM_MEAN", Feature2D::STAT_NASSENSTEIN_DIAM_MEAN},
		{ "STAT_NASSENSTEIN_DIAM_MEDIAN", Feature2D::STAT_NASSENSTEIN_DIAM_MEDIAN},
		{ "STAT_NASSENSTEIN_DIAM_STDDEV", Feature2D::STAT_NASSENSTEIN_DIAM_STDDEV},
		{ "STAT_NASSENSTEIN_DIAM_MODE", Feature2D::STAT_NASSENSTEIN_DIAM_MODE},

		{ "MAXCHORDS_MAX", Feature2D::MAXCHORDS_MAX},
		{ "MAXCHORDS_MAX_ANG", Feature2D::MAXCHORDS_MAX_ANG},
		{ "MAXCHORDS_MIN", Feature2D::MAXCHORDS_MIN},
		{ "MAXCHORDS_MIN_ANG", Feature2D::MAXCHORDS_MIN_ANG},
		{ "MAXCHORDS_MEDIAN", Feature2D::MAXCHORDS_MEDIAN},
		{ "MAXCHORDS_MEAN", Feature2D::MAXCHORDS_MEAN},
		{ "MAXCHORDS_MODE", Feature2D::MAXCHORDS_MODE},
		{ "MAXCHORDS_STDDEV", Feature2D::MAXCHORDS_STDDEV},
		{ "ALLCHORDS_MAX", Feature2D::ALLCHORDS_MAX},
		{ "ALLCHORDS_MAX_ANG", Feature2D::ALLCHORDS_MAX_ANG},
		{ "ALLCHORDS_MIN", Feature2D::ALLCHORDS_MIN},
		{ "ALLCHORDS_MIN_ANG", Feature2D::ALLCHORDS_MIN_ANG},
		{ "ALLCHORDS_MEDIAN", Feature2D::ALLCHORDS_MEDIAN},
		{ "ALLCHORDS_MEAN", Feature2D::ALLCHORDS_MEAN},
		{ "ALLCHORDS_MODE", Feature2D::ALLCHORDS_MODE},
		{ "ALLCHORDS_STDDEV", Feature2D::ALLCHORDS_STDDEV},

		// Miscellaneous:
		{ "EULER_NUMBER", Feature2D::EULER_NUMBER},

		{ "POLYGONALITY_AVE", Feature2D::POLYGONALITY_AVE},
		{ "HEXAGONALITY_AVE", Feature2D::HEXAGONALITY_AVE},
		{ "HEXAGONALITY_STDDEV", Feature2D::HEXAGONALITY_STDDEV},

		{ "DIAMETER_MIN_ENCLOSING_CIRCLE", Feature2D::DIAMETER_MIN_ENCLOSING_CIRCLE},
		{ "DIAMETER_CIRCUMSCRIBING_CIRCLE", Feature2D::DIAMETER_CIRCUMSCRIBING_CIRCLE },
		{ "DIAMETER_INSCRIBING_CIRCLE", Feature2D::DIAMETER_INSCRIBING_CIRCLE },
		{ "GEODETIC_LENGTH", Feature2D::GEODETIC_LENGTH },
		{ "THICKNESS", Feature2D::THICKNESS },

		// ROI radius:
		{ "ROI_RADIUS_MEAN", Feature2D::ROI_RADIUS_MEAN },
		{ "ROI_RADIUS_MAX", Feature2D::ROI_RADIUS_MAX },
		{ "ROI_RADIUS_MEDIAN", Feature2D::ROI_RADIUS_MEDIAN },

		// Texture:
		{ "GLCM_ASM", Feature2D::GLCM_ASM },
		{ "GLCM_ACOR", Feature2D::GLCM_ACOR},
		{ "GLCM_CLUPROM", Feature2D::GLCM_CLUPROM },
		{ "GLCM_CLUSHADE", Feature2D::GLCM_CLUSHADE },
		{ "GLCM_CLUTEND", Feature2D::GLCM_CLUTEND },
		{ "GLCM_CONTRAST", Feature2D::GLCM_CONTRAST },
		{ "GLCM_CORRELATION", Feature2D::GLCM_CORRELATION },
		{ "GLCM_DIFAVE", Feature2D::GLCM_DIFAVE },
		{ "GLCM_DIFVAR", Feature2D::GLCM_DIFVAR },
		{ "GLCM_DIFENTRO", Feature2D::GLCM_DIFENTRO },
		{ "GLCM_DIS", Feature2D::GLCM_DIS },
		{ "GLCM_ENERGY", Feature2D::GLCM_ENERGY },
		{ "GLCM_ENTROPY", Feature2D::GLCM_ENTROPY },
		{ "GLCM_HOM1", Feature2D::GLCM_HOM1 },
		{ "GLCM_HOM2", Feature2D::GLCM_HOM2 },
		{ "GLCM_ID", Feature2D::GLCM_ID },
		{ "GLCM_IDN", Feature2D::GLCM_IDN },
		{ "GLCM_IDM", Feature2D::GLCM_IDM },
		{ "GLCM_IDMN", Feature2D::GLCM_IDMN },
		{ "GLCM_INFOMEAS1", Feature2D::GLCM_INFOMEAS1 },
		{ "GLCM_INFOMEAS2", Feature2D::GLCM_INFOMEAS2 },
		{ "GLCM_IV", Feature2D::GLCM_IV },
		{ "GLCM_JAVE", Feature2D::GLCM_JAVE },
		{ "GLCM_JE", Feature2D::GLCM_JE },
		{ "GLCM_JMAX", Feature2D::GLCM_JMAX },
		{ "GLCM_JVAR", Feature2D::GLCM_JVAR },
		{ "GLCM_SUMAVERAGE", Feature2D::GLCM_SUMAVERAGE },
		{ "GLCM_SUMVARIANCE", Feature2D::GLCM_SUMVARIANCE },
		{ "GLCM_SUMENTROPY", Feature2D::GLCM_SUMENTROPY },
		{ "GLCM_VARIANCE", Feature2D::GLCM_VARIANCE },

		{ "GLCM_ASM_AVE", Feature2D::GLCM_ASM_AVE },
		{ "GLCM_ACOR_AVE", Feature2D::GLCM_ACOR_AVE },
		{ "GLCM_CLUPROM_AVE", Feature2D::GLCM_CLUPROM_AVE },
		{ "GLCM_CLUSHADE_AVE", Feature2D::GLCM_CLUSHADE_AVE },
		{ "GLCM_CLUTEND_AVE", Feature2D::GLCM_CLUTEND_AVE },
		{ "GLCM_CONTRAST_AVE", Feature2D::GLCM_CONTRAST_AVE },
		{ "GLCM_CORRELATION_AVE", Feature2D::GLCM_CORRELATION_AVE },
		{ "GLCM_DIFAVE_AVE", Feature2D::GLCM_DIFAVE_AVE },
		{ "GLCM_DIFVAR_AVE", Feature2D::GLCM_DIFVAR_AVE },
		{ "GLCM_DIFENTRO_AVE", Feature2D::GLCM_DIFENTRO_AVE },
		{ "GLCM_DIS_AVE", Feature2D::GLCM_DIS_AVE },
		{ "GLCM_ENERGY_AVE", Feature2D::GLCM_ENERGY_AVE },
		{ "GLCM_ENTROPY_AVE", Feature2D::GLCM_ENTROPY_AVE },
		{ "GLCM_HOM1_AVE", Feature2D::GLCM_HOM1_AVE },
		{ "GLCM_ID_AVE", Feature2D::GLCM_ID_AVE },
		{ "GLCM_IDN_AVE", Feature2D::GLCM_IDN_AVE },
		{ "GLCM_IDM_AVE", Feature2D::GLCM_IDM_AVE },
		{ "GLCM_IDMN_AVE", Feature2D::GLCM_IDMN_AVE },
		{ "GLCM_IV_AVE", Feature2D::GLCM_IV_AVE },
		{ "GLCM_JAVE_AVE", Feature2D::GLCM_JAVE_AVE },
		{ "GLCM_JE_AVE", Feature2D::GLCM_JE_AVE },
		{ "GLCM_INFOMEAS1_AVE", Feature2D::GLCM_INFOMEAS1_AVE },
		{ "GLCM_INFOMEAS2_AVE", Feature2D::GLCM_INFOMEAS2_AVE },
		{ "GLCM_VARIANCE_AVE", Feature2D::GLCM_VARIANCE_AVE },
		{ "GLCM_JMAX_AVE", Feature2D::GLCM_JMAX_AVE },
		{ "GLCM_JVAR_AVE", Feature2D::GLCM_JVAR_AVE },
		{ "GLCM_SUMAVERAGE_AVE", Feature2D::GLCM_SUMAVERAGE_AVE },
		{ "GLCM_SUMVARIANCE_AVE", Feature2D::GLCM_SUMVARIANCE_AVE },
		{ "GLCM_SUMENTROPY_AVE", Feature2D::GLCM_SUMENTROPY_AVE },
		{ "GLCM_VARIANCE_AVE", Feature2D::GLCM_VARIANCE_AVE },

		{ "GLRLM_SRE", Feature2D::GLRLM_SRE },
		{ "GLRLM_LRE", Feature2D::GLRLM_LRE },
		{ "GLRLM_GLN", Feature2D::GLRLM_GLN},
		{ "GLRLM_GLNN", Feature2D::GLRLM_GLNN},
		{ "GLRLM_RLN", Feature2D::GLRLM_RLN},
		{ "GLRLM_RLNN", Feature2D::GLRLM_RLNN},
		{ "GLRLM_RP", Feature2D::GLRLM_RP},
		{ "GLRLM_GLV", Feature2D::GLRLM_GLV},
		{ "GLRLM_RV", Feature2D::GLRLM_RV},
		{ "GLRLM_RE", Feature2D::GLRLM_RE},
		{ "GLRLM_LGLRE", Feature2D::GLRLM_LGLRE},
		{ "GLRLM_HGLRE", Feature2D::GLRLM_HGLRE},
		{ "GLRLM_SRLGLE", Feature2D::GLRLM_SRLGLE},
		{ "GLRLM_SRHGLE", Feature2D::GLRLM_SRHGLE},
		{ "GLRLM_LRLGLE", Feature2D::GLRLM_LRLGLE},
		{ "GLRLM_LRHGLE", Feature2D::GLRLM_LRHGLE},

		{ "GLRLM_SRE_AVE", Feature2D::GLRLM_SRE_AVE},
		{ "GLRLM_LRE_AVE", Feature2D::GLRLM_LRE_AVE},
		{ "GLRLM_GLN_AVE", Feature2D::GLRLM_GLN_AVE },
		{ "GLRLM_GLNN_AVE", Feature2D::GLRLM_GLNN_AVE },
		{ "GLRLM_RLN_AVE", Feature2D::GLRLM_RLN_AVE },
		{ "GLRLM_RLNN_AVE", Feature2D::GLRLM_RLNN_AVE },
		{ "GLRLM_RP_AVE", Feature2D::GLRLM_RP_AVE },
		{ "GLRLM_GLV_AVE", Feature2D::GLRLM_GLV_AVE },
		{ "GLRLM_RV_AVE", Feature2D::GLRLM_RV_AVE },
		{ "GLRLM_RE_AVE", Feature2D::GLRLM_RE_AVE },
		{ "GLRLM_LGLRE_AVE", Feature2D::GLRLM_LGLRE_AVE },
		{ "GLRLM_HGLRE_AVE", Feature2D::GLRLM_HGLRE_AVE },
		{ "GLRLM_SRLGLE_AVE", Feature2D::GLRLM_SRLGLE_AVE },
		{ "GLRLM_SRHGLE_AVE", Feature2D::GLRLM_SRHGLE_AVE },
		{ "GLRLM_LRLGLE_AVE", Feature2D::GLRLM_LRLGLE_AVE },
		{ "GLRLM_LRHGLE_AVE", Feature2D::GLRLM_LRHGLE_AVE },

		{ "GLDZM_SDE", Feature2D::GLDZM_SDE},
		{ "GLDZM_LDE", Feature2D::GLDZM_LDE},
		{ "GLDZM_LGLZE", Feature2D::GLDZM_LGLZE},
		{ "GLDZM_HGLZE", Feature2D::GLDZM_HGLZE},
		{ "GLDZM_SDLGLE", Feature2D::GLDZM_SDLGLE},
		{ "GLDZM_SDHGLE", Feature2D::GLDZM_SDHGLE},
		{ "GLDZM_LDLGLE", Feature2D::GLDZM_LDLGLE},
		{ "GLDZM_LDHGLE", Feature2D::GLDZM_LDHGLE},
		{ "GLDZM_GLNU", Feature2D::GLDZM_GLNU},
		{ "GLDZM_GLNUN", Feature2D::GLDZM_GLNUN},
		{ "GLDZM_ZDNU", Feature2D::GLDZM_ZDNU},
		{ "GLDZM_ZDNUN", Feature2D::GLDZM_ZDNUN},
		{ "GLDZM_ZP", Feature2D::GLDZM_ZP},
		{ "GLDZM_GLM", Feature2D::GLDZM_GLM},
		{ "GLDZM_GLV", Feature2D::GLDZM_GLV},
		{ "GLDZM_ZDM", Feature2D::GLDZM_ZDM},
		{ "GLDZM_ZDV", Feature2D::GLDZM_ZDV},
		{ "GLDZM_ZDE", Feature2D::GLDZM_ZDE},

		{ "GLSZM_SAE", Feature2D::GLSZM_SAE},
		{ "GLSZM_LAE", Feature2D::GLSZM_LAE},
		{ "GLSZM_GLN", Feature2D::GLSZM_GLN },
		{ "GLSZM_GLNN", Feature2D::GLSZM_GLNN },
		{ "GLSZM_SZN", Feature2D::GLSZM_SZN },
		{ "GLSZM_SZNN", Feature2D::GLSZM_SZNN },
		{ "GLSZM_ZP", Feature2D::GLSZM_ZP },
		{ "GLSZM_GLV", Feature2D::GLSZM_GLV },
		{ "GLSZM_ZV", Feature2D::GLSZM_ZV },
		{ "GLSZM_ZE", Feature2D::GLSZM_ZE },
		{ "GLSZM_LGLZE", Feature2D::GLSZM_LGLZE },
		{ "GLSZM_HGLZE", Feature2D::GLSZM_HGLZE },
		{ "GLSZM_SALGLE", Feature2D::GLSZM_SALGLE },
		{ "GLSZM_SAHGLE", Feature2D::GLSZM_SAHGLE },
		{ "GLSZM_LALGLE", Feature2D::GLSZM_LALGLE},
		{ "GLSZM_LAHGLE", Feature2D::GLSZM_LAHGLE},

		{ "GLDM_SDE", Feature2D::GLDM_SDE },
		{ "GLDM_LDE", Feature2D::GLDM_LDE },
		{ "GLDM_GLN", Feature2D::GLDM_GLN },
		{ "GLDM_DN", Feature2D::GLDM_DN },
		{ "GLDM_DNN", Feature2D::GLDM_DNN },
		{ "GLDM_GLV", Feature2D::GLDM_GLV },
		{ "GLDM_DV", Feature2D::GLDM_DV },
		{ "GLDM_DE", Feature2D::GLDM_DE },
		{ "GLDM_LGLE", Feature2D::GLDM_LGLE },
		{ "GLDM_HGLE", Feature2D::GLDM_HGLE },
		{ "GLDM_SDLGLE", Feature2D::GLDM_SDLGLE },
		{ "GLDM_SDHGLE", Feature2D::GLDM_SDHGLE },
		{ "GLDM_LDLGLE", Feature2D::GLDM_LDLGLE },
		{ "GLDM_LDHGLE", Feature2D::GLDM_LDHGLE },

		{ "NGLDM_LDE", Feature2D::NGLDM_LDE},
		{ "NGLDM_HDE", Feature2D::NGLDM_HDE },
		{ "NGLDM_LGLCE", Feature2D::NGLDM_LGLCE },
		{ "NGLDM_HGLCE", Feature2D::NGLDM_HGLCE },
		{ "NGLDM_LDLGLE", Feature2D::NGLDM_LDLGLE },
		{ "NGLDM_LDHGLE", Feature2D::NGLDM_LDHGLE },
		{ "NGLDM_HDLGLE", Feature2D::NGLDM_HDLGLE },
		{ "NGLDM_HDHGLE", Feature2D::NGLDM_HDHGLE },
		{ "NGLDM_GLNU", Feature2D::NGLDM_GLNU },
		{ "NGLDM_GLNUN", Feature2D::NGLDM_GLNUN },
		{ "NGLDM_DCNU", Feature2D::NGLDM_DCNU },
		{ "NGLDM_DCNUN", Feature2D::NGLDM_DCNUN },
		{ "NGLDM_DCP", Feature2D::NGLDM_DCP },
		{ "NGLDM_GLM", Feature2D::NGLDM_GLM },
		{ "NGLDM_GLV", Feature2D::NGLDM_GLV },
		{ "NGLDM_DCM", Feature2D::NGLDM_DCM },
		{ "NGLDM_DCV", Feature2D::NGLDM_DCV },
		{ "NGLDM_DCENT", Feature2D::NGLDM_DCENT },
		{ "NGLDM_DCENE", Feature2D::NGLDM_DCENE },

		{ "NGTDM_COARSENESS", Feature2D::NGTDM_COARSENESS },
		{ "NGTDM_CONTRAST", Feature2D::NGTDM_CONTRAST },
		{ "NGTDM_BUSYNESS", Feature2D::NGTDM_BUSYNESS },
		{ "NGTDM_COMPLEXITY", Feature2D::NGTDM_COMPLEXITY },
		{ "NGTDM_STRENGTH", Feature2D::NGTDM_STRENGTH },

		// Radial distribution:
		{ "ZERNIKE2D", Feature2D::ZERNIKE2D},
		{ "FRAC_AT_D", Feature2D::FRAC_AT_D },
		{ "MEAN_FRAC", Feature2D::MEAN_FRAC },
		{ "RADIAL_CV", Feature2D::RADIAL_CV },

		// 2D image moments:
		{ "SPAT_MOMENT_00", Feature2D::SPAT_MOMENT_00},
		{ "SPAT_MOMENT_01", Feature2D::SPAT_MOMENT_01},
		{ "SPAT_MOMENT_02", Feature2D::SPAT_MOMENT_02},
		{ "SPAT_MOMENT_03", Feature2D::SPAT_MOMENT_03},
		{ "SPAT_MOMENT_10", Feature2D::SPAT_MOMENT_10},
		{ "SPAT_MOMENT_11", Feature2D::SPAT_MOMENT_11},
		{ "SPAT_MOMENT_12", Feature2D::SPAT_MOMENT_12},
		{ "SPAT_MOMENT_13", Feature2D::SPAT_MOMENT_13},
		{ "SPAT_MOMENT_20", Feature2D::SPAT_MOMENT_20},
		{ "SPAT_MOMENT_21", Feature2D::SPAT_MOMENT_21},
		{ "SPAT_MOMENT_22", Feature2D::SPAT_MOMENT_22},
		{ "SPAT_MOMENT_23", Feature2D::SPAT_MOMENT_23},
		{ "SPAT_MOMENT_30", Feature2D::SPAT_MOMENT_30},

		{ "WEIGHTED_SPAT_MOMENT_00", Feature2D::WEIGHTED_SPAT_MOMENT_00},
		{ "WEIGHTED_SPAT_MOMENT_01", Feature2D::WEIGHTED_SPAT_MOMENT_01},
		{ "WEIGHTED_SPAT_MOMENT_02", Feature2D::WEIGHTED_SPAT_MOMENT_02},
		{ "WEIGHTED_SPAT_MOMENT_03", Feature2D::WEIGHTED_SPAT_MOMENT_03},
		{ "WEIGHTED_SPAT_MOMENT_10", Feature2D::WEIGHTED_SPAT_MOMENT_10},
		{ "WEIGHTED_SPAT_MOMENT_11", Feature2D::WEIGHTED_SPAT_MOMENT_11},
		{ "WEIGHTED_SPAT_MOMENT_12", Feature2D::WEIGHTED_SPAT_MOMENT_12},
		{ "WEIGHTED_SPAT_MOMENT_20", Feature2D::WEIGHTED_SPAT_MOMENT_20},
		{ "WEIGHTED_SPAT_MOMENT_21", Feature2D::WEIGHTED_SPAT_MOMENT_21},
		{ "WEIGHTED_SPAT_MOMENT_30", Feature2D::WEIGHTED_SPAT_MOMENT_30},

		{ "CENTRAL_MOMENT_00", Feature2D::CENTRAL_MOMENT_00 },
		{ "CENTRAL_MOMENT_01", Feature2D::CENTRAL_MOMENT_01 },
		{ "CENTRAL_MOMENT_02", Feature2D::CENTRAL_MOMENT_02 },
		{ "CENTRAL_MOMENT_03", Feature2D::CENTRAL_MOMENT_03 },
		{ "CENTRAL_MOMENT_10", Feature2D::CENTRAL_MOMENT_10 },
		{ "CENTRAL_MOMENT_11", Feature2D::CENTRAL_MOMENT_11 },
		{ "CENTRAL_MOMENT_12", Feature2D::CENTRAL_MOMENT_12 },
		{ "CENTRAL_MOMENT_13", Feature2D::CENTRAL_MOMENT_13 },
		{ "CENTRAL_MOMENT_20", Feature2D::CENTRAL_MOMENT_20 },
		{ "CENTRAL_MOMENT_21", Feature2D::CENTRAL_MOMENT_21 },
		{ "CENTRAL_MOMENT_22", Feature2D::CENTRAL_MOMENT_22 },
		{ "CENTRAL_MOMENT_23", Feature2D::CENTRAL_MOMENT_23 },
		{ "CENTRAL_MOMENT_30", Feature2D::CENTRAL_MOMENT_30 },
		{ "CENTRAL_MOMENT_31", Feature2D::CENTRAL_MOMENT_31 },
		{ "CENTRAL_MOMENT_32", Feature2D::CENTRAL_MOMENT_32 },
		{ "CENTRAL_MOMENT_33", Feature2D::CENTRAL_MOMENT_33 },

		{ "WEIGHTED_CENTRAL_MOMENT_02", Feature2D::WEIGHTED_CENTRAL_MOMENT_02 },
		{ "WEIGHTED_CENTRAL_MOMENT_03", Feature2D::WEIGHTED_CENTRAL_MOMENT_03 },
		{ "WEIGHTED_CENTRAL_MOMENT_11", Feature2D::WEIGHTED_CENTRAL_MOMENT_11 },
		{ "WEIGHTED_CENTRAL_MOMENT_12", Feature2D::WEIGHTED_CENTRAL_MOMENT_12 },
		{ "WEIGHTED_CENTRAL_MOMENT_20", Feature2D::WEIGHTED_CENTRAL_MOMENT_20 },
		{ "WEIGHTED_CENTRAL_MOMENT_21", Feature2D::WEIGHTED_CENTRAL_MOMENT_21 },
		{ "WEIGHTED_CENTRAL_MOMENT_30", Feature2D::WEIGHTED_CENTRAL_MOMENT_30 },

		{ "NORM_CENTRAL_MOMENT_02", Feature2D::NORM_CENTRAL_MOMENT_02},
		{ "NORM_CENTRAL_MOMENT_03", Feature2D::NORM_CENTRAL_MOMENT_03},
		{ "NORM_CENTRAL_MOMENT_11", Feature2D::NORM_CENTRAL_MOMENT_11},
		{ "NORM_CENTRAL_MOMENT_12", Feature2D::NORM_CENTRAL_MOMENT_12},
		{ "NORM_CENTRAL_MOMENT_20", Feature2D::NORM_CENTRAL_MOMENT_20},
		{ "NORM_CENTRAL_MOMENT_21", Feature2D::NORM_CENTRAL_MOMENT_21},
		{ "NORM_CENTRAL_MOMENT_30", Feature2D::NORM_CENTRAL_MOMENT_30},

		{ "NORM_SPAT_MOMENT_00", Feature2D::NORM_SPAT_MOMENT_00 },
		{ "NORM_SPAT_MOMENT_01", Feature2D::NORM_SPAT_MOMENT_01 },
		{ "NORM_SPAT_MOMENT_02", Feature2D::NORM_SPAT_MOMENT_02 },
		{ "NORM_SPAT_MOMENT_03", Feature2D::NORM_SPAT_MOMENT_03 },
		{ "NORM_SPAT_MOMENT_10", Feature2D::NORM_SPAT_MOMENT_10 },
		{ "NORM_SPAT_MOMENT_11", Feature2D::NORM_SPAT_MOMENT_11 },
		{ "NORM_SPAT_MOMENT_12", Feature2D::NORM_SPAT_MOMENT_12 },
		{ "NORM_SPAT_MOMENT_13", Feature2D::NORM_SPAT_MOMENT_13 },
		{ "NORM_SPAT_MOMENT_20", Feature2D::NORM_SPAT_MOMENT_20 },
		{ "NORM_SPAT_MOMENT_21", Feature2D::NORM_SPAT_MOMENT_21 },
		{ "NORM_SPAT_MOMENT_22", Feature2D::NORM_SPAT_MOMENT_22 },
		{ "NORM_SPAT_MOMENT_23", Feature2D::NORM_SPAT_MOMENT_23 },
		{ "NORM_SPAT_MOMENT_30", Feature2D::NORM_SPAT_MOMENT_30 },
		{ "NORM_SPAT_MOMENT_31", Feature2D::NORM_SPAT_MOMENT_31 },
		{ "NORM_SPAT_MOMENT_32", Feature2D::NORM_SPAT_MOMENT_32 },
		{ "NORM_SPAT_MOMENT_33", Feature2D::NORM_SPAT_MOMENT_33 },

		{ "HU_M1", Feature2D::HU_M1 },
		{ "HU_M2", Feature2D::HU_M2 },
		{ "HU_M3", Feature2D::HU_M3},
		{ "HU_M4", Feature2D::HU_M4},
		{ "HU_M5", Feature2D::HU_M5},
		{ "HU_M6", Feature2D::HU_M6},
		{ "HU_M7", Feature2D::HU_M7},

		{ "WT_NORM_CTR_MOM_02", Feature2D::WT_NORM_CTR_MOM_02 },
		{ "WT_NORM_CTR_MOM_03", Feature2D::WT_NORM_CTR_MOM_03 },
		{ "WT_NORM_CTR_MOM_11", Feature2D::WT_NORM_CTR_MOM_11 },
		{ "WT_NORM_CTR_MOM_12", Feature2D::WT_NORM_CTR_MOM_12 },
		{ "WT_NORM_CTR_MOM_20", Feature2D::WT_NORM_CTR_MOM_20 },
		{ "WT_NORM_CTR_MOM_21", Feature2D::WT_NORM_CTR_MOM_21 },
		{ "WT_NORM_CTR_MOM_30", Feature2D::WT_NORM_CTR_MOM_30 },

		{ "WEIGHTED_HU_M1", Feature2D::WEIGHTED_HU_M1},
		{ "WEIGHTED_HU_M2", Feature2D::WEIGHTED_HU_M2},
		{ "WEIGHTED_HU_M3", Feature2D::WEIGHTED_HU_M3},
		{ "WEIGHTED_HU_M4", Feature2D::WEIGHTED_HU_M4},
		{ "WEIGHTED_HU_M5", Feature2D::WEIGHTED_HU_M5},
		{ "WEIGHTED_HU_M6", Feature2D::WEIGHTED_HU_M6},
		{ "WEIGHTED_HU_M7", Feature2D::WEIGHTED_HU_M7},

		// Gabor filter based feature:
		{ "GABOR", Feature2D::GABOR },
	};

	// Image quality features
	std::map <std::string, FeatureIMQ> UserFacingIMQFeatureNames = {
		{"FOCUS_SCORE", FeatureIMQ::FOCUS_SCORE},
		{"LOCAL_FOCUS_SCORE", FeatureIMQ::LOCAL_FOCUS_SCORE},
		{"POWER_SPECTRUM_SLOPE", FeatureIMQ::POWER_SPECTRUM_SLOPE},
		{"MAX_SATURATION", FeatureIMQ::MAX_SATURATION},
		{"MIN_SATURATION", FeatureIMQ::MIN_SATURATION},
		{"SHARPNESS", FeatureIMQ::SHARPNESS}
	};

	std::map <std::string, FgroupIMQ> UserFacingIMQFeaturegroupNames = {
		{ "*ALL_IMQ*", FgroupIMQ::ALL_IMQ}
	};

	std::map <std::string, Fgroup2D> UserFacing2dFeaturegroupNames =
	{
		{ "*ALL*", Fgroup2D::FG2_ALL},
		{ "*ALL_INTENSITY*", Fgroup2D::FG2_INTENSITY },
		{ "*ALL_MORPHOLOGY*", Fgroup2D::FG2_MORPHOLOGY },
		{ "*BASIC_MORPHOLOGY*", Fgroup2D::FG2_BASIC_MORPHOLOGY },
		{ "*ALL_GLCM*", Fgroup2D::FG2_GLCM },
		{ "*ALL_GLRLM*", Fgroup2D::FG2_GLRLM },
		{ "*ALL_GLDZM*", Fgroup2D::FG2_GLDZM },
		{ "*ALL_GLSZM*", Fgroup2D::FG2_GLSZM },
		{ "*ALL_GLDM*", Fgroup2D::FG2_GLDM },
		{ "*ALL_NGLDM*", Fgroup2D::FG2_NGLDM },
		{ "*ALL_NGTDM*", Fgroup2D::FG2_NGTDM },
		{ "*ALL_BUT_GABOR*", Fgroup2D::FG2_BUT_GABOR },
		{ "*ALL_BUT_GLCM*", Fgroup2D::FG2_ALL_BUT_GLCM },
		{ "*ALL_EASY*", Fgroup2D::FG2_EASY },
		{ "*ALL_NEIGHBOR*", Fgroup2D::FG2_NEIG },
		{ "*2DMOMENTS*", Fgroup2D::FG2_MOMENTS },
	};

	std::map <std::string, Feature3D> UserFacing_3D_featureNames =
	{
		// Intensity
		{ "3D_COV", Nyxus::Feature3D::COV },
		{ "3D_COVERED_IMAGE_INTENSITY_RANGE", Nyxus::Feature3D::COVERED_IMAGE_INTENSITY_RANGE },
		{ "3D_ENERGY", Nyxus::Feature3D::ENERGY },
		{ "3D_ENTROPY", Nyxus::Feature3D::ENTROPY },
		{ "3D_EXCESS_KURTOSIS", Nyxus::Feature3D::EXCESS_KURTOSIS },
		{ "3D_HYPERFLATNESS", Nyxus::Feature3D::HYPERFLATNESS },
		{ "3D_HYPERSKEWNESS", Nyxus::Feature3D::HYPERSKEWNESS },
		{ "3D_INTEGRATED_INTENSITY", Nyxus::Feature3D::INTEGRATED_INTENSITY },
		{ "3D_INTERQUARTILE_RANGE", Nyxus::Feature3D::INTERQUARTILE_RANGE },
		{ "3D_KURTOSIS", Nyxus::Feature3D::KURTOSIS },
		{ "3D_MAX", Nyxus::Feature3D::MAX },
		{ "3D_MEAN", Nyxus::Feature3D::MEAN },
		{ "3D_MEAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::MEAN_ABSOLUTE_DEVIATION },
		{ "3D_MEDIAN", Nyxus::Feature3D::MEDIAN },
		{ "3D_MEDIAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::MEDIAN_ABSOLUTE_DEVIATION },
		{ "3D_MIN", Nyxus::Feature3D::MIN },
		{ "3D_MODE", Nyxus::Feature3D::MODE },
		{ "3D_P01", Nyxus::Feature3D::P01 },
		{ "3D_P10", Nyxus::Feature3D::P10 },
		{ "3D_P25", Nyxus::Feature3D::P25 },
		{ "3D_P75", Nyxus::Feature3D::P75 },
		{ "3D_P90", Nyxus::Feature3D::P90 },
		{ "3D_P99", Nyxus::Feature3D::P99 },
		{ "3D_QCOD", Nyxus::Feature3D::QCOD },
		{ "3D_RANGE", Nyxus::Feature3D::RANGE },
		{ "3D_ROBUST_MEAN", Nyxus::Feature3D::ROBUST_MEAN },
		{ "3D_ROBUST_MEAN_ABSOLUTE_DEVIATION", Nyxus::Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION },
		{ "3D_ROOT_MEAN_SQUARED", Nyxus::Feature3D::ROOT_MEAN_SQUARED },
		{ "3D_SKEWNESS", Nyxus::Feature3D::SKEWNESS },
		{ "3D_STANDARD_DEVIATION", Nyxus::Feature3D::STANDARD_DEVIATION },
		{ "3D_STANDARD_DEVIATION_BIASED", Nyxus::Feature3D::STANDARD_DEVIATION_BIASED },
		{ "3D_STANDARD_ERROR", Nyxus::Feature3D::STANDARD_ERROR },
		{ "3D_UNIFORMITY", Nyxus::Feature3D::UNIFORMITY },
		{ "3D_UNIFORMITY_PIU", Nyxus::Feature3D::UNIFORMITY_PIU },
		{ "3D_VARIANCE", Nyxus::Feature3D::VARIANCE },
		{ "3D_VARIANCE_BIASED", Nyxus::Feature3D::VARIANCE_BIASED },

		// Morphology
		{ "3D_AREA", Feature3D::AREA },
		{ "3D_MESH_VOLUME", Feature3D::MESH_VOLUME },
		{ "3D_VOLUME_CONVEXHULL", Feature3D::VOLUME_CONVEXHULL },
		{ "3D_DIAMETER_EQUAL_AREA", Feature3D::DIAMETER_EQUAL_AREA },
		{ "3D_DIAMETER_EQUAL_VOLUME", Feature3D::DIAMETER_EQUAL_VOLUME },

		{ "3D_VOLUME_PIXELS", Feature3D::VOLUME_PIXELS },
		{ "3D_CENTROID_X", Feature3D::CENTROID_X },
		{ "3D_CENTROID_Y", Feature3D::CENTROID_Y },
		{ "3D_CENTROID_Z", Feature3D::CENTROID_Z },
		{ "3D_BBOX_XMIN", Feature3D::BBOX_XMIN },
		{ "3D_BBOX_YMIN", Feature3D::BBOX_YMIN },
		{ "3D_BBOX_ZMIN", Feature3D::BBOX_ZMIN },
		{ "3D_BBOX_HEIGHT", Feature3D::BBOX_HEIGHT },
		{ "3D_BBOX_WIDTH", Feature3D::BBOX_WIDTH },
		{ "3D_BBOX_DEPTH", Feature3D::BBOX_DEPTH },

		// Neighbor features
		{ "3D_NUM_NEIGHBORS", Feature3D::NUM_NEIGHBORS },
		{ "3D_PERCENT_TOUCHING", Feature3D::PERCENT_TOUCHING },
		{ "3D_CLOSEST_NEIGHBOR1_DIST", Feature3D::CLOSEST_NEIGHBOR1_DIST },
		{ "3D_CLOSEST_NEIGHBOR1_ANG", Feature3D::CLOSEST_NEIGHBOR1_ANG },
		{ "3D_CLOSEST_NEIGHBOR2_DIST", Feature3D::CLOSEST_NEIGHBOR2_DIST },
		{ "3D_CLOSEST_NEIGHBOR2_ANG", Feature3D::CLOSEST_NEIGHBOR2_ANG },
		{ "3D_ANG_BW_NEIGHBORS_MEAN", Feature3D::ANG_BW_NEIGHBORS_MEAN },
		{ "3D_ANG_BW_NEIGHBORS_STDDEV", Feature3D::ANG_BW_NEIGHBORS_STDDEV },
		{ "3D_ANG_BW_NEIGHBORS_MODE", Feature3D::ANG_BW_NEIGHBORS_MODE },

		// Geometric moments
		{ "3D_SPAT_MOMENT_00", Feature3D::SPAT_MOMENT_00 },
		{ "3D_SPAT_MOMENT_01", Feature3D::SPAT_MOMENT_01 },
		{ "3D_SPAT_MOMENT_02", Feature3D::SPAT_MOMENT_02 },
		{ "3D_SPAT_MOMENT_03", Feature3D::SPAT_MOMENT_03 },
		{ "3D_SPAT_MOMENT_10", Feature3D::SPAT_MOMENT_10 },
		{ "3D_SPAT_MOMENT_11", Feature3D::SPAT_MOMENT_11 },
		{ "3D_SPAT_MOMENT_12", Feature3D::SPAT_MOMENT_12 },
		{ "3D_SPAT_MOMENT_13", Feature3D::SPAT_MOMENT_13 },
		{ "3D_SPAT_MOMENT_20", Feature3D::SPAT_MOMENT_20 },
		{ "3D_SPAT_MOMENT_21", Feature3D::SPAT_MOMENT_21 },
		{ "3D_SPAT_MOMENT_22", Feature3D::SPAT_MOMENT_22 },
		{ "3D_SPAT_MOMENT_23", Feature3D::SPAT_MOMENT_23 },
		{ "3D_SPAT_MOMENT_30", Feature3D::SPAT_MOMENT_30 },

	};

	std::map <std::string, Fgroup3D> UserFacing3dFeaturegroupNames =
	{
		{ "*3D_ALL*", Fgroup3D::FG3_ALL },
		{ "*3D_ALL_INTENSITY*", Fgroup3D::FG3_INTENSITY },
		{ "*3D_ALL_MORPHOLOGY*", Fgroup3D::FG3_MORPHOLOGY },
		{ "*3D_ALL_TEXTURE*", Fgroup3D::FG3_TEXTURE },
		{ "*3D_ALL_NEIGHBOR*", Fgroup3D::FG3_NEIG },
		{ "*3D_MOMENTS*", Fgroup3D::FG3_MOMENTS },
	};

} // namespace Nyxus

FeatureSet::FeatureSet()
{
	enableAll(true);
}

bool FeatureSet::find_2D_FeatureByString (const std::string& featureName, Feature2D& f)
{
	auto it_f = Nyxus::UserFacingFeatureNames.find(featureName);

	if (it_f == Nyxus::UserFacingFeatureNames.end())
		return false;

	f = it_f->second;
	return true;
}

bool FeatureSet::find_2D_GroupByString (const std::string & grpName, Fgroup2D & grpCode)
{
	auto itr = Nyxus::UserFacing2dFeaturegroupNames.find (grpName);

	if (itr == Nyxus::UserFacing2dFeaturegroupNames.end())
		return false;

	grpCode = itr->second;
	return true;
}

bool FeatureSet::find_3D_FeatureByString (const std::string & featureName, Feature3D & f)
{
	auto it_f = Nyxus::UserFacing_3D_featureNames.find (featureName);

	if (it_f == Nyxus::UserFacing_3D_featureNames.end())
		return false;

	f = it_f->second;
	return true;
}

bool FeatureSet::find_3D_GroupByString (const std::string & grpName, Fgroup3D & grpCode)
{
	auto itr = Nyxus::UserFacing3dFeaturegroupNames.find (grpName);

	if (itr == Nyxus::UserFacing3dFeaturegroupNames.end())
		return false;

	grpCode = itr->second;
	return true;
}

bool FeatureSet::find_IMQ_FeatureByString (const std::string& featureName, FeatureIMQ& f)
{
	auto it_f = Nyxus::UserFacingIMQFeatureNames.find(featureName);

	if (it_f == Nyxus::UserFacingIMQFeatureNames.end())
		return false;

	f = it_f->second;
	return true;
}

bool FeatureSet::find_IMQ_GroupByString (const std::string & grpName, FgroupIMQ & grpCode)
{
	auto itr = Nyxus::UserFacingIMQFeaturegroupNames.find (grpName);

	if (itr == Nyxus::UserFacingIMQFeaturegroupNames.end())
		return false;

	grpCode = itr->second;
	return true;
}

std::string FeatureSet::findFeatureNameByCode (Feature2D fcode)
{
	// Search
	auto result = std::find_if(
		Nyxus::UserFacingFeatureNames.begin(),
		Nyxus::UserFacingFeatureNames.end(),
		[fcode](const auto& finfo)
		{ return finfo.second == fcode; });

	// Return the feature name if found
	if (result != Nyxus::UserFacingFeatureNames.end())
		return result->first;

	// Nothing is found
	return "[UNNAMED 2D FEATURE]";
}

std::string FeatureSet::findFeatureNameByCode (Feature3D fcode)
{
	// Search
	auto result = std::find_if(
		Nyxus::UserFacing_3D_featureNames.begin(),
		Nyxus::UserFacing_3D_featureNames.end(),
		[fcode](const auto& finfo)
		{ return finfo.second == fcode; });

	// Return the feature name if found
	if (result != Nyxus::UserFacing_3D_featureNames.end())
		return result->first;

	// Nothing is found
	return "[UNNAMED 3D FEATURE]";
}

std::string FeatureSet::findGroupNameByCode (Fgroup2D code)
{
	// Search
	auto result = std::find_if(
		Nyxus::UserFacing2dFeaturegroupNames.begin(),
		Nyxus::UserFacing2dFeaturegroupNames.end(),
		[code](const auto& finfo)
		{ return finfo.second == code; });

	// Return the feature name if found
	if (result != Nyxus::UserFacing2dFeaturegroupNames.end())
		return result->first;

	// Not found
	return "UNNAMED_FEATURE_GROUP_" + std::to_string((int)code);
}

std::string FeatureSet::findGroupNameByCode (Fgroup3D code)
{
	// Search
	auto result = std::find_if(
		Nyxus::UserFacing3dFeaturegroupNames.begin(),
		Nyxus::UserFacing3dFeaturegroupNames.end(),
		[code](const auto& finfo)
		{ return finfo.second == code; });

	// Return the feature name if found
	if (result != Nyxus::UserFacing3dFeaturegroupNames.end())
		return result->first;

	// Not found
	return "UNNAMED_FEATURE_GROUP_" + std::to_string((int)code);
}

std::string FeatureSet::findGroupNameByCode (FgroupIMQ code)
{
	// Search
	auto result = std::find_if(
		Nyxus::UserFacingIMQFeaturegroupNames.begin(),
		Nyxus::UserFacingIMQFeaturegroupNames.end(),
		[code](const auto& finfo)
		{ return finfo.second == code; });

	// Return the feature name if found
	if (result != Nyxus::UserFacingIMQFeaturegroupNames.end())
		return result->first;

	// Not found
	return "UNNAMED_FEATURE_GROUP_" + std::to_string((int)code);
}

// Relying on RVO rather than std::move
std::vector<std::tuple<std::string, int>> FeatureSet::getEnabledFeatures()
{
	std::vector<std::tuple<std::string, int>> F;

	// 2D features
	for (int i = 0; i < (int) Nyxus::Feature2D::_COUNT_; i++)
	{
		if (m_enabledFeatures[i])
		{
			// Find feature i's name
			std::string fname = "2D-feature" + std::to_string(i);	// name feature<number> will indicate a number of a missing user-facing feature name, if any
			for (const auto& f : Nyxus::UserFacingFeatureNames)
			{
				if (f.second == (Nyxus::Feature2D) i)
				{
					fname = f.first;
					break;
				}
			}

			// Save the pair
			std::tuple<std::string, int> f (fname, i);
			F.push_back(f);
		}
	}

	// 3D features
	for (int i = (int) Nyxus::Feature2D::_COUNT_; i < (int) Nyxus::Feature3D::_COUNT_; i++)
	{
		if (m_enabledFeatures[i])
		{
			// Find feature i's name
			std::string fname = "3D-feature" + std::to_string(i);	// name feature<number> will indicate a number of a missing user-facing feature name, if any
			for (const auto& f : Nyxus::UserFacing_3D_featureNames)
			{
				if (f.second == (Nyxus::Feature3D) i)
				{
					fname = f.first;
					break;
				}
			}

			// Save the pair
			std::tuple<std::string, int> f (fname, i);
			F.push_back(f);
		}
	}

	// Image Quality features
	for (int i = (int) Nyxus::Feature3D::_COUNT_; i < (int) Nyxus::FeatureIMQ::_COUNT_; i++)
	{
		if (m_enabledFeatures[i])
		{
			// Find feature i's name
			std::string fname = "IMQ-feature" + std::to_string(i);	// name feature<number> will indicate a number of a missing user-facing feature name, if any
			for (const auto& f : Nyxus::UserFacingIMQFeatureNames)
			{
				if (f.second == (Nyxus::FeatureIMQ) i)
				{
					fname = f.first;
					break;
				}
			}

			// Save the pair
			std::tuple<std::string, int> f (fname, i);
			F.push_back(f);
		}
	}

	return F;
}
