#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include <iostream>
#include <sstream>
#include <iomanip>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include "environment.h"
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/glcm.h"
#include "features/glrlm.h"
#include "features/zernike.h"
#include "globals.h"

namespace Nyxus
{
	// Macro to make some file i/o calls platform-independent
#ifndef _WIN32
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif

	double auto_precision (std::stringstream& ss, double x)
	{
		if (std::abs(x) >= 1.0)
			ss << std::setprecision(theEnvironment.get_floating_point_precision());
		else
			if (x == 0.0)
				ss << std::setprecision(1);
			else
			{
				// int n = int(std::abs(std::log10(x)) + 0.5);

				double tmp1 = std::abs(x),
					tmp2 = std::log10(tmp1),
					tmp3 = std::abs(tmp2), 
					tmp4 = tmp3 + 0.5;
				int n = int(tmp4);

				ss << std::setprecision(theEnvironment.get_floating_point_precision() + n);
			}

		return x;
	}

	// Saves the result of image scanning and feature calculation. Must be called after the reduction phase.
	bool save_features_2_csv (std::string intFpath, std::string segFpath, std::string outputDir)
	{
		// Sort the labels
		std::vector<int> L{ uniqueLabels.begin(), uniqueLabels.end() };
		std::sort(L.begin(), L.end());

		static bool mustRenderHeader = true;	// This can be flipped to 'false' in 'singlecsv' scenario

		// Make the file name and write mode
		std::string fullPath;
		const char* mode; // = "w";
		if (theEnvironment.separateCsv)
		{
			fullPath = outputDir + "/_INT_" + getPureFname(intFpath) + "_SEG_" + getPureFname(segFpath) + ".csv";
			VERBOSLVL1(std::cout << "\t--> " << fullPath << "\n";)
		}
		else
		{
			fullPath = outputDir + "/" + "NyxusFeatures.csv";
			VERBOSLVL1(std::cout << "\t--> " << fullPath << "\n";)
			mode = mustRenderHeader ? "w" : "a";
		}

		// Open it
		FILE* fp = nullptr;
		fopen_s(&fp, fullPath.c_str(), mode);

		if (!fp)
		{
			std::string errmsg = "Cannot open file " + fullPath + " for writing";
			std::perror (errmsg.c_str());
			return false;
		}

		// -- Configure buffered write
		if (std::setvbuf(fp, nullptr, _IOFBF, 32768) != 0) {
			std::perror("setvbuf failed");
			return false;
		}

		// Learn what features need to be displayed
		std::vector<std::tuple<std::string, AvailableFeatures>> F = theFeatureSet.getEnabledFeatures();

		// -- Header
		if (mustRenderHeader)
		{
			std::stringstream ssHead;

			ssHead << "mask_image,intensity_image,label";

			for (auto& enabdF : F)
			{
				auto fn = std::get<0>(enabdF);	// feature name
				auto fc = std::get<1>(enabdF);	// feature code

				// Handle missing feature name (which is a significant issue!) in order to at least be able to trace back to the feature code
				if (fn.empty())
				{
					std::stringstream temp;
					temp << "feature" << fc;
					fn = temp.str();
				}

				// Parameterized feature
				// --GLCM family
				bool angledGlcmFeature = std::find (GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), fc) != GLCMFeature::featureset.end();
				if (angledGlcmFeature)
				{
					// Polulate with angles
					for (auto ang : theEnvironment.glcmAngles)
					{
						// CSV separator
						//if (ang != theEnvironment.rotAngles[0])
						//	ssHead << ",";
						ssHead << "," << fn << "_" << ang;
					}
					// Proceed with other features
					continue;
				}

				// --GLRLM family
				bool glrlmFeature = std::find (GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), fc) != GLRLMFeature::featureset.end();
				if (glrlmFeature)
				{
					// Polulate with angles
					for (auto ang : GLRLMFeature::rotAngles)
					{
						ssHead << "," << fn << "_" << ang;
					}
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == GABOR)
				{
					// Generate the feature value list
					for (auto i = 0; i < GaborFeature::f0.size(); i++)
						ssHead << "," << fn << "_" << i;

					// Proceed with other features
					continue;
				}

				if (fc == FRAC_AT_D)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
						ssHead << "," << fn << "_" << i;

					// Proceed with other features
					continue;
				}

				if (fc == MEAN_FRAC)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
						ssHead << "," << fn << "_" << i;

					// Proceed with other features
					continue;
				}

				if (fc == RADIAL_CV)
				{
					// Generate the feature value list
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
						ssHead << "," << fn << "_" << i;

					// Proceed with other features
					continue;
				}

				// --Zernike features header 
				if (fc == ZERNIKE2D)
				{
					// Populate with indices
					for (int i = 0; i < ZernikeFeature::num_feature_values_calculated; i++)	// i < ZernikeFeature::num_feature_values_calculated
						ssHead << "," << fn << "_Z" << i;						

					// Proceed with other features
					continue;
				}

				// Regular feature
				ssHead << "," << fn;
			}

			fprintf(fp, "%s\n", ssHead.str().c_str());

			// Prevent rendering the header again for another image's portion of labels
			if (theEnvironment.separateCsv == false)
				mustRenderHeader = false;
		}

		// -- Values
		for (auto l : L)
		{
			LR& r = roiData[l];			
			
			// Skip blacklisted ROI
			if (r.blacklisted)
				continue;

			std::stringstream ssVals;

			// Floating point precision
			ssVals << std::fixed;

			// Tear off pure file names from segment and intensity file paths
			fs::path pseg(r.segFname), pint(r.intFname);
			ssVals << pseg.filename() << "," << pint.filename() << "," << l;

			for (auto& enabdF : F)
			{
				auto fc = std::get<1>(enabdF);
				auto fn = std::get<0>(enabdF);	// debug
				auto vv = r.get_fvals(fc);

				// Parameterized feature
				// --GLCM family
				bool angledGlcmFeature = std::find (GLCMFeature::featureset.begin(), GLCMFeature::featureset.end(), fc) != GLCMFeature::featureset.end();
				if (angledGlcmFeature)
				{
					// Mock angled values if they haven't been calculated for some error reason
					if (vv.size() < GLCMFeature::angles.size())
						vv.resize(GLCMFeature::angles.size(), 0.0);
					// Output the sub-values
					for (int i = 0; i < GLCMFeature::angles.size(); i++) // theEnvironment.rotAngles.size(); i++)
					{
						#ifndef DIAGNOSE_NYXUS_OUTPUT
							ssVals << "," << vv[i];
						#else
							//--diagnoze misalignment-- 
							ssVals << "," << fn << "-" << vv[i];	
						#endif
					}
					// Proceed with other features
					continue;
				}

				// --GLRLM family
				bool glrlmFeature = std::find (GLRLMFeature::featureset.begin(), GLRLMFeature::featureset.end(), fc) != GLRLMFeature::featureset.end();
				if (glrlmFeature)
				{
					// Polulate with angles
					auto nAng = 4; // sizeof(GLRLMFeature::rotAngles) / sizeof(GLRLMFeature::rotAngles[0]);
					for (int i = 0; i < nAng; i++)
					{
						#ifndef DIAGNOSE_NYXUS_OUTPUT
							ssVals << "," << vv[i];
						#else
							//--diagnoze misalignment-- 
							ssVals << "," << fn << "-" << vv[i];	
						#endif
					}
					// Proceed with other features
					continue;
				}

				// --Gabor
				if (fc == GABOR)
				{
					for (auto i = 0; i < GaborFeature::f0.size(); i++)
					{
						#ifndef DIAGNOSE_NYXUS_OUTPUT
							ssVals << "," << vv[i];
						#else	
							//--diagnoze misalignment-- 
							ssVals << "," << fn << "-" << vv[i];	
						#endif			
					}

					// Proceed with other features
					continue;
				}

				// --Zernike feature values
				if (fc == ZERNIKE2D)
				{
					for (int i = 0; i < ZernikeFeature::num_feature_values_calculated; i++)
					{
						#ifndef DIAGNOSE_NYXUS_OUTPUT
							ssVals << "," << vv[i];
						#else
							//--diagnoze misalignment-- 
							ssVals << "," << fn << "-" << vv[i];	
						#endif
					}

					// Proceed with other features
					continue;
				}

				// --Radial distribution features
				if (fc == FRAC_AT_D)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_FracAtD; i++)
					{
						#ifndef DIAGNOSE_NYXUS_OUTPUT
							ssVals << "," << vv[i];
						#else
							//--diagnoze misalignment-- 
							ssVals << "," << fn << "-" << vv[i];	
						#endif
					}
					// Proceed with other features
					continue;
				}
				if (fc == MEAN_FRAC)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_MeanFrac; i++)
					{
						#ifndef DIAGNOSE_NYXUS_OUTPUT
							ssVals << "," << vv[i];
						#else
							//--diagnoze misalignment-- 
							ssVals << "," << fn << "-" << vv[i];	
						#endif
					}
					// Proceed with other features
					continue;
				}
				if (fc == RADIAL_CV)
				{
					for (auto i = 0; i < RadialDistributionFeature::num_features_RadialCV; i++)
					{
						#ifndef DIAGNOSE_NYXUS_OUTPUT
							ssVals << "," << vv[i];
						#else
							//--diagnoze misalignment-- 
							ssVals << "," << fn << "-" << vv[i];	
						#endif
					}
					// Proceed with other features
					continue;
				}

				// Regular feature
				#ifndef DIAGNOSE_NYXUS_OUTPUT
					ssVals << "," << auto_precision (ssVals, vv[0]);
				#else
					//--diagnoze misalignment-- 
					ssVals << "," << fn << "-" << vv[0];	
				#endif
			}

			fprintf(fp, "%s\n", ssVals.str().c_str());
		}

		std::fflush(fp);
		std::fclose(fp);

#ifdef SANITY_CHECK_INTENSITIES_FOR_LABEL
		// Output label's intensities for debug
		for (auto l : L)
		{
			if (l != SANITY_CHECK_INTENSITIES_FOR_LABEL)
				continue;

			std::stringstream ss;
			LR& lr = roiData[l];
			auto& I = lr.raw_intensities;
			ss << outputDir << "/" << "intensities_label_" << l << ".txt";
			fullPath = ss.str();
			std::cout << "Dumping intensities of label " << l << " to file " << fullPath << std::endl;


			fopen_s(&fp, fullPath.c_str(), "w");
			if (fp)
			{
				ss.clear();
				ss << "I_" << l << " = [\n";
				for (auto w : I)
					ss << "\t" << w << ", \n";
				ss << "\t]; \n";
				fprintf(fp, "%s\n", ss.str().c_str());
				std::fclose(fp);
			}
		}
#endif

		return true;
	}

	// Diagnostic function
	void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns)
	{
		std::stringstream ss;

		std::cout << std::endl << featureName << std::endl;

		int i = 1;
		for (auto& x : L)
		{
			ss << 'L' << x.first << ':' << x.second;
			std::cout << std::setw(10) << ss.str();
			ss.str(std::string()); // Clear ss

			if (i++ % numColumns == 0)
				std::cout << std::endl;
		}

	}

	// Another diagnostic function
	void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns)
	{
		std::stringstream ss;

		std::cout << std::endl << featureName << std::endl;

		int i = 1;
		for (auto& x : L)
		{
			ss << 'L' << x.first << ':' << x.second;
			std::cout << std::setw(30) << ss.str();
			ss.str(std::string()); // Clear ss

			if (i++ % numColumns == 0)
				std::cout << std::endl;
		}

	}

}
