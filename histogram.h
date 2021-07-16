#pragma once

#include <math.h>
#include <vector>

#define N_HISTO_BINS 20	// Max number of histogram channels. 5% step.

template <class PixIntens>
class OnlineHistogram
{
public:
	OnlineHistogram() :
		numObs(0),
		useCounts2(false)
	{
		// Reserve buffers for N_HIST_CHANNELS channels
		counts.reserve(N_HISTO_BINS);
		counts2.reserve(N_HISTO_BINS);
	}
	
	void reset()
	{
		numObs = 0;
		counts.clear();
		counts2.clear();
		useCounts2 = false;
		left_bounds.clear();
		right_bounds.clear();
	}
	
	// Diagnostic output
	void print(std::string remark = "")
	{
		unsigned int totAllBins = 0;
		for (int i = 0; i < N_HISTO_BINS; i++)
			totAllBins += counts[i];

		std::cout << "\nHistogram " << remark << std::endl << numObs << " observations, all bins total = " << totAllBins << std::endl;
		for (int i = 0; i < counts.size(); i++)
		{
			std::cout << '[' << i << "]\t" << left_bounds[i] << " -- " << right_bounds[i] << "\t\t" << counts[i] << std::endl;
		}
	}

	// Appends a value to the histogram
	void add_observation (PixIntens x)
	{
		// Count observations for the sanity check
		numObs++;

		if (counts.size() == 0)
		{
			// No histogram initially. Allocate
			double l = x,
				r = l + 1,
				s = (r - l) / double(N_HISTO_BINS);
			for (int i = 0; i < N_HISTO_BINS; i++)
			{
				double a = l + s * i,
					b = a + s;
				left_bounds.push_back(a);
				right_bounds.push_back(b);
				counts.push_back(0);
				counts2.push_back(0);
			}
			// Find the bin
			int bi = findBinIndex(x);
			if (bi < 0 || bi >= N_HISTO_BINS)	// Sanity check
				throw std::exception();
			counts[bi] = counts[bi] + 1;
		}
		else
		{
			// Regular histogram

			// Find the bin
			int bi = findBinIndex(x);
			if (bi >= 0)	// No need to extend the bounds
				counts[bi] = counts[bi] + 1;
			else	// We should extend the histogram
			{
				// Default new bounds same as old ones
				double newL = left_bounds[0],
					newR = right_bounds[N_HISTO_BINS - 1];

				// New (extended) bounds
				switch (bi)
				{
				case -1:
					// Insert to the right
					newR = x + 1;
					break;
				case -2:
					// Insert to the left
					newL = x;
					break;
				default:
					// Something insane. Throw an exception
					throw std::exception();
					break;
				}

				// New step
				double newS = (newR - newL) / double(N_HISTO_BINS);

				// Transfer the old -> new histogram 

				// --Step 1
				std::fill(counts2.begin(), counts2.end(), 0); // Zero the helper histogram's bins

				// --Step 2
				for (int j = 0; j < N_HISTO_BINS; j++)	// Iterate new bins
				{
					double new_l = RoundN (double(j) * newS),
						new_r = RoundN (new_l + newS);

					for (int g = 0; g < N_HISTO_BINS; g++)	// Iterate old bins
					{
						double old_l = left_bounds[g],
							old_r = right_bounds[g];

						// Case 1 -- an entire old bin goes into a new bin
						if (old_l >= new_l && old_r <= new_r)
						{
							// No split bin, we swallow it all
							counts2[j] = counts2[j] + counts[g];
							continue;
						}

						// Case 2 -- an old bin is split between new bins
						if (old_l < new_l && new_l < old_r)
						{
							// We need to split bin 'g'
							double share = old_r - new_l;
							double frac = 1.0; // (old_r - old_l) / share;
							counts2[j] = counts2[j] + counts[g] * frac;
						}
					}
				}

				// --Step 3
				counts = counts2;

				// --Step 4
				for (int i = 0; i < N_HISTO_BINS; i++)
				{
					double a = RoundN (newL + newS * double(i)),
						b = RoundN (a + newS);
					left_bounds[i] = a;
					right_bounds[i] = b;
				}

				// --Step 5: increment the counter
				int bi = findBinIndex (x);
				if (bi < 0 || bi >= N_HISTO_BINS)	// Sanity check
					throw std::exception();
				counts[bi] = counts[bi] + 1;
			}
		}
	}

	// Returns bin's number of items 
	double getBinStats(int binIdx)
	{
		return counts[binIdx];
	}

	// Returns bin's central value 
	double getBinValue(int binIdx)
	{
		double val = (left_bounds[binIdx] + right_bounds[binIdx]) / 2.0;
		return val;
	}

	// Returns estimated robust mean absolute deviation
	double getEstRMAD (int lbound = 10, int rbound = 90)
	{
		if (lbound < 0 || lbound >= 50)
			lbound = 10;
		if (rbound <= 50 || rbound > 100)
			rbound = 90;

		double percentPerBin = 100.f / float(N_HISTO_BINS);

		double robSum = 0.0;
		int robCount = 0;

		// Calculate robust mean
		for (int i = 0; i < N_HISTO_BINS; i++)
		{
			double binSlice = percentPerBin * i;
			if (binSlice >= lbound && binSlice <= rbound)
			{
				robSum += getBinValue(i);
				robCount += counts[i];
			}
		}

		// Empty histogram?
		if (robCount == 0)
			return 0.0;

		double robMean = robSum / robCount;

		// Calculate robust MAD
		robSum = 0.0;
		for (int i = 0; i < N_HISTO_BINS; i++)
		{
			double binSlice = percentPerBin * i;
			if (binSlice >= lbound && binSlice <= rbound)
			{
				robSum += std::abs (getBinValue(i) - robMean);
			}
		}
		double robMAD = robSum / robCount;
		return robMAD;
	}

protected:
	// Returns the bin index for the value 'x'. In case of an error, returns an (illegal) negative bin index
	int findBinIndex(PixIntens x)
	{
		// x is to the left from the current histogram?
		if (x < left_bounds[0])
			return -2;

		// x is to the right from the current histogram?
		if (x > right_bounds[N_HISTO_BINS - 1])
			return -1;

		// Brute force search
		for (int i = 0; i < N_HISTO_BINS; i++)
			if (left_bounds[i] <= x && x <= right_bounds[i])
				return i;

		// Sanity
		throw std::exception();
		return -3;
	}

	// Helper function
	inline double RoundN(double x)
	{
		const double roundFactor = 1000.0;
		auto retval = std::round(x * roundFactor) / roundFactor;
		return retval;
	}

	// Buffers and state variables
	unsigned int numObs;
	std::vector <float> counts, counts2;
	bool useCounts2;
	std::vector <double> left_bounds, right_bounds;
};



