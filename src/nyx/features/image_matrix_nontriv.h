#pragma once

#include <string>
#include "aabb.h"
#include "pixel.h"
#include "../image_loader.h"


class OutOfRamPixelCloud
{
public:
	OutOfRamPixelCloud();
	void init (unsigned int _roi_label, std::string name);
	void clear();
	void add_pixel (const Pixel2& p);
	size_t get_size() const;
	Pixel2 get_at (size_t idx) const;

private:
	size_t n_items = 0;
	std::string filepath;
	FILE* pF = nullptr;
	size_t item_size = sizeof(Pixel2::x) + sizeof(Pixel2::y) + sizeof(Pixel2::inten);
};

/// @brief Writable out of RAM version of class ImageMatrix
class WriteImageMatrix_nontriv
{
public:
	WriteImageMatrix_nontriv (const std::string&  _name, unsigned int _roi_label);
	~WriteImageMatrix_nontriv();
	
	// Initialization
	void allocate (int w, int h=1, double ini_value=0.0);
	void init_with_cloud (const OutOfRamPixelCloud& cloud, const AABB& aabb);

	void set_at(int row, int col, double val);
	void set_at(size_t idx,  double val);
	double get_at(int row, int col);
	double get_at(size_t idx);
	double get_max();
	size_t size();
	size_t get_width();
	size_t get_height();
	size_t get_chlen(size_t x);

private:
	std::string filepath;
	FILE* pF = nullptr;
	int width, height;
	size_t item_size = sizeof(double);
};


/// @brief Readable out of RAM version of class ImageMatrix
class ReadImageMatrix_nontriv
{
public:
	ReadImageMatrix_nontriv (const AABB & aabb);
	~ReadImageMatrix_nontriv();
	double get_at (ImageLoader& imloader, size_t row, size_t col);
	double get_at (ImageLoader& imloader, size_t idx);
	int get_width();
	int get_height();

private:
	AABB aabb;
};