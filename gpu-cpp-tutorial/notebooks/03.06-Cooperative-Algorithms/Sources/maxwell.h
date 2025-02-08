#pragma once

#include <algorithm>
#include <vector>
#include <fstream>

#include <thrust/universal_vector.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

struct host_state {
  float dx;
  float dy;
  float dt;
  int cells_along_dimension;

  std::vector<float> hx;
  std::vector<float> hy;
  std::vector<float> ez;
  std::vector<float> dz;
};

static void store(const std::string &filename, const std::vector<float>& ez)
{
  std::ofstream file(filename, std::ios::binary);
  file.write(reinterpret_cast<const char*>(ez.data()), ez.size() * sizeof(float));
}

static void store(const std::vector<float>& ez)
{
  store("solution.bin", ez);
}

static host_state load(const std::string &filename)
{
  host_state state;
  std::ifstream file(filename, std::ios::binary);

  file.read(reinterpret_cast<char*>(&state.dx), sizeof(state.dx));
  file.read(reinterpret_cast<char*>(&state.dy), sizeof(state.dy));
  file.read(reinterpret_cast<char*>(&state.dt), sizeof(state.dt));
  file.read(reinterpret_cast<char*>(&state.cells_along_dimension), sizeof(state.cells_along_dimension));

  const int cells = state.cells_along_dimension * state.cells_along_dimension;
  state.hx.resize(cells);
  state.hy.resize(cells);
  state.ez.resize(cells);
  state.dz.resize(cells);

  file.read(reinterpret_cast<char*>(state.hx.data()), cells * sizeof(float));
  file.read(reinterpret_cast<char*>(state.hy.data()), cells * sizeof(float));
  file.read(reinterpret_cast<char*>(state.dz.data()), cells * sizeof(float));
  std::transform(state.dz.begin(), state.dz.end(), state.ez.begin(), [](float d) { return d / 1.3f; });

  return state;
}

static host_state load()
{
  return load("assignment.bin");
}

// Constants
constexpr float C0 = 299792458.0f;            // Speed of light [metres per second]
constexpr float epsilon_0 = 8.854187817e-12f; // Permittivity of free space
constexpr float mu_0 = 4 * M_PI * 1e-7f;      // Permeability of free space
