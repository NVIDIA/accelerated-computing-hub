#include "ach.h"

struct zip_iterator 
{
  int *a;
  int *b;

  std::tuple<int, int> operator[](int i) 
  {
    return {a[i], b[i]};
  }
};

struct transform_iterator 
{
  zip_iterator zip;

  int operator[](int i) 
  {
    auto [a, b] = zip[i];
    return abs(a - b);
  }
};

int main() 
{
  std::array<int, 3> a{ 0, 1, 2 };
  std::array<int, 3> b{ 5, 4, 2 };

  zip_iterator zip{a.data(), b.data()};
  transform_iterator it{zip};

  std::printf("it[0]: %d\n", it[0]); // prints 5
  std::printf("it[0]: %d\n", it[1]); // prints 3
}
