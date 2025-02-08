#include "ach.h"

struct counting_iterator 
{
  int operator[](int i) 
  {
    return i;
  }
};

int main() 
{
  counting_iterator it;

  std::printf("it[0]: %d\n", it[0]); // prints 0
  std::printf("it[1]: %d\n", it[1]); // prints 1
}
