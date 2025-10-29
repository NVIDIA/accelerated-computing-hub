#include "ach.h"

struct wrapper
{
   void operator=(int value) {
      // discard value
   }
};

struct discard_iterator 
{
  wrapper operator[](int i) 
  {
    return {};
  }
};

int main() 
{
  discard_iterator it{};

  it[0] = 10;
  it[1] = 20;
}
