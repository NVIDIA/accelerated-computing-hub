#include "ach.h"

int main() 
{
    auto ones = thrust::make_constant_iterator(1);

    for (int i = 0; i < 5; i++) {
        std::printf("*iterator: %d\n", *ones);
        ones++;
    }
}
