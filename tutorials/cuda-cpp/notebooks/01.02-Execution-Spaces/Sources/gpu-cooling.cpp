
#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    float k = 0.5;
    float ambient_temp = 20;
    std::vector<float> temp{ 42, 24, 50 };
    auto transformation = [=] (float temp) { return temp + k * (ambient_temp - temp); };

    std::printf("step  temp[0]  temp[1]  temp[2]\n");
    for (int step = 0; step < 3; step++) {
        std::transform(temp.begin(), temp.end(), temp.begin(), transformation);
        std::printf("%d     %.2f    %.2f    %.2f\n", step, temp[0], temp[1], temp[2]);
    }
}
