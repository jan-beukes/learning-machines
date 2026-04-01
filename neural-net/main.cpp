#include "engine.hpp"

#include <print>

int main()
{
    auto a = Value(2.0);
    auto b = Value(-3.0);
    auto c = Value(10.0);
    auto d = a*b + c;

    std::println("{}", d);
    return 0;
}
