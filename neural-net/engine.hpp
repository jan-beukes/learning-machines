#pragma once
#include "subprocess.h"

#include <vector>
#include <format>

template<typename T>
struct Value {
    T data;
    std::vector<const Value*> children;
    char op = 0;

    Value(T data) {
        this->data = data;
    }

    Value(T data, std::vector<const Value*> children, char op) {
        this->data = data;
        this->children = children;
        this->op = op;
    }

    Value operator+(const Value &other) const {
        std::vector<const Value*> children = { this, &other };
        return Value(data + other.data, children, '+');
    }

    Value operator-(const Value &other) const {
        std::vector<const Value*> children = { this, &other };
        return Value(data - other.data, children, '-');
    }

    Value operator*(const Value &other) const {
        std::vector<const Value*> children = { this, &other };
        return Value(data * other.data, children, '*');
    }
};

// minimal formatter
template <typename T>
struct std::formatter<Value<T>> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }
    auto format(const Value<T> &v, std::format_context &ctx) const {
        return std::format_to(ctx.out(), "Value({})", v.data);
    }
};
