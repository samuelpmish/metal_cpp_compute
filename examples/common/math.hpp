#pragma once

#include <array>
#include <iostream>

using float3 = std::array<float, 3>;
using float4x3 = std::array<float3, 4>;
using uint4 = std::array<uint32_t, 4>;

float3 operator-(const float3 & x, const float3 & y) {
  return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

float4x3 operator-(const float4x3 & x, const float4x3 & y) {
  return {x[0] - y[0], x[1] - y[1], x[2] - y[2], x[3] - y[3]};
}

float3 operator+(const float3 & x, const float3 & y) {
  return {x[0] + y[0], x[1] + y[1], x[2] + y[2]};
}

float4x3 operator+(const float4x3 & x, const float4x3 & y) {
  return {x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]};
}

void operator+=(float3 & x, const float3 & y) {
  x[0] += y[0];
  x[1] += y[1];
  x[2] += y[2];
}

float3 operator*(const float & scale, const float3 & y) {
  return {scale * y[0], scale * y[1], scale * y[2]};
}

float4x3 operator*(const float & scale, const float4x3 & A) {
  return {scale * A[0], scale * A[1], scale * A[2], scale * A[3]};
}

float3 operator/(const float3 & x, const float & scale) {
  return {x[0] / scale, x[1] / scale, x[2] / scale};
}

float4x3 operator/(const float4x3 & A, const float & scale) {
  return {A[0] / scale, A[1] / scale, A[2] / scale, A[3] / scale};
}
 
float dot(const float3 & x, const float3 & y) {
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

float det(const float3 & A0, const float3 & A1, const float3 & A2) {
  return A0[0] * A1[1] * A2[2] + A0[1] * A1[2] * A2[0] +
         A0[2] * A1[0] * A2[1] - A0[0] * A1[2] * A2[1] -
         A0[1] * A1[0] * A2[2] - A0[2] * A1[1] * A2[0];
}


std::ostream& operator<<(std::ostream & out, const float3 & x) {
  out << "{" << x[0] << " " << x[1] << " " << x[2] << "}";
  return out;
}