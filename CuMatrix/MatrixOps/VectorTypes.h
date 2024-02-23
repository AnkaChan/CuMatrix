#pragma once
#include "CuMatrixDefs.h"
#include <vector_types.h>

typedef unsigned char uchar;
typedef unsigned short ushort;

template<typename T> struct vector_helper { };
template<> struct vector_helper<uchar> { typedef float  ftype; typedef int  itype; };
template<> struct vector_helper<uchar2> { typedef float2 ftype; typedef int2 itype; };
template<> struct vector_helper<uchar4> { typedef float4 ftype; typedef int4 itype; };
template<> struct vector_helper<ushort> { typedef float  ftype; typedef int  itype; };
template<> struct vector_helper<ushort2> { typedef float2 ftype; typedef int2 itype; };
template<> struct vector_helper<ushort4> { typedef float4 ftype; typedef int4 itype; };
template<> struct vector_helper<int> { typedef float  ftype; typedef int  itype; };
template<> struct vector_helper<int2> { typedef float2 ftype; typedef int2 itype; };
template<> struct vector_helper<int4> { typedef float4 ftype; typedef int4 itype; };

#define floatT typename vector_helper<T>::ftype
#define intT typename vector_helper<T>::itype

template<typename T, typename V> inline __device__ V to_floatN(const T& a) { return (V)a; }
template<typename T, typename V> inline __device__ T from_floatN(const V& a) { return (T)a; }

// arithmetic operators fo the built-in vector types
#define OPERATORS2(T) \
    template<typename V> DEVICE_INLINE_FUNC T operator+(const T &a, const V &b) { return make_ ## T (a.x + b.x, a.y + b.y); } \
    template<typename V> DEVICE_INLINE_FUNC T operator-(const T &a, const V &b) { return make_ ## T (a.x - b.x, a.y - b.y); } \
    template<typename V> DEVICE_INLINE_FUNC T operator*(const T &a, V b) { return make_ ## T (a.x * b, a.y * b); } \
    template<typename V> DEVICE_INLINE_FUNC T operator/(const T &a, V b) { return make_ ## T (a.x / b, a.y / b); } \
    template<typename V> DEVICE_INLINE_FUNC T operator>>(const T &a, V b) { return make_ ## T (a.x >> b, a.y >> b); } \
    template<typename V> DEVICE_INLINE_FUNC T operator<<(const T &a, V b) { return make_ ## T (a.x << b, a.y << b); } \
    template<typename V> DEVICE_INLINE_FUNC T &operator+=(T &a, const V &b) { a.x += b.x; a.y += b.y; return a; } \
    template<typename V> DEVICE_INLINE_FUNC void vec_set(T &a, const V &b) { a.x = b.x; a.y = b.y; } \
    template<typename V> DEVICE_INLINE_FUNC void vec_set_scalar(T &a, V b) { a.x = b; a.y = b; } \
    template<> DEVICE_INLINE_FUNC float2 to_floatN<T, float2>(const T &a) { return make_float2(a.x, a.y); } \
    template<> DEVICE_INLINE_FUNC T from_floatN<T, float2>(const float2 &a) { return make_ ## T(a.x, a.y); }

#define OPERATORS4(T) \
    template<typename V> DEVICE_INLINE_FUNC T operator+(const T &a, const V &b) { return make_ ## T (a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); } \
    template<typename V> DEVICE_INLINE_FUNC T operator-(const T &a, const V &b) { return make_ ## T (a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); } \
    template<typename V> DEVICE_INLINE_FUNC T operator*(const T &a, V b) { return make_ ## T (a.x * b, a.y * b, a.z * b, a.w * b); } \
    template<typename V> DEVICE_INLINE_FUNC T operator/(const T &a, V b) { return make_ ## T (a.x / b, a.y / b, a.z / b, a.w / b); } \
    template<typename V> DEVICE_INLINE_FUNC T operator>>(const T &a, V b) { return make_ ## T (a.x >> b, a.y >> b, a.z >> b, a.w >> b); } \
    template<typename V> DEVICE_INLINE_FUNC T operator<<(const T &a, V b) { return make_ ## T (a.x << b, a.y << b, a.z << b, a.w << b); } \
    template<typename V> DEVICE_INLINE_FUNC T &operator+=(T &a, const V &b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; } \
    template<typename V> DEVICE_INLINE_FUNC T &operator-=(T &a, const V &b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; } \
    template<typename V> DEVICE_INLINE_FUNC void vec_set(T &a, const V &b) { a.x = b.x; a.y = b.y; a.z = b.z; a.w = b.w; } \
    template<typename V> DEVICE_INLINE_FUNC void vec_set_scalar(T &a, V b) { a.x = b; a.y = b; a.z = b; a.w = b; } \
    template<> DEVICE_INLINE_FUNC float4 to_floatN<T, float4>(const T &a) { return make_float4(a.x, a.y, a.z, a.w); } \
    template<> DEVICE_INLINE_FUNC T from_floatN<T, float4>(const float4 &a) { return make_ ## T(a.x, a.y, a.z, a.w); } \
    DEVICE_INLINE_FUNC float normSquare(const T& a) { return SQR(a.x) + SQR(a.y) + SQR(a.z) + SQR(a.w); } \
    DEVICE_INLINE_FUNC float norm(const T& a) { return sqrtf(normSquare(a)); } \
    DEVICE_INLINE_FUNC float normVec3Square(const T& a) { return SQR(a.x) + SQR(a.y) + SQR(a.z); } \
    DEVICE_INLINE_FUNC float normVec3(const T& a) { return sqrtf(normVec3Square(a)); } \
    DEVICE_INLINE_FUNC float disVec3Square(const T& a, const T &b) { return normVec3Square(a - b); } \
    DEVICE_INLINE_FUNC float disVec3(const T& a, const T &b) { return sqrtf(disVec3Square(a, b)); } 

OPERATORS2(int2)
OPERATORS2(uchar2)
OPERATORS2(ushort2)
OPERATORS2(float2)
OPERATORS4(int4)
OPERATORS4(uchar4)
OPERATORS4(ushort4)
OPERATORS4(float4)

template<typename V> DEVICE_INLINE_FUNC void vec_set(int& a, V b) { a = b; }
template<typename V> DEVICE_INLINE_FUNC void vec_set(float& a, V b) { a = b; }
template<typename V> DEVICE_INLINE_FUNC void vec_set(uchar& a, V b) { a = b; }
template<typename V> DEVICE_INLINE_FUNC void vec_set(ushort& a, V b) { a = b; }
template<typename V> DEVICE_INLINE_FUNC void vec_set_scalar(int& a, V b) { a = b; }
template<typename V> DEVICE_INLINE_FUNC void vec_set_scalar(float& a, V b) { a = b; }
template<typename V> DEVICE_INLINE_FUNC void vec_set_scalar(uchar& a, V b) { a = b; }
template<typename V> DEVICE_INLINE_FUNC void vec_set_scalar(ushort& a, V b) { a = b; }

template<typename T>
inline __device__ T lerp_scalar(T v0, T v1, float t) {
    return t * v1 + (1.0f - t) * v0;
}

template<>
inline __device__ float2 lerp_scalar<float2>(float2 v0, float2 v1, float t) {
    return make_float2(
        lerp_scalar(v0.x, v1.x, t),
        lerp_scalar(v0.y, v1.y, t)
    );
}

template<>
inline __device__ float4 lerp_scalar<float4>(float4 v0, float4 v1, float t) {
    return make_float4(
        lerp_scalar(v0.x, v1.x, t),
        lerp_scalar(v0.y, v1.y, t),
        lerp_scalar(v0.z, v1.z, t),
        lerp_scalar(v0.w, v1.w, t)
    );
}

namespace CuMatrix {

    namespace detail
    {
        template<typename T> struct vector_of;
        template<> struct vector_of<float> { using type = float4; };
        template<> struct vector_of<double> { using type = double4; };
    }
    template<typename T>
    using vector_of_t = typename detail::vector_of<T>::type;

    // 16 bytes aligned 3d vector, with 4th component for padding; 
    // the 4th component is not used in the vector operations, but it can be customized for other purposes
    template<typename DType>
    struct  Vec3a
    {
        HOST_DEVICE_INLINE_FUNC Vec3a(const Vec3a<DType>& v) {
            d.x = v.x();
            d.y = v.y();
            d.z = v.z();
        }
        HOST_DEVICE_INLINE_FUNC Vec3a(const DType* data) {
            d.x = data[0];
            d.y = data[1];
            d.z = data[2];
        }
        HOST_DEVICE_INLINE_FUNC Vec3a(DType x, DType y, DType z) { d.x = x;  d.y = y; d.z = z; }
        HOST_DEVICE_INLINE_FUNC Vec3a(DType x, DType y, DType z, DType w) { d.x = x;  d.y = y; d.z = z; d.w = w; }
        HOST_DEVICE_INLINE_FUNC Vec3a(const vector_of_t<DType>& v) { d = v; }
        HOST_DEVICE_INLINE_FUNC Vec3a(DType val) { d.x = val;  d.y = val; d.z = val; }
        HOST_DEVICE_INLINE_FUNC Vec3a() {}

        HOST_DEVICE_INLINE_FUNC void set3(DType x, DType y, DType z) { d.x = x; d.y = y; d.z = z; }
        HOST_DEVICE_INLINE_FUNC void set3(const DType* data) { d.x = data[0]; d.y = data[1]; d.z = data[2]; }
        HOST_DEVICE_INLINE_FUNC void set3(const Vec3a<DType>& v) { d.x = v.x(); d.y = v.y(); d.z = v.z(); }

        HOST_DEVICE_INLINE_FUNC DType& x() { return d.x; }
        HOST_DEVICE_INLINE_FUNC DType& y() { return d.y; }
        HOST_DEVICE_INLINE_FUNC DType& z() { return d.z; }
        HOST_DEVICE_INLINE_FUNC DType& w() { return d.w; }

        HOST_DEVICE_INLINE_FUNC const DType& x() const { return d.x; }
        HOST_DEVICE_INLINE_FUNC const DType& y() const { return d.y; }
        HOST_DEVICE_INLINE_FUNC const DType& z() const { return d.z; }
        HOST_DEVICE_INLINE_FUNC const DType& w() const { return d.w; }

        HOST_DEVICE_INLINE_FUNC DType& operator[](int i) { return (&d.x)[i]; }
        const DType& operator[](int i) const { return (&d.x)[i]; }

        HOST_DEVICE_INLINE_FUNC Vec3a<DType> operator+(const Vec3a<DType>& v) const {
            return Vec3a<DType>(d.x + v.x(), d.y + v.y(), d.z + v.z());
        }
        HOST_DEVICE_INLINE_FUNC Vec3a<DType> operator-(const Vec3a<DType>& v) const
        {
            return Vec3a<DType>(d.x - v.x(), d.y - v.y(), d.z - v.z());
        }
        HOST_DEVICE_INLINE_FUNC Vec3a<DType> operator*(DType v) const
        {
            return Vec3a<DType>(d.x * v, d.y * v, d.z * v);
        }
        HOST_DEVICE_INLINE_FUNC Vec3a<DType> operator/(DType v) const {
            return Vec3a<DType>(d.x / v, d.y / v, d.z / v);
        }
        HOST_DEVICE_INLINE_FUNC Vec3a<DType>& operator+=(const Vec3a<DType>& v)
        {
            set3(d.x + v.x(), d.y + v.y(), d.z + v.z());
            return *this;
        }
        HOST_DEVICE_INLINE_FUNC Vec3a<DType>& operator-=(const Vec3a<DType>& v)
        {
            set3(d.x - v.x(), d.y - v.y(), d.z - v.z());
            return *this;
        }

        HOST_DEVICE_INLINE_FUNC DType normSquare() const { return SQR(d.x) + SQR(d.y) + SQR(d.z); }
        HOST_DEVICE_INLINE_FUNC DType norm() const { return sqrtf(normSquare()); }
        HOST_DEVICE_INLINE_FUNC DType disSquare(const Vec3a<DType>& v) const { return normSquare(v - *this); }
        HOST_DEVICE_INLINE_FUNC DType dis(const Vec3a<DType>& v) const { return sqrtf(disSquare(v)); }

        HOST_DEVICE_INLINE_FUNC DType dot(const Vec3a<DType>& v) const { return d.x * v.x() + d.y * v.y() + d.z * v.z(); }
        HOST_DEVICE_INLINE_FUNC Vec3a<DType> cross(const Vec3a<DType>& v) const
        {
            return Vec3a<DType>(d.y * v.z() - d.z * v.y(), d.z * v.x() - d.x * v.z(), d.x * v.y() - d.y * v.x());
        }


        HOST_DEVICE_INLINE_FUNC void setW(DType w) { d.w = w; }
        HOST_DEVICE_INLINE_FUNC void getW() { return d.w; };

        HOST_DEVICE_INLINE_FUNC const vector_of_t<DType>& getData() const { return d; }
        HOST_DEVICE_INLINE_FUNC const DType* getDataPtr() const { return &d.x; }

        HOST_DEVICE_INLINE_FUNC void print() const {
			printf("(%f, %f, %f)", d.x, d.y, d.z);
		}
    private:
        vector_of_t<DType> d;
    };

    typedef Vec3a<float> Vec3af;
    typedef Vec3a<double> Vec3ad;

}