// Directed binary64 dot products for a represented symmetric matrix.
// Compile with the exact flags recorded by certify_spectral_residual.py.
#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <xmmintrin.h>

static void rounding(int mode) {
    if (std::fesetround(mode) != 0 || std::fegetround() != mode)
        throw std::runtime_error("Cannot set requested rounding mode");
    if (_mm_getcsr() & 0x8040)
        throw std::runtime_error("FTZ or DAZ enabled");
}

static void preflight() {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    std::uint16_t endian = 1;
    if (*reinterpret_cast<unsigned char*>(&endian) != 1)
        throw std::runtime_error("Input format requires little endian");
    volatile double one=1., tiny=0x1p-54, negative=-1.;
    rounding(FE_DOWNWARD);
    volatile double down=one+tiny, negdown=negative-tiny;
    if (down != 1. || negdown != -0x1.0000000000001p+0)
        throw std::runtime_error("Downward addition check failed");
    rounding(FE_UPWARD);
    volatile double up=one+tiny, negup=negative-tiny;
    if (up != 0x1.0000000000001p+0 || negup != -1.)
        throw std::runtime_error("Upward addition check failed");
    volatile double a=1.+0x1p-27,b=1.-0x1p-27;
    rounding(FE_DOWNWARD); volatile double pd=a*b;
    rounding(FE_UPWARD); volatile double pu=a*b;
    if (pd != 0x1.fffffffffffffp-1 || pu != 1.)
        throw std::runtime_error("Directed product check failed");
    volatile double sub=std::numeric_limits<double>::denorm_min(),half=.5;
    rounding(FE_DOWNWARD); volatile double sd=sub*half;
    rounding(FE_UPWARD); volatile double su=sub*half;
    if (sd != 0. || su != std::numeric_limits<double>::denorm_min())
        throw std::runtime_error("Gradual underflow check failed");
    rounding(FE_TONEAREST);
}

int main(int argc,char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected binary matrix input");
        preflight();
        std::ifstream input(argv[1],std::ios::binary);
        std::uint64_t size=0;input.read(reinterpret_cast<char*>(&size),sizeof(size));
        if (size==0 || size>4096) throw std::runtime_error("Invalid dimension");
        const std::size_t n=size,entries=n*n;
        std::vector<double> h(entries),q(entries),lambda(n);
        input.read(reinterpret_cast<char*>(h.data()),entries*sizeof(double));
        input.read(reinterpret_cast<char*>(q.data()),entries*sizeof(double));
        input.read(reinterpret_cast<char*>(lambda.data()),n*sizeof(double));
        if (!input) throw std::runtime_error("Truncated input");
        for (double x:h) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite H");
        for (double x:q) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite Q");
        for (double x:lambda) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite eigenvalue");
        for (std::size_t i=0;i<n;i++) for (std::size_t j=0;j<n;j++)
            if (h[i*n+j] != h[j*n+i]) throw std::runtime_error("Represented matrix is not symmetric");
        auto start=std::chrono::steady_clock::now();
        std::vector<double> pup(entries),pdown(entries),glo(entries),rlo(entries);
        rounding(FE_UPWARD);
        for (std::size_t i=0;i<n;i++) for (std::size_t j=0;j<n;j++)
            pup[i*n+j]=q[i*n+j]*lambda[j];
        rounding(FE_DOWNWARD);
        for (std::size_t i=0;i<n;i++) for (std::size_t j=0;j<n;j++)
            pdown[i*n+j]=q[i*n+j]*lambda[j];
        for (std::size_t i=0;i<n;i++) for (std::size_t j=0;j<n;j++) {
            volatile double gram=0.,residual=0.;
            for (std::size_t k=0;k<n;k++) {
                volatile double gterm=q[k*n+i]*q[k*n+j];
                volatile double rterm=h[i*n+k]*q[k*n+j];
                gram=gram+gterm;residual=residual+rterm;
            }
            glo[i*n+j]=gram-(i==j ? 1. : 0.);
            rlo[i*n+j]=residual-pup[i*n+j];
        }
        rounding(FE_UPWARD);
        double maxg=0.,maxr=0.;
        for (std::size_t i=0;i<n;i++) for (std::size_t j=0;j<n;j++) {
            volatile double gram=0.,residual=0.;
            for (std::size_t k=0;k<n;k++) {
                volatile double gterm=q[k*n+i]*q[k*n+j];
                volatile double rterm=h[i*n+k]*q[k*n+j];
                gram=gram+gterm;residual=residual+rterm;
            }
            volatile double gu=gram-(i==j ? 1. : 0.);
            volatile double ru=residual-pdown[i*n+j];
            maxg=std::max(maxg,std::max(std::fabs(glo[i*n+j]),std::fabs(gu)));
            maxr=std::max(maxr,std::max(std::fabs(rlo[i*n+j]),std::fabs(ru)));
        }
        rounding(FE_TONEAREST);
        if (!std::isfinite(maxg) || !std::isfinite(maxr)) throw std::runtime_error("Nonfinite bound");
        std::cout << "rounding_checks true\n" << "dimension " << n << "\n"
                  << "max_gram_hex " << std::hexfloat << maxg << "\n"
                  << "max_residual_hex " << maxr << "\n" << std::defaultfloat
                  << "wall_seconds " << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << "\n";
        return 0;
    } catch(const std::exception& error) {
        std::fesetround(FE_TONEAREST);std::cerr << error.what() << "\n";return 1;
    }
}
