// Bound H_exact=A sym(B)-D B^T B relative to a represented symmetric H.
#include <algorithm>
#include <cfenv>
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
    if (std::fesetround(mode) || std::fegetround()!=mode || (_mm_getcsr() & 0x8040))
        throw std::runtime_error("Directed rounding/gradual underflow unavailable");
}
static double product_min(double a,double b,double c,double d) {
    volatile double ac=a*c,ad=a*d,bc=b*c,bd=b*d;
    return std::min(std::min(ac,ad),std::min(bc,bd));
}
static double product_max(double a,double b,double c,double d) {
    volatile double ac=a*c,ad=a*d,bc=b*c,bd=b*d;
    return std::max(std::max(ac,ad),std::max(bc,bd));
}
int main(int argc,char**argv) {
    try {
        static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
        if(argc!=2) throw std::runtime_error("Expected binary construction input");
        volatile double one=1.,tiny=0x1p-54,neg=-1.;
        rounding(FE_DOWNWARD);volatile double a=one+tiny,b=neg-tiny;
        rounding(FE_UPWARD);volatile double c=one+tiny,d=neg-tiny;
        if(a!=1. || b!=-0x1.0000000000001p+0 || c!=0x1.0000000000001p+0 || d!=-1.)
            throw std::runtime_error("Addition preflight failed");
        volatile double x=1.+0x1p-27,y=1.-0x1p-27;
        rounding(FE_DOWNWARD);volatile double pd=x*y;
        rounding(FE_UPWARD);volatile double pu=x*y;
        volatile double sub=std::numeric_limits<double>::denorm_min(),half=.5;
        rounding(FE_DOWNWARD);volatile double sd=sub*half;
        rounding(FE_UPWARD);volatile double su=sub*half;
        if(pd!=0x1.fffffffffffffp-1 || pu!=1. || sd!=0. || su!=sub)
            throw std::runtime_error("Product/underflow preflight failed");
        std::ifstream in(argv[1],std::ios::binary);std::uint64_t n=0;
        in.read(reinterpret_cast<char*>(&n),8);
        if(!n || n>4096) throw std::runtime_error("Invalid dimension");
        std::vector<double>B(n*n),H(n*n),nl(n*n),nu(n*n),kl(n*n),ku(n*n);
        double v[4];in.read(reinterpret_cast<char*>(B.data()),n*n*8);
        in.read(reinterpret_cast<char*>(H.data()),n*n*8);in.read(reinterpret_cast<char*>(v),32);
        if(!in) throw std::runtime_error("Truncated input");
        for(double z:B)if(!std::isfinite(z))throw std::runtime_error("Nonfinite B");
        for(double z:H)if(!std::isfinite(z))throw std::runtime_error("Nonfinite H");
        for(double z:v)if(!std::isfinite(z))throw std::runtime_error("Nonfinite scalar");
        if(v[0]>v[1] || v[2]>v[3])throw std::runtime_error("Unordered scalar bounds");
        for(std::size_t i=0;i<n;i++)for(std::size_t j=0;j<n;j++)
            if(H[i*n+j]!=H[j*n+i])throw std::runtime_error("H is not symmetric");
        rounding(FE_DOWNWARD);
        for(std::size_t i=0;i<n;i++)for(std::size_t j=0;j<n;j++) {
            volatile double sum=0.;
            for(std::size_t k=0;k<n;k++) { volatile double z=B[k*n+i]*B[k*n+j];sum=sum+z; }
            nl[i*n+j]=sum;
            volatile double z=B[i*n+j]+B[j*n+i];kl[i*n+j]=z*.5;
        }
        rounding(FE_UPWARD);
        for(std::size_t i=0;i<n;i++)for(std::size_t j=0;j<n;j++) {
            volatile double sum=0.;
            for(std::size_t k=0;k<n;k++) { volatile double z=B[k*n+i]*B[k*n+j];sum=sum+z; }
            nu[i*n+j]=sum;
            volatile double z=B[i*n+j]+B[j*n+i];ku[i*n+j]=z*.5;
        }
        double maximum=0.;
        for(std::size_t i=0;i<n*n;i++) {
            rounding(FE_DOWNWARD);double akl=product_min(v[0],v[1],kl[i],ku[i]);
            double dnl=product_min(v[2],v[3],nl[i],nu[i]);
            rounding(FE_UPWARD);double aku=product_max(v[0],v[1],kl[i],ku[i]);
            double dnu=product_max(v[2],v[3],nl[i],nu[i]);
            volatile double hu=aku-dnl;volatile double eu=hu-H[i];
            rounding(FE_DOWNWARD);volatile double hl=akl-dnu;volatile double el=hl-H[i];
            maximum=std::max(maximum,std::max(std::fabs(el),std::fabs(eu)));
        }
        rounding(FE_TONEAREST);
        if(!std::isfinite(maximum))throw std::runtime_error("Nonfinite construction bound");
        std::cout<<"rounding_checks true\n"<<"dimension "<<n<<"\n"
                 <<"max_entry_error_hex "<<std::hexfloat<<maximum<<"\n";
    }catch(const std::exception&e){std::fesetround(FE_TONEAREST);std::cerr<<e.what()<<"\n";return 1;}
}
