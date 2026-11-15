// Exact-factor Adam2 endpoint audit. All stored floats are exact inputs.
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
static void mode(int m){if(std::fesetround(m)||std::fegetround()!=m||(_mm_getcsr()&0x8040))throw std::runtime_error("Rounding unavailable");}
static void preflight(){
 static_assert(sizeof(double)==8&&std::numeric_limits<double>::is_iec559);
 volatile double one=1.,tiny=0x1p-54,neg=-1.,a=1.+0x1p-27,b=1.-0x1p-27,sub=std::numeric_limits<double>::denorm_min(),half=.5;
 mode(FE_DOWNWARD);volatile double dl=one+tiny,dn=neg-tiny,dp=a*b,ds=sub*half;
 mode(FE_UPWARD);volatile double ul=one+tiny,un=neg-tiny,up=a*b,us=sub*half;
 if(dl!=1.||dn!=-0x1.0000000000001p+0||ul!=0x1.0000000000001p+0||un!=-1.||dp!=0x1.fffffffffffffp-1||up!=1.||ds!=0.||us!=sub)throw std::runtime_error("Rounding preflight failed");
 mode(FE_TONEAREST);
}
struct I {double lo,hi;I(double a=0.):lo(a),hi(a){} I(double a,double b):lo(a),hi(b){if(!std::isfinite(a)||!std::isfinite(b)||a>b)throw std::runtime_error("Invalid interval");}};
static I operator+(I a,I b){mode(FE_DOWNWARD);volatile double l=a.lo+b.lo;mode(FE_UPWARD);volatile double u=a.hi+b.hi;return I(l,u);}
static I operator-(I a,I b){mode(FE_DOWNWARD);volatile double l=a.lo-b.hi;mode(FE_UPWARD);volatile double u=a.hi-b.lo;return I(l,u);}
static double pmin(I a,I b){volatile double aa=a.lo*b.lo,ab=a.lo*b.hi,ba=a.hi*b.lo,bb=a.hi*b.hi;return std::min(std::min(aa,ab),std::min(ba,bb));}
static double pmax(I a,I b){volatile double aa=a.lo*b.lo,ab=a.lo*b.hi,ba=a.hi*b.lo,bb=a.hi*b.hi;return std::max(std::max(aa,ab),std::max(ba,bb));}
static I operator*(I a,I b){mode(FE_DOWNWARD);double l=pmin(a,b);mode(FE_UPWARD);double u=pmax(a,b);return I(l,u);}
static I operator/(I a,I b){if(b.lo<=0.)throw std::runtime_error("Nonpositive divisor");mode(FE_DOWNWARD);volatile double l=1./b.hi;mode(FE_UPWARD);volatile double u=1./b.lo;return a*I(l,u);}
static I rat(int a,int b){return I(double(a))/I(double(b));}
static I absval(I a){return a.lo>=0.?a:(a.hi<=0.?I(-a.hi,-a.lo):I(0.,std::max(-a.lo,a.hi)));}
static I square(I a){I b=absval(a);mode(FE_DOWNWARD);volatile double l=b.lo*b.lo;mode(FE_UPWARD);volatile double u=b.hi*b.hi;return I(l,u);}
// sqrt is only an initial guess: directed square comparisons prove endpoints.
static double rootlo(double x){if(x==0.)return 0.;mode(FE_TONEAREST);double s=std::sqrt(x);mode(FE_UPWARD);volatile double v=s*s;while(v>x){s=std::nextafter(s,0.);v=s*s;}return s;}
static double roothi(double x){if(x==0.)return 0.;mode(FE_TONEAREST);double s=std::sqrt(x);mode(FE_DOWNWARD);volatile double v=s*s;while(v<x){s=std::nextafter(s,std::numeric_limits<double>::infinity());v=s*s;}return s;}
static I root(I a){if(a.lo<0.)throw std::runtime_error("Negative radicand");return I(rootlo(a.lo),roothi(a.hi));}
static double magnitude(I a){return std::max(std::fabs(a.lo),std::fabs(a.hi));}
static I dot(const double*a,const double*b,std::size_t n,std::size_t stride=1){
 mode(FE_DOWNWARD);volatile double l=0.;for(std::size_t k=0;k<n;k++){volatile double p=a[k]*b[k*stride];l=l+p;}
 mode(FE_UPWARD);volatile double u=0.;for(std::size_t k=0;k<n;k++){volatile double p=a[k]*b[k*stride];u=u+p;}return I(l,u);
}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("Expected input.bin endpoint_intervals.bin");preflight();
 std::ifstream in(argv[1],std::ios::binary);std::uint64_t d=0,n=0;in.read(reinterpret_cast<char*>(&d),8);in.read(reinterpret_cast<char*>(&n),8);
 if(!d||d>1024||n<=2*d||n>0x100000000ULL)throw std::runtime_error("Invalid dimensions");const std::size_t q=2*d;
 std::vector<double>f(q*q),uhat(d*d),left(q*d),bhat(q*q);double scale;
 for(auto*v:{&f,&uhat,&left,&bhat})in.read(reinterpret_cast<char*>(v->data()),v->size()*8);in.read(reinterpret_cast<char*>(&scale),8);
 if(!in||!std::isfinite(scale)||scale<=0.||scale>0x1p40)throw std::runtime_error("Invalid input/scale");
 for(auto*v:{&f,&uhat,&left,&bhat})for(double z:*v)if(!std::isfinite(z)||std::fabs(z)>0x1p40)throw std::runtime_error("Input magnitude cap");
 for(std::size_t i=0;i<q;i++){if(f[i*q+i]<=0.)throw std::runtime_error("Nonpositive factor diagonal");for(std::size_t j=i+1;j<q;j++)if(f[i*q+j]!=0.)throw std::runtime_error("Factor not triangular");}
 std::vector<I>S(d*d),Z(d*d),u1(d*d),g(d*d),u2(d*d);I sqrtn=root(I(double(n))),lr=rat(3,10),be1=rat(9,10),be2=rat(99,100);
 for(std::size_t i=0;i<d;i++)for(std::size_t j=0;j<d;j++){
  S[i*d+j]=dot(&f[i*q],&f[j*q],q);Z[i*d+j]=sqrtn*dot(&f[(i+d)*q],&f[j*q],q);
  u1[i*d+j]=lr*Z[i*d+j]/(absval(Z[i*d+j])+I(1.));
 }
 for(std::size_t i=0;i<d;i++)for(std::size_t j=0;j<d;j++){
  mode(FE_DOWNWARD);volatile double l=0.;for(std::size_t k=0;k<d;k++){double v=pmin(u1[i*d+k],S[k*d+j]);l=l+v;}
  mode(FE_UPWARD);volatile double u=0.;for(std::size_t k=0;k<d;k++){double v=pmax(u1[i*d+k],S[k*d+j]);u=u+v;}
  g[i*d+j]=I(l,u)-Z[i*d+j];
 }
 double erru=0.,errleft=0.,errb=0.;
 std::ofstream out(argv[2],std::ios::binary);out.write(reinterpret_cast<char*>(&d),8);
 for(std::size_t i=0;i<d*d;i++){
  I m=(g[i]-be1*Z[i])/(I(1.)+be1);
  I v=(be2*square(Z[i])+square(g[i]))/(I(1.)+be2);
  u2[i]=u1[i]-lr*m/(root(v)+I(1.));erru=std::max(erru,magnitude(u2[i]-I(uhat[i])));
  out.write(reinterpret_cast<char*>(&u2[i].lo),8);out.write(reinterpret_cast<char*>(&u2[i].hi),8);
 }
 for(std::size_t i=0;i<q;i++)for(std::size_t j=0;j<d;j++){
  // Fy is the bottom block of f, so gather its i-th column.
  mode(FE_DOWNWARD);volatile double l=0.;for(std::size_t k=0;k<d;k++){volatile double v=f[(d+k)*q+i]*uhat[k*d+j];l=l+v;}
  mode(FE_UPWARD);volatile double u=0.;for(std::size_t k=0;k<d;k++){volatile double v=f[(d+k)*q+i]*uhat[k*d+j];u=u+v;}
  errleft=std::max(errleft,magnitude(I(l,u)-I(left[i*d+j])));
 }
 for(std::size_t i=0;i<q;i++)for(std::size_t j=0;j<q;j++){
  I value=dot(&left[i*d],&f[j],d,q)-I(scale)*I(bhat[i*q+j]);errb=std::max(errb,magnitude(value));
 }
 if(!out)throw std::runtime_error("Cannot write endpoint intervals");mode(FE_TONEAREST);
 std::cout<<"rounding_checks true\n"<<"dimension "<<d<<"\n"<<"sample_size "<<n<<"\n"<<std::hexfloat
 <<"max_endpoint_error_hex "<<erru<<"\n"<<"max_left_residual_hex "<<errleft<<"\n"<<"max_block_residual_hex "<<errb<<"\n";
 return 0;
}catch(const std::exception&e){std::fesetround(FE_TONEAREST);std::cerr<<e.what()<<"\n";return 1;}}
