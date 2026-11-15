// Reuse the frozen R12 interval primitives without altering its source/hash.
// The renamed old entry point is unused. Both sources are recorded in BUILD.
#define main archived_r12_kernel_entrypoint
#include "rounding_adam2_endpoint.cpp"
#undef main

int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("Expected scatter_input.bin endpoint_intervals.bin");preflight();
 std::ifstream in(argv[1],std::ios::binary);std::uint64_t d=0,n=0;in.read(reinterpret_cast<char*>(&d),8);in.read(reinterpret_cast<char*>(&n),8);
 if(!d||d>1024||n<=2*d||n>0x100000000ULL)throw std::runtime_error("Invalid dimensions");std::size_t q=2*d;
 std::vector<double>c(q*q),inverse(q*q),f(q*q),uhat(d*d);
 for(auto*v:{&c,&inverse,&f,&uhat})in.read(reinterpret_cast<char*>(v->data()),v->size()*8);
 if(!in)throw std::runtime_error("Truncated input");
 for(auto*v:{&c,&inverse,&f,&uhat})for(double x:*v)if(!std::isfinite(x)||std::fabs(x)>0x1p40)throw std::runtime_error("Magnitude cap");
 for(std::size_t i=0;i<q;i++){
  if(f[i*q+i]<=0.)throw std::runtime_error("Nonpositive diagonal");
  for(std::size_t j=0;j<q;j++)if(c[i*q+j]!=c[j*q+i]||(j>i&&f[i*q+j]!=0.))throw std::runtime_error("Invalid scatter/factor symmetry");
 }
 double scatter_bound=0.,inverse_max=0.;
 for(std::size_t i=0;i<q;i++){
  volatile double row=0.;
  for(std::size_t j=0;j<q;j++){
   I error=I(c[i*q+j])-dot(&f[i*q],&f[j*q],q);
   mode(FE_UPWARD);row=row+magnitude(error);
   I residual=dot(&f[i*q],&inverse[j],q,q)-I(i==j?1.:0.);
   inverse_max=std::max(inverse_max,magnitude(residual));
  }
  scatter_bound=std::max(scatter_bound,double(row));
 }
 std::vector<I>S(d*d),Z(d*d),u1(d*d),g(d*d);I sqrtn=root(I(double(n))),lr=rat(3,10),be1=rat(9,10),be2=rat(99,100);
 for(std::size_t i=0;i<d;i++)for(std::size_t j=0;j<d;j++){
  S[i*d+j]=I(c[i*q+j]);Z[i*d+j]=sqrtn*I(c[(i+d)*q+j]);u1[i*d+j]=lr*Z[i*d+j]/(absval(Z[i*d+j])+I(1.));
 }
 for(std::size_t i=0;i<d;i++)for(std::size_t j=0;j<d;j++){
  mode(FE_DOWNWARD);volatile double l=0.;for(std::size_t k=0;k<d;k++){double v=pmin(u1[i*d+k],S[k*d+j]);l=l+v;}
  mode(FE_UPWARD);volatile double u=0.;for(std::size_t k=0;k<d;k++){double v=pmax(u1[i*d+k],S[k*d+j]);u=u+v;}
  g[i*d+j]=I(l,u)-Z[i*d+j];
 }
 double error=0.;std::ofstream out(argv[2],std::ios::binary);out.write(reinterpret_cast<char*>(&d),8);
 for(std::size_t i=0;i<d*d;i++){
  I m=(g[i]-be1*Z[i])/(I(1.)+be1),v=(be2*square(Z[i])+square(g[i]))/(I(1.)+be2);
  I target=u1[i]-lr*m/(root(v)+I(1.));error=std::max(error,magnitude(target-I(uhat[i])));
  out.write(reinterpret_cast<char*>(&target.lo),8);out.write(reinterpret_cast<char*>(&target.hi),8);
 }
 if(!out)throw std::runtime_error("Cannot save intervals");mode(FE_TONEAREST);
 std::cout<<"rounding_checks true\n"<<"dimension "<<d<<"\n"<<"sample_size "<<n<<"\n"<<std::hexfloat
 <<"scatter_operator_error_hex "<<scatter_bound<<"\n"<<"inverse_max_residual_hex "<<inverse_max<<"\n"<<"max_endpoint_error_hex "<<error<<"\n";
 return 0;
}catch(const std::exception&e){std::fesetround(FE_TONEAREST);std::cerr<<e.what()<<"\n";return 1;}}
