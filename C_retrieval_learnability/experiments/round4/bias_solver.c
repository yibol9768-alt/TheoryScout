/* Same bracketed Newton roots as Python reference, independently per row. */
#include <math.h>
static void eval(const double *x, double b, double *g, double *h) {
    *g=-3.; *h=0.;
    for(int j=0;j<16;j++) {
        double z=x[j]+b, e=exp(-fabs(z)), small=e/(1.+e);
        *g += z>=0 ? 1.-small : small;
        *h += small*(1.-small);
    }
}
int solve_rows(const double *base, const double *warm, int has_warm, int rows,
               double *out, double *maxres, long *iterations, long *fallbacks) {
    *maxres=0.; *iterations=0; *fallbacks=0;
    for(int i=0;i<rows;i++) {
        const double *x=base+16*i;
        double top[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
        double xmin=INFINITY,xmax=-INFINITY;
        for(int j=0;j<16;j++) {
            xmin=fmin(xmin,x[j]); xmax=fmax(xmax,x[j]);
            double v=x[j];
            for(int k=0;k<4;k++) if(v>top[k]) { double tmp=top[k]; top[k]=v; v=tmp; }
        }
        double lo=-xmax-40.,hi=-xmin+40.;
        double middle=-(top[2]+top[3])/2.;
        double b=has_warm ? fmax(lo,fmin(hi,warm[i])) : middle;
        double g,h,gm,hm; eval(x,b,&g,&h);
        if(has_warm && fabs(g)>5e-11) {
            eval(x,middle,&gm,&hm);
            if(fabs(gm)<fabs(g)) { b=middle; g=gm; h=hm; }
        }
        int it;
        for(it=0;it<80;it++) {
            if(fabs(g)<=5e-11) break;
            if(g<0) lo=b; else hi=b;
            double q=b-g/fmax(h,1e-300);
            if(it>=12 || q<=lo || q>=hi || !isfinite(q)) {
                q=(lo+hi)/2.; (*fallbacks)++;
            }
            b=q; eval(x,b,&g,&h);
        }
        if(it==80) return i+1;
        out[i]=b; *maxres=fmax(*maxres,fabs(g)); *iterations+=it+1;
    }
    return 0;
}
