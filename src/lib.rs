extern crate nalgebra as na;
use na::{Vector3, Matrix3};
use std::time::{Instant};

static P3P_CUBIC_SOLVER_ITER : i32 = 500;
static P3P_GAUSS_NEWTON_ITER : i32 = 5;

#[allow(non_snake_case)]
fn eigwithknown0(x : &Matrix3<f64>) -> (Vec<f64>, Vec<Vector3<f64>>) {
    let mut L = Vec::new();

    let mut v3 = Vector3::new(
        x[1]*x[5] - x[2]*x[4], 
        x[2]*x[3] - x[5]*x[0], 
        x[4]*x[0] - x[1]*x[3]
    );
    v3.normalize_mut();

    let x01_squared= x[(0,1)] * x[(0,1)];
    let b= -x[(0,0)] - x[(1,1)] - x[(2,2)];
    let c= -x01_squared - x[(0,2)]*x[(0,2)] - x[(1,2)]*x[(1,2)] +
                x[(0,0)]*(x[(1,1)] + x[(2,2)]) + x[(1,1)]*x[(2,2)];
    let mut e1 = 0.;
    let mut e2 = 0.;
    root2real(b,c,&mut e1,&mut e2);

    if e1.abs() < e2.abs() {
        L.push(e2);
        L.push(e1);
    } else {
        L.push(e1);
        L.push(e2);
    }

    let mx0011= -x[(0,0)]*x[(1,1)];
    let prec_0 = x[(0,1)]*x[(1,2)] - x[(0,2)]*x[(1,1)];
    let prec_1 = x[(0,1)]*x[(0,2)] - x[(0,0)]*x[(1,2)];


    let e=e1;
    let tmp= 1.0/(e*(x[(0,0)] + x[(1,1)]) + mx0011 - e*e + x01_squared);
    let mut a1= -(e*x[(0,2)] + prec_0)*tmp;
    let mut a2= -(e*x[(1,2)] + prec_1)*tmp;
    let rnorm= (1.) / (a1*a1 +a2*a2 + 1.).sqrt();
    a1*=rnorm;
    a2*=rnorm;
    let v1 = Vector3::new(a1,a2,rnorm);

    let tmp2=1.0/(e2*(x[(0,0)] + x[(1,1)]) + mx0011 - e2*e2 + x01_squared);
    let mut a21= -(e2*x[(0,2)] + prec_0)*tmp2;
    let mut a22= -(e2*x[(1,2)] + prec_1)*tmp2;
    let rnorm2=1.0/(a21*a21 +a22*a22 +1.0).sqrt();
    a21*=rnorm2;
    a22*=rnorm2;
    let v2 = Vector3::new(a21,a22,rnorm2);

    (L, vec![v1, v2])
}

#[allow(non_snake_case)]
fn gauss_newton_refineL(L: &mut Vector3<f64>,
    a12: f64, a13: f64, a23: f64,
    b12: f64, b13: f64, b23: f64, iterations: i32) {

    for _ in 0..iterations {
        let l1= L[0];
        let l2= L[1];
        let l3= L[2];
        let r1= l1*l1 + l2*l2 + b12*l1*l2 - a12;
        let r2= l1*l1 + l3*l3 + b13*l1*l3 - a13;
        let r3= l2*l2 + l3*l3 + b23*l2*l3 - a23;

        if r1.abs() + r2.abs() + r3.abs() < 1e-10 {
            break;
        }

        let dr1dl1=(2.0)*l1 +b12*l2;
        let dr1dl2=(2.0)*l2 +b12*l1;

        let dr2dl1=(2.0)*l1 +b13*l3;
        let dr2dl3=(2.0)*l3 +b13*l1;

        let dr3dl2=(2.0)*l2 + b23*l3;
        let dr3dl3=(2.0)*l3 + b23*l2;

        let r = Vector3::new(r1, r2, r3);

        let v0=dr1dl1;
        let v1=dr1dl2;
        let v3=dr2dl1;
        let v5=dr2dl3;
        let v7=dr3dl2;
        let v8=dr3dl3;
        let det=(1.0)/(-v0*v5*v7 - v1*v3*v8);

        let Ji: Matrix3<f64> = Matrix3::new( 
            -v5*v7, -v1*v8,  v1*v5,
            -v3*v8,  v0*v8, -v0*v5,
             v3*v7, -v0*v7, -v1*v3
        );
        let L1: Vector3<f64> = *L - det * Ji * r;
        let l1=L1[0];
        let l2=L1[1];
        let l3=L1[2];
        let r11=l1*l1 + l2*l2 +b12*l1*l2 -a12;
        let r12=l1*l1 + l3*l3 +b13*l1*l3 -a13;
        let r13=l2*l2 + l3*l3 +b23*l2*l3 -a23;
        
        if r11.abs() + r12.abs() + r13.abs() > r1.abs() + r2.abs() + r3.abs(){
            break;
        }
        *L=L1;
    }
}

pub fn root2real(b: f64, c: f64, r1: &mut f64, r2: &mut f64) -> bool{
    let v= b*b -4.*c;
    if v <= 0. {
        *r1 = -0.5 * b;
        *r2 = *r1;
        return v.abs() < 1e-10;
    }

    let y = v.sqrt();
    if b < 0. {
        *r1 = 0.5*(-b+y);
        *r2 = 0.5*(-b-y);
    } else {
        *r1 = 2.0*c/(-b+y);
        *r2 = 2.0*c/(-b-y);
    }
    return true;
}

fn cubick(b: f64, c: f64, d : f64) -> f64 {
    /* Choose initial solution */
    let mut r0;
    // not monotonic
    if b*b  >= 3. * c {
        // h has two stationary points, compute them
        let v= (b*b - 3. * c).sqrt();
        let t1 = (-b - v)/3.;

        // Check if h(t1) > 0, in this case make a 2-order approx of h around t1
        let mut k = ((t1 + b) * t1 + c) * t1 + d;

        if k > 0.0 {
            //Find leftmost root of 0.5*(r0 -t1)^2*(6*t1+2*b) +  k = 0
            r0 = t1 - (-k / (3. * t1 + b)).sqrt();
        } else {
            let t2 = (-b + v) / 3.;
            k = ((t2+b)*t2+c)*t2+d;
            //Find rightmost root of 0.5*(r0 -t2)^2*(6*t2+2*b) +  k1 = 0
            r0 = t2 + (-k / (3. * t2 + b)).sqrt();
        }
    }
    else{
        r0 = -b / 3.;
        if (((3.*r0+2.*b)*r0+c)).abs() < 1e-4 {
            r0 += 1.;
        }
    }

    /* Do ITER Newton-Raphson iterations */
    /* Break if position of root changes less than 1e.16 */
    //T starterr=std::abs(r0*(r0*(r0 + b) + c) + d);
    let mut fx;
    let mut fpx;

    for _ in 0..P3P_CUBIC_SOLVER_ITER {
        fx = ((r0 + b) * r0 + c) * r0 + d;

        if fx.abs() < 1e-4 {
            break;
        }
        
        fpx = (3. * r0 + 2. * b) * r0+c;

        r0 -= fx/fpx;
    }
    return r0;
}

#[allow(non_snake_case)]
pub fn p3p(ys: &Vec<Vector3<f64>>, xs: &Vec<Vector3<f64>>) -> Vec<(Vector3<f64>, Matrix3<f64>)> {
    let start = Instant::now();
    let ys : Vec<Vector3<f64>> = ys.iter().map(|y| y.normalize()).collect();

    let b12 = -2. * ys[0].dot(&ys[1]);
    let b13 = -2. * ys[0].dot(&ys[2]);
    let b23 = -2. * ys[1].dot(&ys[2]);

    let d12 = xs[0] - xs[1];
    let d13 = xs[0] - xs[2];
    let d23 = xs[1] - xs[2];
    let d12xd13 = d12.cross(&d13);

    let a12 = d12.norm_squared();
    let a13 = d13.norm_squared();
    let a23 = d23.norm_squared();

    let c31 = -0.5 * b13;
    let c23 = -0.5 * b23;
    let c12 = -0.5 * b12;
    let blob = c12*c23*c31 - 1.;

    let s31_squared = 1. - c31.powi(2);
    let s23_squared = 1. - c23.powi(2);
    let s12_squared = 1. - c12.powi(2);

    let mut p3 = a13 * (a23 * s31_squared - a13 * s23_squared);
    let mut p2 = 2. * blob * a23 * a13 
                + a13 * (2. * a12 + a13) * s23_squared 
                + a23 * (a23 - a12) *s31_squared;
    let mut p1 = a23 * (a13 - a23) * s12_squared 
                - a12 * a12  *s23_squared - 2. * a12 * (blob * a23 + a13 * s23_squared);
    let mut p0 = a12 * (a12 * s23_squared - a23 * s12_squared);


    p3 = 1./p3;
    p2 *= p3;
    p1 *= p3;
    p0 *= p3;

    let g = cubick(p2, p1, p0);

    let A00= a23*(1.0- g);
    let A01= (a23*b12)*0.5;
    let A02= (a23*b13*g)*(-0.5);
    let A11= a23 - a12 + a13*g;
    let A12= b23*(a13*g - a12)*0.5;
    let A22= g*(a13 - a23) - a12;

    let A : Matrix3<f64> = Matrix3::new(
        A00, A01, A02, 
        A01, A11, A12,
        A02, A12, A22
    );

    let evs = eigwithknown0(&A);
    let es = evs.0;
    let vs = evs.1;
    

    let v = 0f64.max(-es[1] / es[0]).sqrt();

    let mut valid = 0;
    let mut Ls = Vec::new();

    let v0 = vs[0];
    let v1 = vs[1];

    for s in [v, -v] {
        let u1= v0[0] - s*v1[0];
        let u2= v0[1] - s*v1[1];
        let u3= v0[2] - s*v1[2];

        if u1.abs() < u2.abs() {
            // solve for l2
            let a= (a23 - a12)*u3*u3 - a12*u2*u2 + a12*b23*u2*u3;
            let b= (2.*a23*u1*u3 - 2.*a12*u1*u3 + a12*b23*u1*u2 - a23*b12*u2*u3)/a;
            let c= (a23*u1*u1 - a12*u1*u1 + a23*u2*u2 - a23*b12*u1*u2)/a;

            let mut r1 = 0.;
            let mut r2 = 0.;
            if !root2real(b, c, &mut r1, &mut r2) {
                continue;
            }
            for tau in [r1, r2] {
                if tau <= 0. {
                    continue;
                }
                let l1= (a13/(tau*(tau + b13) + 1.)).sqrt();
                let l3= tau * l1;
                let l2= -(u1*l1 + u3*l3)/u2;
                if l2 <= 0. {
                    continue;
                }
                Ls.push(Vector3::new(l1, l2, l3));
                valid += 1;
            }
        } else { 
            let w2= 1./(-u1);
            let w0= u2*w2;
            let w1= u3*w2;

            let a= 1./((a13 - a12)*w1*w1 - a12*b13*w1 - a12);
            let b= (a13*b12*w1 - a12*b13*w0 - 2.*w0*w1*(a12 - a13))*a;
            let c= ((a13 - a12)*w0*w0 + a13*b12*w0 + a13)*a;

            let mut r1 = 0.;
            let mut r2 = 0.;
            if !root2real(b, c, &mut r1, &mut r2) {
                continue;
            }
            for tau in [r1, r2] {
                if tau <= 0. {
                    continue;
                }
                let d= a23/(tau*(b23 + tau) + 1.);
                let l2= d.sqrt();
                let l3= tau * l2;
                let l1= w0*l2 +w1*l3;
                if l1 <= 0. {
                    continue;
                }
                Ls.push(Vector3::new(l1, l2, l3));
                valid += 1;
            }
        }
    }

    for i in 0..valid {
        gauss_newton_refineL(&mut Ls[i], a12, a13, a23, b12, b13, b23, P3P_GAUSS_NEWTON_ITER);
    }
    let mut ry1 ; 
    let mut ry2 ;
    let mut ry3 ;
    let mut yd1 ;
    let mut yd2 ;
    let mut yd1xd2 ;
    let mut X = Matrix3::new(
        d12[0],d13[0],d12xd13[0],
        d12[1], d13[1],d12xd13[1],
        d12[2], d13[2], d12xd13[2]
    );
    X = X.try_inverse().unwrap();

    let mut results = Vec::new();
    for i in 0..valid {
        ry1 = ys[0] * Ls[i][0];
        ry2 = ys[1] * Ls[i][1];
        ry3 = ys[2] * Ls[i][2];

        yd1 = ry1-ry2;
        yd2 = ry1-ry3;
        yd1xd2 = yd1.cross(&yd2);

        let Y = Matrix3::new(
            yd1[0],yd2[0],yd1xd2[0],
            yd1[1],yd2[1],yd1xd2[1],
            yd1[2],yd2[2],yd1xd2[2]
        );

        let R = Y * X;
        let t = ry1 - R * xs[0];
        results.push((
            t,
            R
        ));
    }
    let end = Instant::now();
    println!("{}", (end - start).as_nanos());
    results
}
