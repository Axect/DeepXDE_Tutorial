use statrs::distribution::MultivariateNormal;
use rand::thread_rng;
use peroxide::fuga::*;

fn main() {
    let x = linspace(0, 1, 10);

    let mu = vec![0f64; x.len()];
    let cov = Matrix::from_index(|i, j| k(x[i], x[j]), (x.len(), x.len()));

    let mvn = MultivariateNormal::new(mu, cov.data.clone()).unwrap();
    
    let s = mvn.sample_iter(thread_rng())
        .map(|v| v.data.as_vec().clone())
        .take(5).collect::<Vec<_>>();

    let x_new = linspace(0, 1, 100);
    let u = s.iter().map(|v| {
        cubic_hermite_spline(&x, v, Quadratic)
    }).collect::<Vec<_>>();
    let s = u.iter().map(|f| {
        f.eval_vec(&x_new)
    }).collect::<Vec<_>>();

    let c = chebyshev_nodes(1000, 0, 1);
    let y = (0 .. u.len()).map(|_| {
        c.sample(x.len())
    }).collect::<Vec<_>>();
    let gu = u.iter().map(|f| {
        y.iter().map(|v| {
            v.iter().map(|x| {
                integrate(|x| f.eval(x), (0f64, *x), G7K15(1e-3))
            })
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x_new));
    df.push("s1", Series::new(s[0].clone()));
    df.push("s2", Series::new(s[1].clone()));
    df.push("s3", Series::new(s[2].clone()));
    df.push("s4", Series::new(s[3].clone()));
    df.push("s5", Series::new(s[4].clone()));
    df.push("y1", Series::new(y[0].clone()));
    df.push("y2", Series::new(y[1].clone()));
    df.push("y3", Series::new(y[2].clone()));
    df.push("y4", Series::new(y[3].clone()));
    df.push("y5", Series::new(y[4].clone()));
    df.print();

    df.write_nc("test.nc").unwrap();
}

/// Gaussian kernel
fn k(x: f64, y: f64) -> f64 {
    (-(x-y).powi(2) / (2.0 * 0.5f64.powi(2))).exp()
}
