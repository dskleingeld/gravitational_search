use grav_search::{GSA, Minimize};

mod gsa_paper {
    pub fn f1(x: &[f32]) -> f32 {
        x.iter().map(|x| x*x).sum()
    }
}

fn main() {
    let g0 = 100f32;
    let alpha = 20f32;
    let t0 = 20f32;
    let max_n = 1000;
    const POPULATION: usize = 50;
    const DIMENSION: usize = 2;

    let stop = |n: usize, _v: f32| n > max_n;
    let mut gsa: GSA<f32, _, Minimize, _, DIMENSION> = GSA::new(g0, t0, alpha, max_n, gsa_paper::f1, stop);
    gsa.search(-5f32..=5f32, POPULATION);
}
