use grav_search::{GSA, Minimize};
use grav_search::{R32, r32};

mod gsa_paper {
    use std::ops::Mul;
    use std::iter::Sum;
    pub fn f1<T: Copy+Mul+ Sum<<T as Mul>::Output>>(x: &[T]) -> T {
        x.iter().map(|x| (*x)*(*x) ).sum()
    }
}

fn main() {
    let g0 = r32(100f32);
    let alpha = r32(20f32);
    let t0 = r32(20f32);
    let max_n = 5;
    const SEED: u64 = 0;
    const POPULATION: usize = 2;
    const DIMENSION: usize = 2;

    let stop = |n: usize, _| n > max_n;
    let mut gsa: GSA<R32, _, Minimize, _, DIMENSION> = GSA::new(g0, t0, alpha, max_n, SEED, gsa_paper::f1, stop);
    gsa.search(r32(-5f32)..=r32(5f32), POPULATION);
}
