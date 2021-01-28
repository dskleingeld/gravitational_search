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
    let g0 = r32(100f32); // check
    let alpha = r32(20f32); // check
    let t0 = r32(1_000f32); // unused
    let max_n = 1_000; // check
    const SEED: u64 = 1;
    const POPULATION: usize = 50; // check
    const DIMENSION: usize = 30; // check

    let stop = |n: usize, _| n > max_n;
    let mut gsa: GSA<R32, _, Minimize, _, DIMENSION> = GSA::new(g0, t0, alpha, max_n, SEED, gsa_paper::f1, stop);
    gsa.search(r32(-100f32)..=r32(100f32), POPULATION);
}
