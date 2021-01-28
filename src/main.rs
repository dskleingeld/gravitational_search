// use grav_search::{r32, R32};
use grav_search::{r64, R64};
use grav_search::{Minimize, GSA};

mod gsa_paper {
    use std::iter::Sum;
    use std::ops::Mul;
    pub fn f1<T: Copy + Mul + Sum<<T as Mul>::Output>>(x: &[T]) -> T {
        x.iter().map(|x| (*x) * (*x)).sum()
    }
}

fn main() {
    const SEED: u64 = 0;
    const POPULATION: usize = 50; // check
    const DIMENSION: usize = 30; // check

    let g0 = r64(100.); // check
    let alpha = r64(20.); // check
    let t0 = r64(1_000.); // unused
    let max_n = 1_000; // check

    let stop = |n: usize, _| n > max_n;
    
    for seed in 0..100 {
        let mut gsa: GSA<R64, _, Minimize, _, DIMENSION> =
            GSA::new(g0, t0, alpha, max_n, seed, gsa_paper::f1, stop);
        let res = gsa.search(r64(-100.)..=r64(100.), POPULATION);

        println!("fitness: {:+e}", res.fitness);
        println!("params:");
        for param in &res.params {
            println!("{:+e}", param);
        }
    }
}
