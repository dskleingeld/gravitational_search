use std::fs::File;

// use grav_search::{r32, R32};
use grav_search::{r64, R64};
use grav_search::{Minimize, GSA};
use grav_search::TrackFitness;

mod gsa_paper {
    use grav_search::Number;
    use std::iter::Sum;
    use std::ops::Mul;

    /// GSA paper function F1, GSASA function F1 
    pub fn f1<T: Copy + Mul + Sum<<T as Mul>::Output>>(x: &[T]) -> T {
        x.iter().map(|x| (*x) * (*x)).sum()
    }
    /// GSA paper function F5, GSASA function F3 
    pub fn f2<T: Number>(x: &[T]) -> T {
        let c100: T = T::from_f64(100f64).unwrap();
        x.chunks(2).map(|x| {
            c100 * (x[1] - x[0].powi(2)).powi(2) + (x[0] - T::one()).powi(2) 
        }).sum()
    }
}

fn main() {
    const SEED: u64 = 0;
    const POPULATION: usize = 50; // check
    const DIMENSION: usize = 30; // check

    let g0 = r64(100.); // check
    let alpha = r64(20.); // check
    let t0 = r64(20.);
    // let kb = r64() //check
    let max_n = 1_000; // check

    let stop = |n: usize, _| n > max_n;
    
    let mut file = File::create("data/gsa/f1_gsa.stats").unwrap();
    for seed in 0..100 {
        let mut stats = TrackFitness::default();
        let mut gsa: GSA<R64, _, Minimize, _, DIMENSION> =
            GSA::new(g0, alpha, max_n, gsa_paper::f1, stop)
                .seed(seed);
        let _res = gsa.search_w_stats(r64(-100.)..=r64(100.), POPULATION, &mut stats);
        stats.best_to_file(&mut file);
    }

    let mut file = File::create("data/gsa/f2_gsa.stats").unwrap();
    for seed in 0..100 {
        let mut stats = TrackFitness::default();
        let mut gsa: GSA<R64, _, Minimize, _, DIMENSION> =
            GSA::new(g0, alpha, max_n, gsa_paper::f2, stop)
                .seed(seed);
        let res = gsa.search_w_stats(r64(-30.)..=r64(30.), POPULATION, &mut stats);
        stats.best_to_file(&mut file);
    }
}
