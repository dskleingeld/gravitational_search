#![feature(iterator_fold_self)]

use rand::Fill;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus as Rng;
use std::marker::PhantomData;

trait Stratagy {
    fn best(a: f32, b: f32) -> f32;
    fn worst(a: f32, b: f32) -> f32;
}

struct Minimize;
impl Stratagy for Minimize {
    fn best(a: f32, b: f32) -> f32 {
        a.min(b)
    }
    fn worst(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

struct GSA<Stratagy, const D: usize> {
    rng: Rng,
    agents: Vec<Agent<D>>,
    best: f32,
    worst: f32,
    g0: f32,
    alpha: f32,
    t: f32,
    n: usize,
    strat: PhantomData<Stratagy>,
}

impl<S: Stratagy, const D: usize> GSA<S, D> {
    pub fn new(g0: f32, t0: f32) -> GSA<S, D> {
        GSA {
            rng: Rng::seed_from_u64(0),
            agents: Vec::new(),
            best: S::best(f32::MAX, f32::MIN),
            worst: S::worst(f32::MAX, f32::MIN),
            alpha: 0.0,
            g0,
            t: t0,
            n: 0,
            strat: PhantomData,
        }
    }

    pub fn search(&mut self) {
        self.initialize_pop(100, 0.0..1.0);

        loop {
            let fitness = self.eval_fitness();
            let g = self.g();
            self.best = fitness.iter()
                .cloned()
                .fold_first(S::best).unwrap();
            self.worst = fitness.iter()
                .cloned()
                .fold_first(S::worst).unwrap();

            self.update_mass_acceleration();
            self.update_velocity_position();

            if self.end_criterion() {
                break
            }
        }
    }
    fn initialize_pop(&mut self, n: usize, range: std::ops::Range<f32>) {
        let center = |v: &mut f32| *v = (*v)*(range.end-range.start)+range.start;
        for _ in 0..n {
            let mut v = [0f32; D];
            let mut x = [0f32; D];
            v.try_fill(&mut self.rng).unwrap();
            x.try_fill(&mut self.rng).unwrap();
            v.iter_mut().for_each(center);
            x.iter_mut().for_each(center);
            self.agents.push( Agent {v, x})
        }
    }
    fn g(&self) -> f32 {
        self.g0*f32::exp(-1.0*self.alpha*self.t/(self.n as f32))
    }
    fn eval(params: &[f32]) -> f32 {
        todo!()
    }
    fn eval_fitness(&self) -> Vec<f32> {
        self.agents.iter()
            .map(|a| Self::eval(&a.x))
            .collect()
    }
    fn update_mass_acceleration(&self) {
        todo!()
    }
    fn update_velocity_position(&self) {
        todo!()
    }
    fn end_criterion(&self) -> bool {
        todo!()
    }
}

struct Agent<const D: usize> {
    v: [f32; D],
    x: [f32; D],
}

fn main() {
    println!("Hello, world!");
}
