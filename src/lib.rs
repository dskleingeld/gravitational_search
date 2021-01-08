#![feature(iterator_fold_self)]
#![feature(test)]

// needed for bench macro
extern crate test;

use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus as RandNumGen;
use std::marker::PhantomData;
use std::ops::{RangeInclusive, AddAssign};
use num_traits::float::Float;
use num_traits::cast::FromPrimitive;
use std::iter::Sum;
use std::fmt::Debug;
use rand::distributions::uniform::SampleUniform;
use derivative::Derivative;
pub use noisy_float::types::{R32, r32};

const EPSILON: f32 = f32::EPSILON;

pub trait Number: Float + Debug + Copy + Clone 
    + FromPrimitive
    + std::convert::TryFrom<f32>
    + Sum + AddAssign + SampleUniform {}

impl Number for R32 {}
impl Number for f32 {}
impl Number for f64 {}

pub trait Stratagy<T> 
    where 
        T: Number,
{
    fn best(a: T, b: T) -> T;
    fn worst(a: T, b: T) -> T;
}

pub struct Minimize;
impl<T> Stratagy<T> for Minimize 
    where
        T: Number,
{
    fn best(a: T, b: T) -> T {
        a.min(b)
    }
    fn worst(a: T, b: T) -> T {
        a.max(b)
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct GSA<T, E, S, C, const D: usize> 
    where 
        T: Number,
        E: Fn(&[T]) -> T,
        S: Stratagy<T>,
        C: Fn(usize, T) -> bool,
{
    #[derivative(Debug="ignore")]
    rng: RandNumGen,
    agents: Vec<Agent<T,D>>,
    g0: T,
    alpha: T,
    // t: T,
    max_n: usize,
    n: usize,
    strat: PhantomData<S>,
    #[derivative(Debug="ignore")]
    eval: E,
    #[derivative(Debug="ignore")]
    end_criterion: C,
}

impl<T, E, S, C, const D: usize> GSA<T,E,S,C,D> 
    where 
        T: Number,
        E: Fn(&[T]) -> T,
        S: Stratagy<T>,
        C: Fn(usize, T) -> bool,
{
    pub fn new(g0: T, _t0: T, alpha: T, max_n: usize, seed: u64, eval: E, end_criterion: C) -> GSA<T,E,S,C,D> {
        GSA {
            rng: RandNumGen::seed_from_u64(seed),
            agents: Vec::new(),
            alpha,
            g0,
            // t: t0,
            max_n,
            n: 0,
            strat: PhantomData,
            eval,
            end_criterion,
        }
    }

    pub fn search(&mut self, range: RangeInclusive<T>, population: usize) {
        assert!(population > 1, "population has to be at least 2");
        self.initialize_pop(population, range);

        loop {
            self.n += 1;
            let fitness = self.eval_fitness();
            let g = self.g();
            let best = fitness.iter()
                .cloned()
                .fold_first(S::best).unwrap();
            let worst = fitness.iter()
                .cloned()
                .fold_first(S::worst).unwrap();

            self.update_masses(best, worst, fitness);
            let forces = self.update_forces(g);
            self.update_agents(forces);

            if (self.end_criterion)(self.n, best) {
                dbg!(best);
                break
            }
        }
    }
    fn initialize_pop(&mut self, n: usize, range: RangeInclusive<T>) {
        for _ in 0..n {
            let mut v = [T::zero(); D]; //TODO
            let mut x = [T::zero(); D];
            for v in &mut v {
                *v = self.rng.gen_range(range.clone());
            }
            for x in &mut x {
                *x = self.rng.gen_range(range.clone());
            }
            self.agents.push( Agent {v, x, m: T::one()})
        }
    }
    fn g(&self) -> T {
        let n: T = T::from_usize(self.n).unwrap(); 
        let max_n: T = T::from_usize(self.max_n).unwrap();
        let minus: T = T::try_from(-1.0).map_err(|_| ()).unwrap();
        self.g0*T::exp(minus*self.alpha*n/(max_n))
    }
    fn eval_fitness(&self) -> Vec<T> {
        self.agents.iter()
            .map(|a| (self.eval)(&a.x))
            .collect()
    }
    fn update_masses(&mut self, best: T, worst: T, fitness: Vec<T>) {
        let masses = fitness.into_iter().map(move |f| (f-worst)/(best-worst));
        let sum: T = masses.clone().sum();
        let masses = masses.map(move |m| m/sum);
        for (mass, agent) in masses.zip(self.agents.iter_mut()) {
            agent.m = mass;
        }
    }
    fn update_forces(&mut self, g: T) -> Vec<[T; D]> {
        let mut f: Vec<[T; D]> = Vec::with_capacity(self.agents.len());
        for i in &self.agents {
            let mut f_ij = [T::zero(); D];
            for j in self.agents.iter().filter(|j| **j != *i) {
                let r = i.euclid_dist(j);
                // removed the multiplication with i.m as it cancels 
                // out with the divide by i.m while calculating the 
                // acceleration later
                // let gmmr = g*(i.m*j.m)/(r+f32::EPSILON);
                let epsilon: T = T::try_from(EPSILON).map_err(|_| ()).unwrap();
                let gmmr = g*j.m/(r+epsilon);
                let rand = self.rng.gen_range(T::one()..=T::zero());
                //set value of the force in every dimension
                for ((f, xi),xj) in f_ij.iter_mut()
                    .zip(i.x.iter())
                    .zip(j.x.iter()) {
                    *f += gmmr*(*xj-*xi)*rand;
                }
            }
            f.push(f_ij);
        }
        f
    }
    fn update_agents(&mut self, forces: Vec<[T; D]>) {
        for (forces, agent) in forces.iter().zip(&mut self.agents) {
            let rand = self.rng.gen_range(T::zero()..=T::one());
            for ((f, v), x) in forces.iter().zip(&mut agent.v).zip(&mut agent.x) {
                // let a = f/agent.m;
                let a = f;
                *v = rand*(*v) + *a;
                *x += *v
            }
        }
    }
}

use std::fmt;
impl<T, E, S, C, const D: usize> fmt::Display for GSA<T,E,S,C,D>
    where 
        T: Number,
        E: Fn(&[T]) -> T,
        S: Stratagy<T>,
        C: Fn(usize, T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "iteration: {}", self.n)
    }
}

#[derive(PartialEq, Debug)]
struct Agent<T, const D: usize> {
    v: [T; D],
    x: [T; D],
    m: T,
}

impl<T: Number, const D: usize> Agent<T,D> {
    pub fn euclid_dist(&self, other: &Agent<T,D>) -> T {
        self.x.iter().zip(other.x.iter())
            .map(|(p,q)| (*p-*q).powi(2))
            .sum::<T>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::{GSA, Minimize};
    use test::Bencher;

    pub fn f1(x: &[f32]) -> f32 {
        x.iter().map(|x| x*x).sum()
    }

    #[bench]
    fn bench_f1(b: &mut Bencher) {
        let g0 = 100f32;
        let alpha = 20f32;
        let t0 = 20f32;
        let max_n = 1000;
        const SEED: u64 = 0;
        const POPULATION: usize = 50;
        const DIMENSION: usize = 2;

        let stop = |n: usize, _| n > max_n;
        b.iter(|| {
            let mut gsa: GSA<f32,_, Minimize, _, DIMENSION> = GSA::new(g0, t0, alpha, max_n, SEED, f1, stop);
            gsa.search(-5f32..=5f32, POPULATION);
        });
    }
}
