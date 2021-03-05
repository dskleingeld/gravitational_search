#![feature(test)]

// needed for bench macro
extern crate test;
mod traits;
mod stats;
pub use traits::{Minimize, Number, Stratagy, Stats};
pub use stats::TrackFitness;

use itertools::izip;
use derivative::Derivative;
pub use noisy_float::types::{r32, R32, r64, R64};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus as RandNumGen;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::RangeInclusive;

const EPSILON: f32 = f32::EPSILON;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct GSA<T, E, S, C, const D: usize>
where
    T: Number,
    E: Fn(&[T]) -> T,
    S: Stratagy<T>,
    C: Fn(usize, T) -> bool,
{
    #[derivative(Debug = "ignore")]
    rng: RandNumGen,
    agents: Vec<Agent<T, D>>,
    g0: T,
    alpha: T,
    // start temperature for annealing
    // None if we should not anneal
    t0: Option<T>,
    // maximum iterations
    max_n: usize,
    // dimension
    n: usize,
    // optimization problem: minimize, maximize or something that implements Stratagy
    strat: PhantomData<S>,
    #[derivative(Debug = "ignore")]
    // the evaluation function
    eval: E,
    #[derivative(Debug = "ignore")]
    // end criterion that can be used for early stopping
    end_criterion: C,
}

pub struct SearchResult<T, const D: usize> {
    pub fitness: T,
    pub params: [T; D],
    pub early_return: bool,
}

impl<T, E, S, C, const D: usize> GSA<T, E, S, C, D>
where
    T: Number,
    E: Fn(&[T]) -> T,
    S: Stratagy<T>,
    C: Fn(usize, T) -> bool,
{
    pub fn new(
        g0: T,
        alpha: T,
        max_n: usize,
        eval: E,
        end_criterion: C,
    ) -> GSA<T, E, S, C, D> {
        GSA {
            rng: RandNumGen::seed_from_u64(0),
            agents: Vec::new(),
            alpha,
            g0,
            t0: None,
            max_n,
            n: 0,
            strat: PhantomData,
            eval,
            end_criterion,
        }
    }
    pub fn with_annealing(mut self, t0: T) -> Self {
        self.t0 = Some(t0);
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = RandNumGen::seed_from_u64(seed);
        self
    }

    fn search_result(&mut self, fitness: Vec<T>, best: T, has_nan: bool) -> SearchResult<T, D> {
        let best_idx = fitness
            .into_iter()
            .enumerate()
            .find(|(_, f)| *f==best)
            .unwrap().0;
        SearchResult {
            fitness: best,
            params: self.agents[best_idx].x,
            early_return: has_nan,
        }
    }

    pub fn search(&mut self, range: RangeInclusive<T>, population: usize)
        -> SearchResult<T, D> 
    {
        let mut stats = traits::NoStats;
        self.search_w_stats(range, population, &mut stats)
    }

    pub fn search_w_stats(&mut self, range: RangeInclusive<T>, population: usize, stats: &mut impl Stats<T,D>)
        -> SearchResult<T, D> {
        assert!(population > 1, "population has to be at least 2");
        self.initialize_pop(population, range);

        loop {
            self.n += 1;
            let fitness = self.eval_fitness();

            let g = self.g();
            let best = fitness.iter().cloned().reduce(S::best).unwrap();
            let worst = fitness.iter().cloned().reduce(S::worst).unwrap();

            if Self::converged(best,worst) { // converged to a single point, can not search further
                stats.gather(&self.agents, best, worst, g, &fitness);
                return self.search_result(fitness, best, true);
            }

            self.update_masses(best, worst, &fitness);
            stats.gather(&self.agents, best, worst, g, &fitness);

            //halt if we reach the end criterion
            if (self.end_criterion)(self.n, best) {
                return self.search_result(fitness, best, false);
            }

            // sort the agents so the force function can use only the K best agents
            self.agents.sort_unstable_by(|a, b| {
                b.m.partial_cmp(&a.m).unwrap()
            });

            let forces = self.update_forces(g);
            if let Some(t0) = self.t0 {
                let temp = self.t(t0);
                self.update_agents_annealed(temp, forces, &fitness);
            } else {
                self.update_agents(forces);
            }
        }
    }

    fn converged(best: T, worst: T) -> bool {
        best == worst
    }

    fn initialize_pop(&mut self, n: usize, range: RangeInclusive<T>) {
        for _ in 0..n {
            let mut v = [T::zero(); D];
            let mut x = [T::zero(); D];
            for v in &mut v {
                *v = self.rng.gen_range(range.clone());
            }
            for x in &mut x {
                *x = self.rng.gen_range(range.clone());
            }
            self.agents.push(Agent { v, x, m: T::one() })
        }
    }
    fn g(&self) -> T {
        let n: T = T::from_usize(self.n).unwrap();
        let max_n: T = T::from_usize(self.max_n).unwrap();
        let minus: T = T::try_from(-1.0).map_err(|_| ()).unwrap();
        self.g0 * T::exp(minus * self.alpha * n / (max_n))
    }
    fn eval_fitness(&self) -> Vec<T> {
        self.agents.iter().map(|a| (self.eval)(&a.x)).collect()
    }
    fn update_masses(&mut self, best: T, worst: T, fitness: &[T]) {
        let masses = fitness
            .iter()
            .map(move |f| (*f - worst) / (best - worst)); //TODO FIXME div by zero (but how...)
        let sum: T = masses.clone().sum();
        let masses = masses.map(move |m| m / sum);
        for (mass, agent) in masses.zip(self.agents.iter_mut()) {
            agent.m = mass;
        }
    }
    fn k(&self) -> usize {
        let population = self.agents.len() as f32;
        let progress = self.n as f32/ self.max_n as f32;
        let k = population - population*progress + 1f32;
        k.trunc() as usize
    }
    fn update_forces(&mut self, g: T) -> Vec<[T; D]> {
        let k = self.k();
        let mut f: Vec<[T; D]> = Vec::with_capacity(self.agents.len());
        // calculating net force on agent i
        for i in &self.agents {
            let mut f_ij = [T::zero(); D];
            // agents j attracting agent i
            for j in self
                .agents
                .iter()
                .filter(|j| **j != *i)
                .take(k)
            // only take K best agents, linearly decreasing
            {
                let r = i.euclid_dist(j);
                // removed the multiplication with i.m as it cancels
                // out with the divide by i.m while calculating the
                // acceleration later
                // let gmmr = g*(i.m*j.m)/(r+f32::EPSILON);
                let epsilon: T = T::try_from(EPSILON.into()).map_err(|_| ()).unwrap();
                let gmmr = g * j.m / (r + epsilon);
                //set value of the force in every dimension
                for ((f, xi), xj) in f_ij.iter_mut().zip(i.x.iter()).zip(j.x.iter()) {
                    let rand = self.rng.gen_range(T::zero()..=T::one());
                    *f += gmmr * (*xj - *xi) * rand;
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
                *v = rand * (*v) + *a;
                *x += *v
            }
        }
    }
    /// TODO currently only works for minimization
    fn t(&self, t0: T) -> T { // TODO more interesting options: (geometric reduction, slow-decrease)
        let n: T = T::from_usize(self.n).unwrap();
        let max_n: T = T::from_usize(self.max_n).unwrap();
        let alpha = T::from_f64(0.01).unwrap();
        t0 - alpha 
    }
    /// TODO currently only works for minimization
    fn update(f: &[T;D], old_agent: &Agent<T,D>, rng: &mut RandNumGen) -> Agent<T, D> {
        let mut agent = old_agent.clone();
        let rand = rng.gen_range(T::zero()..=T::one());
        for (x,x_new,v,v_new,f) in izip!(
            old_agent.x.iter().copied(),
            agent.x.iter_mut(),
            old_agent.v.iter().copied(),
            agent.v.iter_mut(),
            f.iter().copied()) {

            *v_new = rand * v + f;
            *x_new = x + *v_new;
        }
        agent
    }
    /// TODO currently only works for minimization
    fn update_agents_annealed(&mut self, temp: T, forces: Vec<[T; D]>, fitness: &[T]) {
        for (force, agent, fit_old) in izip!(forces.iter(), &mut self.agents, fitness.iter().cloned()) {
            let new = Self::update(force, agent, &mut self.rng);
            let fit_new = (self.eval)(&new.x);
            
            let rand = self.rng.gen_range(T::zero()..=T::one());
            let div = (fit_new - fit_old)/temp;
            let exp = (-T::one() * div).exp();

            if fit_new <= fit_old || rand <= exp {
                *agent = new
            }
        }
    }
}

use std::fmt;
impl<T, E, S, C, const D: usize> fmt::Display for GSA<T, E, S, C, D>
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

#[derive(PartialEq, Debug, Clone)]
pub struct Agent<T, const D: usize> {
    v: [T; D],
    x: [T; D],
    m: T,
}
impl<T: Number, const D:usize> Agent<T,D> {
    fn new() -> Self {
        Agent {
            v: [T::zero(); D],
            x: [T::zero(); D],
            m: T::one(),
        }
    }
}

impl<T: Number, const D: usize> Agent<T, D> {
    pub fn euclid_dist(&self, other: &Agent<T, D>) -> T {
        self.x
            .iter()
            .zip(other.x.iter())
            .map(|(p, q)| (*p - *q).powi(2))
            .sum::<T>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::{Minimize, GSA};
    use test::Bencher;

    pub fn f1(x: &[f32]) -> f32 {
        x.iter().map(|x| x * x).sum()
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
            let mut gsa: GSA<f32, _, Minimize, _, DIMENSION> =
                GSA::new(g0, t0, alpha, max_n, SEED, f1, stop);
            gsa.search(-5f32..=5f32, POPULATION);
        });
    }
}
