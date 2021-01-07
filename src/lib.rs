#![feature(iterator_fold_self)]

use rand::{Fill, Rng};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus as RandNumGen;
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use derivative::Derivative;

pub trait Stratagy {
    fn best(a: f32, b: f32) -> f32;
    fn worst(a: f32, b: f32) -> f32;
}

pub struct Minimize;
impl Stratagy for Minimize {
    fn best(a: f32, b: f32) -> f32 {
        a.min(b)
    }
    fn worst(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct GSA<E, S, C, const D: usize> 
    where 
        E: Fn(&[f32]) -> f32,
        S: Stratagy,
        C: Fn(usize, f32) -> bool,
{
    #[derivative(Debug="ignore")]
    rng: RandNumGen,
    agents: Vec<Agent<D>>,
    g0: f32,
    alpha: f32,
    t: f32,
    max_n: usize,
    n: usize,
    strat: PhantomData<S>,
    #[derivative(Debug="ignore")]
    eval: E,
    #[derivative(Debug="ignore")]
    end_criterion: C,
}

impl<E, S, C, const D: usize> GSA<E,S,C,D> 
    where 
        E: Fn(&[f32]) -> f32,
        S: Stratagy,
        C: Fn(usize, f32) -> bool,
{
    pub fn new(g0: f32, t0: f32, alpha: f32, max_n: usize, seed: u64, eval: E, end_criterion: C) -> GSA<E,S,C,D> {
        GSA {
            rng: RandNumGen::seed_from_u64(seed),
            agents: Vec::new(),
            alpha,
            g0,
            t: t0,
            max_n,
            n: 0,
            strat: PhantomData,
            eval,
            end_criterion,
        }
    }

    pub fn search(&mut self, range: RangeInclusive<f32>, population: usize) {
        assert!(population > 1, "population has to be at least 2");
        self.initialize_pop(population, range);

        loop {
            dbg!(&self);
            self.n += 1;
            let fitness = self.eval_fitness();
            let g = self.g();
            let best = fitness.iter()
                .cloned()
                .fold_first(S::best).unwrap();
            let worst = fitness.iter()
                .cloned()
                .fold_first(S::worst).unwrap();

            // dbg!(g,best, worst);
            self.update_masses(best, worst, fitness);
            let forces = self.update_forces(g);
            self.update_agents(forces);

            if (self.end_criterion)(self.n, best) {
                dbg!(best);
                break
            }
        }
    }
    fn initialize_pop(&mut self, n: usize, range: RangeInclusive<f32>) {
        let center = |v: &mut f32| *v = (*v)*(range.end()-range.start())+range.start();
        for _ in 0..n {
            let mut v = [0f32; D];
            let mut x = [0f32; D];
            v.try_fill(&mut self.rng).unwrap();
            x.try_fill(&mut self.rng).unwrap();
            v.iter_mut().for_each(center);
            x.iter_mut().for_each(center);
            self.agents.push( Agent {v, x, m: 1f32})
        }
    }
    fn g(&self) -> f32 {
        self.g0*f32::exp(-1.0*self.alpha*(self.n as f32)/((self.max_n) as f32))
    }
    fn eval_fitness(&self) -> Vec<f32> {
        self.agents.iter()
            .map(|a| (self.eval)(&a.x))
            .collect()
    }
    fn update_masses(&mut self, best: f32, worst: f32, fitness: Vec<f32>) {
        let masses = fitness.into_iter().map(move |f| (f-worst)/(best-worst));
        let sum: f32 = masses.clone().sum();
        let masses = masses.map(move |m| m/sum);
        for (mass, agent) in masses.zip(self.agents.iter_mut()) {
            agent.m = mass;
        }
    }
    fn update_forces(&mut self, g: f32) -> Vec<[f32; D]> {
        let mut f: Vec<[f32; D]> = Vec::with_capacity(self.agents.len());
        for i in &self.agents {
            let mut f_ij = [0f32; D];
            for j in self.agents.iter().filter(|j| **j != *i) {
                let r = i.euclid_dist(j);
                // removed the multiplication with i.m as it cancels 
                // out with the divide by i.m while calculating the 
                // acceleration later
                // let gmmr = g*(i.m*j.m)/(r+f32::EPSILON);
                let gmmr = g*j.m/(r+f32::EPSILON);
                let rand = self.rng.gen_range(0f32..=1f32);
                //set value of the force in every dimension
                for ((f, xi),xj) in f_ij.iter_mut()
                    .zip(i.x.iter())
                    .zip(j.x.iter()) {
                    *f += gmmr*(xj-xi)*rand;
                }
            }
            f.push(f_ij);
        }
        f
    }
    fn update_agents(&mut self, forces: Vec<[f32; D]>) {
        for (forces, agent) in forces.iter().zip(&mut self.agents) {
            let rand = self.rng.gen_range(0f32..=1f32);
            for ((f, v), x) in forces.iter().zip(&mut agent.v).zip(&mut agent.x) {
                // let a = f/agent.m;
                let a = f;
                *v = rand*(*v) + a;
                *x = *x + *v
            }
        }
    }
}

use std::fmt;
impl<E, S, C, const D: usize> fmt::Display for GSA<E,S,C,D>
    where 
        E: Fn(&[f32]) -> f32,
        S: Stratagy,
        C: Fn(usize, f32) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "iteration: {}", self.n)
    }
}

#[derive(PartialEq, Debug)]
struct Agent<const D: usize> {
    v: [f32; D],
    x: [f32; D],
    m: f32,
}

impl<const D: usize> Agent<D> {
    pub fn euclid_dist(&self, other: &Agent<D>) -> f32 {
        self.x.iter().zip(other.x.iter())
            .map(|(p,q)| (p-q).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

