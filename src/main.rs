#![feature(iterator_fold_self)]

use rand::{Fill, Rng};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus as RandNumGen;
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

struct GSA<E, S, const D: usize> 
    where 
        E: Fn(&[f32]) -> f32,
        S: Stratagy,
{
    rng: RandNumGen,
    agents: Vec<Agent<D>>,
    best: f32,
    worst: f32,
    g0: f32,
    alpha: f32,
    t: f32,
    n: usize,
    strat: PhantomData<S>,
    eval: E,
}

impl<E, S, const D: usize> GSA<E,S,D> 
    where 
        E: Fn(&[f32]) -> f32,
        S: Stratagy,
{
    pub fn new(g0: f32, t0: f32, eval: E) -> GSA<E,S,D> {
        GSA {
            rng: RandNumGen::seed_from_u64(0),
            agents: Vec::new(),
            best: S::best(f32::MAX, f32::MIN),
            worst: S::worst(f32::MAX, f32::MIN),
            alpha: 0.0,
            g0,
            t: t0,
            n: 0,
            strat: PhantomData,
            eval,
        }
    }

    pub fn search(&mut self) {
        self.initialize_pop(100, 0.0..1.0);

        loop {
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
            self.agents.push( Agent {v, x, m: 0f32})
        }
    }
    fn g(&self) -> f32 {
        self.g0*f32::exp(-1.0*self.alpha*self.t/(self.n as f32))
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
                let gmmr = g*(i.m*j.m)/(r+f32::EPSILON);
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
    fn end_criterion(&self) -> bool {
        todo!()
    }
}

#[derive(PartialEq)]
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

fn main() {
    println!("Hello, world!");
}
