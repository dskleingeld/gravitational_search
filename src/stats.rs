use std::io::Write;
use std::fs::File;

use crate::traits::{Stats, Number};
use crate::Agent;

#[derive(Default)]
pub struct TrackFitness {
    best: Vec<f64>,
    avg: Vec<f64>,
}

impl<T: Number + Into<f64>, const D: usize> Stats<T, D> for TrackFitness {
    fn gather(&mut self, _agents: &[Agent<T,D>], best: T, _worst: T, _g: T, fitness: &[T]){
        let len = T::from_usize(fitness.len()).unwrap(); 
        let sum = fitness.iter().copied().sum::<T>(); 
        let avg = sum/len;
        self.best.push(best.into());
        self.avg.push(avg.into());
    }
}

impl TrackFitness {
    pub fn best_to_file(&self, file: &mut File) {
        for best in &self.best {
            write!(file, "{} ", best).unwrap();
        }
        writeln!(file, "").unwrap();
    }
}

