pub use noisy_float::types::{r32, R32, r64, R64};
use num_traits::cast::FromPrimitive;
use num_traits::float::Float;
use rand::distributions::uniform::SampleUniform;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

pub trait Number:
    Float
    + Debug
    + Copy
    + Clone
    + PartialOrd
    + FromPrimitive
    + std::convert::TryFrom<f64>
    + Sum
    + AddAssign
    + SampleUniform
{
}

// impl Number for R32 {}
impl Number for R64 {}
// impl Number for f32 {}
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
