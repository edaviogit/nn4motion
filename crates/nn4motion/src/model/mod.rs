//! Transformer building blocks for NN4motion.

pub mod attention;
pub mod ffn;
pub mod layer;
pub mod transformer;

pub use transformer::{build_default, MotionTransformer};
