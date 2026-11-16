//! NN4motion: single-layer Transformer for conditioned motion targets.
//!
//! Predicts continuous **position**, **velocity**, and **current** from a prompt
//! that may include any non-empty subset of those three channels (see [`data`]).

pub mod config;
pub mod data;
pub mod model;
pub mod train;

pub use config::{CONTEXT_LEN, N_CHANNELS, N_HEADS, N_LAYERS};
pub use model::{build_default, MotionTransformer};
pub use train::{run_training, TrainConfig};
