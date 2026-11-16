//! Position-wise feed-forward network with ReLU activation (per spec).

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(d_model, d_ff, vb.pp("fc1"))?,
            fc2: linear(d_ff, d_model, vb.pp("fc2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }
}
