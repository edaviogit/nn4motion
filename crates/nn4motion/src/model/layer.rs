//! Single Transformer encoder layer: MHA + FFN with residual LayerNorm.

use super::attention::MultiHeadAttention;
use super::ffn::FeedForward;
use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    attn: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerLayer {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: MultiHeadAttention::new(d_model, n_heads, vb.pp("attn"))?,
            ffn: FeedForward::new(d_model, d_ff, vb.pp("ffn"))?,
            norm1: layer_norm(d_model, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(d_model, 1e-5, vb.pp("norm2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(x)?;
        let x = self.norm1.forward(&(x + h)?)?;
        let h = self.ffn.forward(&x)?;
        self.norm2.forward(&(x + h)?)
    }
}
