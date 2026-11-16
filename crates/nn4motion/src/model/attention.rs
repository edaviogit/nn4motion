//! Multi-head self-attention (encoder-style, non-causal).

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    n_heads: usize,
    head_dim: usize,
    qkv: Linear,
    proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        assert!(d_model % n_heads == 0, "d_model must divide n_heads");
        let head_dim = d_model / n_heads;
        let qkv = linear(d_model, 3 * d_model, vb.pp("qkv"))?;
        let proj = linear(d_model, d_model, vb.pp("proj"))?;
        Ok(Self {
            n_heads,
            head_dim,
            qkv,
            proj,
        })
    }

    /// `x`: `[batch, seq, d_model]` → same shape.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((b, t, 3, self.n_heads, self.head_dim))?;
        // [3, batch, heads, seq, head_dim]
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let scale = (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(q.rank() - 1, q.rank() - 2)?)? / scale)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?;
        // [batch, seq, heads, head_dim]
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, self.n_heads * self.head_dim))?;
        self.proj.forward(&y)
    }
}
