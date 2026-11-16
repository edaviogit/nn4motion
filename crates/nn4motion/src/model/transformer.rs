//! Conditionally prompted motion Transformer (1 layer by default).

use super::layer::TransformerLayer;
use crate::config::{
    CONTEXT_LEN, D_FF, D_MODEL, INPUT_DIM, N_CHANNELS, N_HEADS, N_LAYERS,
};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Sinusoidal positional encoding `[1, seq, d_model]`, truncated/padded to `seq_len`.
fn sinusoidal_position_encoding(seq_len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    assert!(
        seq_len <= CONTEXT_LEN,
        "seq_len {seq_len} exceeds CONTEXT_LEN {CONTEXT_LEN}"
    );
    let mut data = vec![0f32; seq_len * d_model];
    for pos in 0..seq_len {
        for i in 0..d_model {
            let div = (10000f32).powf((2 * (i / 2)) as f32 / d_model as f32);
            let angle = pos as f32 / div;
            data[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Tensor::from_vec(data, (1, seq_len, d_model), device)
}

#[derive(Debug, Clone)]
pub struct MotionTransformer {
    input_proj: Linear,
    layers: Vec<TransformerLayer>,
    output_head: Linear,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
}

impl MotionTransformer {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Self::with_dims(D_MODEL, N_HEADS, D_FF, N_LAYERS, vb)
    }

    pub fn with_dims(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_proj = linear(INPUT_DIM, d_model, vb.pp("input_proj"))?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(TransformerLayer::new(
                d_model,
                n_heads,
                d_ff,
                vb.pp(format!("layer{i}")),
            )?);
        }
        let output_head = linear(d_model, N_CHANNELS, vb.pp("output_head"))?;
        Ok(Self {
            input_proj,
            layers,
            output_head,
            d_model,
            n_heads,
            n_layers,
        })
    }

    /// `features`: `[batch, seq, INPUT_DIM]` → predictions `[batch, seq, 3]`.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = features.dims3()?;
        let mut x = self.input_proj.forward(features)?;
        let pos = sinusoidal_position_encoding(seq, self.d_model, features.device())?;
        let pos = pos.to_dtype(x.dtype())?.broadcast_as((batch, seq, self.d_model))?;
        x = (x + pos)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        self.output_head.forward(&x)
    }
}

/// Convenience constructor on CPU with fresh weights.
pub fn build_default(device: &Device) -> Result<(MotionTransformer, candle_nn::VarMap)> {
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = MotionTransformer::new(vb)?;
    Ok((model, varmap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::INPUT_DIM;

    #[test]
    fn forward_shapes() -> Result<()> {
        let device = Device::Cpu;
        let (model, _vm) = build_default(&device)?;
        assert_eq!(model.n_layers, 1);
        assert_eq!(model.n_heads, 8);
        let x = Tensor::zeros((2, 16, INPUT_DIM), DType::F32, &device)?;
        let y = model.forward(&x)?;
        assert_eq!(y.dims(), &[2, 16, 3]);
        Ok(())
    }
}
