//! Training loop with AdamW.

use crate::config::{
    ADAMW_BETA1, ADAMW_BETA2, ADAMW_EPS, ADAMW_LR, ADAMW_WEIGHT_DECAY, CONTEXT_LEN,
};
use crate::data::{encode_batch, synthetic_sample};
use crate::model::build_default;
use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    /// Sequence length used for this run (must be `<= CONTEXT_LEN`).
    pub seq_len: usize,
    pub lr: f64,
    pub seed: u64,
    pub log_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 20,
            batch_size: 4,
            seq_len: 32,
            lr: ADAMW_LR,
            seed: 42,
            log_every: 5,
        }
    }
}

impl TrainConfig {
    pub fn smoke() -> Self {
        Self {
            steps: 5,
            batch_size: 2,
            seq_len: 16,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.seq_len == 0 || self.seq_len > CONTEXT_LEN {
            bail!(
                "seq_len must be in 1..=CONTEXT_LEN ({CONTEXT_LEN}), got {}",
                self.seq_len
            );
        }
        if self.batch_size == 0 {
            bail!("batch_size must be > 0");
        }
        if self.steps == 0 {
            bail!("steps must be > 0");
        }
        Ok(())
    }
}

fn mse_loss(pred: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    let diff = (pred - target)?;
    diff.sqr()?.mean_all()
}

/// Run a short supervised training loop on synthetic conditioned trajectories.
pub fn run_training(cfg: &TrainConfig, device: &Device) -> Result<f32> {
    cfg.validate()?;
    let (model, varmap) = build_default(device)?;
    let params = ParamsAdamW {
        lr: cfg.lr,
        beta1: ADAMW_BETA1,
        beta2: ADAMW_BETA2,
        eps: ADAMW_EPS,
        weight_decay: ADAMW_WEIGHT_DECAY,
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);

    let mut last_loss = f32::NAN;
    for step in 1..=cfg.steps {
        let batch: Vec<_> = (0..cfg.batch_size)
            .map(|_| synthetic_sample(cfg.seq_len, &mut rng))
            .collect();
        let (features, targets) = encode_batch(&batch, device)?;
        let preds = model.forward(&features)?;
        let loss = mse_loss(&preds, &targets)?;
        opt.backward_step(&loss)?;
        last_loss = loss.to_scalar::<f32>()?;
        if step == 1 || step % cfg.log_every == 0 || step == cfg.steps {
            println!(
                "step {step:>4}/{:>4}  loss={last_loss:.6}  seq={}  batch={}",
                cfg.steps, cfg.seq_len, cfg.batch_size
            );
        }
    }
    Ok(last_loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_train_runs() -> Result<()> {
        let loss = run_training(&TrainConfig::smoke(), &Device::Cpu)?;
        assert!(loss.is_finite());
        Ok(())
    }
}
