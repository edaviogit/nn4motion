//! Prompt conditioning: any non-empty subset of {position, velocity, current}.
//!
//! # Masking scheme
//!
//! Each timestep is a 6-D feature:
//!
//! ```text
//! [v_pos * m_pos, v_vel * m_vel, v_cur * m_cur, m_pos, m_vel, m_cur]
//! ```
//!
//! - `v_*` are continuous channel values.
//! - `m_* ∈ {0, 1}` mark whether that channel is present in the prompt.
//! - Masked channels contribute `0` in the value slots and `0` in the mask slots,
//!   so the model can tell "absent" from "present and zero".
//! - At least one mask must be `1` for a valid prompt (any non-empty subset).
//!
//! The network always regresses all three targets `(pos, vel, cur)`.

use crate::config::{INPUT_DIM, N_CHANNELS};
use anyhow::{bail, Result};
use candle_core::{Device, Tensor};

/// One sample of conditioned motion data.
#[derive(Clone, Debug)]
pub struct MotionSample {
    /// Shape `[seq, 3]` — ground-truth position / velocity / current.
    pub values: Vec<[f32; N_CHANNELS]>,
    /// Shape `[seq, 3]` — `1.0` if channel is provided in the prompt, else `0.0`.
    pub masks: Vec<[f32; N_CHANNELS]>,
}

impl MotionSample {
    pub fn seq_len(&self) -> usize {
        self.values.len()
    }

    pub fn validate(&self) -> Result<()> {
        if self.values.is_empty() {
            bail!("sample must have at least one timestep");
        }
        if self.values.len() != self.masks.len() {
            bail!("values and masks length mismatch");
        }
        let any_present = self
            .masks
            .iter()
            .any(|m| m.iter().any(|&x| x > 0.5));
        if !any_present {
            bail!("prompt must include a non-empty subset of channels");
        }
        Ok(())
    }

    /// Encode to `[seq, INPUT_DIM]` float features.
    pub fn encode_features(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let mut out = Vec::with_capacity(self.seq_len() * INPUT_DIM);
        for (v, m) in self.values.iter().zip(self.masks.iter()) {
            for c in 0..N_CHANNELS {
                out.push(v[c] * m[c]);
            }
            for c in 0..N_CHANNELS {
                out.push(m[c]);
            }
        }
        Ok(out)
    }

    /// Flat target vector `[seq * 3]`.
    pub fn flat_targets(&self) -> Vec<f32> {
        self.values.iter().flat_map(|v| v.iter().copied()).collect()
    }
}

/// Encode a batch of samples into tensors.
///
/// Returns `(features, targets)` with shapes
/// `[batch, seq, INPUT_DIM]` and `[batch, seq, N_CHANNELS]`.
pub fn encode_batch(samples: &[MotionSample], device: &Device) -> Result<(Tensor, Tensor)> {
    if samples.is_empty() {
        bail!("empty batch");
    }
    let seq = samples[0].seq_len();
    let batch = samples.len();
    for s in samples {
        if s.seq_len() != seq {
            bail!("ragged batch: expected seq={seq}, got {}", s.seq_len());
        }
    }

    let mut feat = Vec::with_capacity(batch * seq * INPUT_DIM);
    let mut tgt = Vec::with_capacity(batch * seq * N_CHANNELS);
    for s in samples {
        feat.extend(s.encode_features()?);
        tgt.extend(s.flat_targets());
    }

    let features = Tensor::from_vec(feat, (batch, seq, INPUT_DIM), device)?;
    let targets = Tensor::from_vec(tgt, (batch, seq, N_CHANNELS), device)?;
    Ok((features, targets))
}

/// Build a random valid mask pattern for one timestep (non-empty subset).
pub fn random_mask(rng: &mut impl rand::Rng) -> [f32; N_CHANNELS] {
    loop {
        let m = [
            if rng.gen_bool(0.5) { 1.0 } else { 0.0 },
            if rng.gen_bool(0.5) { 1.0 } else { 0.0 },
            if rng.gen_bool(0.5) { 1.0 } else { 0.0 },
        ];
        if m.iter().any(|&x| x > 0.5) {
            return m;
        }
    }
}

/// Synthetic toy trajectory for smoke training.
pub fn synthetic_sample(seq_len: usize, rng: &mut impl rand::Rng) -> MotionSample {
    let mut values = Vec::with_capacity(seq_len);
    let mut masks = Vec::with_capacity(seq_len);
    let mut pos = rng.gen_range(-1.0..1.0);
    let mut vel = rng.gen_range(-0.5..0.5);
    for _ in 0..seq_len {
        let cur = rng.gen_range(-1.0..1.0);
        vel += 0.05 * cur;
        pos += vel * 0.05;
        values.push([pos, vel, cur]);
        masks.push(random_mask(rng));
    }
    MotionSample { values, masks }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn rejects_empty_prompt() {
        let sample = MotionSample {
            values: vec![[1.0, 2.0, 3.0]],
            masks: vec![[0.0, 0.0, 0.0]],
        };
        assert!(sample.validate().is_err());
    }

    #[test]
    fn encodes_masked_channels() {
        let sample = MotionSample {
            values: vec![[1.0, 2.0, 3.0]],
            masks: vec![[1.0, 0.0, 1.0]],
        };
        let feat = sample.encode_features().unwrap();
        assert_eq!(feat, vec![1.0, 0.0, 3.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn synthetic_is_valid() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let s = synthetic_sample(8, &mut rng);
        s.validate().unwrap();
        assert_eq!(s.seq_len(), 8);
    }
}
