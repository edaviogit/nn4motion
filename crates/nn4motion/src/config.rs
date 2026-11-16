//! Spec-aligned model and training constants for NN4motion.

/// Maximum sequence length (context window).
pub const CONTEXT_LEN: usize = 1024;

/// Number of Transformer encoder layers.
pub const N_LAYERS: usize = 1;

/// Multi-head self-attention heads.
pub const N_HEADS: usize = 8;

/// Model / residual stream width.
pub const D_MODEL: usize = 64;

/// Feed-forward hidden width (`4 * D_MODEL`).
pub const D_FF: usize = 256;

/// Continuous targets / channels: position, velocity, current.
pub const N_CHANNELS: usize = 3;

/// Prompt feature width: masked values (3) + presence masks (3).
pub const INPUT_DIM: usize = N_CHANNELS * 2;

/// Channel index helpers.
pub const CH_POSITION: usize = 0;
pub const CH_VELOCITY: usize = 1;
pub const CH_CURRENT: usize = 2;

/// Default AdamW hyperparameters (smoke / baseline).
pub const ADAMW_LR: f64 = 1e-3;
pub const ADAMW_BETA1: f64 = 0.9;
pub const ADAMW_BETA2: f64 = 0.999;
pub const ADAMW_EPS: f64 = 1e-8;
pub const ADAMW_WEIGHT_DECAY: f64 = 0.01;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_constants() {
        assert_eq!(CONTEXT_LEN, 1024);
        assert_eq!(N_LAYERS, 1);
        assert_eq!(N_HEADS, 8);
        assert_eq!(N_CHANNELS, 3);
        assert_eq!(D_MODEL % N_HEADS, 0);
    }
}
