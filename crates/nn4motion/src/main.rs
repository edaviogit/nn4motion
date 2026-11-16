use anyhow::Result;
use candle_core::Device;
use clap::{Parser, Subcommand};
use nn4motion::config::{CONTEXT_LEN, D_FF, D_MODEL, N_HEADS, N_LAYERS};
use nn4motion::train::{run_training, TrainConfig};

#[derive(Parser, Debug)]
#[command(
    name = "nn4motion",
    about = "NN4motion: 1-layer Transformer for position/velocity/current"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Print architecture / spec constants.
    Info,
    /// Tiny synthetic training run (CPU) to verify the stack.
    Smoke {
        #[arg(long, default_value_t = 5)]
        steps: usize,
        #[arg(long, default_value_t = 2)]
        batch_size: usize,
        #[arg(long, default_value_t = 16)]
        seq_len: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Train on synthetic data with configurable length (≤ 1024).
    Train {
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        #[arg(long, default_value_t = 32)]
        seq_len: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info => {
            println!("NN4motion Transformer");
            println!("  ML stack     : candle (candle-core / candle-nn)");
            println!("  layers       : {N_LAYERS}");
            println!("  context_len  : {CONTEXT_LEN}");
            println!("  n_heads      : {N_HEADS}");
            println!("  d_model      : {D_MODEL}");
            println!("  d_ff         : {D_FF}  (ReLU)");
            println!("  optimizer    : AdamW");
            println!("  outputs      : position, velocity, current");
            println!(
                "  prompt       : any non-empty subset of channels via value*mask + mask bits"
            );
        }
        Commands::Smoke {
            steps,
            batch_size,
            seq_len,
            seed,
        } => {
            let cfg = TrainConfig {
                steps,
                batch_size,
                seq_len,
                seed,
                log_every: 1,
                ..TrainConfig::smoke()
            };
            let loss = run_training(&cfg, &Device::Cpu)?;
            println!("smoke finished, last_loss={loss:.6}");
        }
        Commands::Train {
            steps,
            batch_size,
            seq_len,
            lr,
            seed,
        } => {
            let cfg = TrainConfig {
                steps,
                batch_size,
                seq_len,
                lr,
                seed,
                log_every: 5,
            };
            let loss = run_training(&cfg, &Device::Cpu)?;
            println!("train finished, last_loss={loss:.6}");
        }
    }
    Ok(())
}
