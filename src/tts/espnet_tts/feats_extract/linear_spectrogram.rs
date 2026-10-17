use std::f64::consts::PI;

use burn::{
    prelude::*,
    tensor::{
        signal::{stft, StftOptions},
        DataError,
    },
};

#[derive(Module, Debug)]
pub struct LinearSpectrogram<B: Backend> {
    window: Option<Tensor<B, 1>>,
    normalized: bool,
    stft_options: StftOptions,
}

impl<B: Backend> LinearSpectrogram<B> {
    pub fn new(
        n_fft: Option<usize>,
        win_length: Option<usize>,
        hop_length: Option<usize>,
        window: Option<&str>,
        center: Option<bool>,
        normalized: Option<bool>,
        onesided: Option<bool>,
        device: &B::Device,
    ) -> Self {
        let n_fft = n_fft.unwrap_or(1024);
        let win_length = Some(win_length.unwrap_or(n_fft));
        let hop_length = hop_length.unwrap_or(256);
        let window = window.unwrap_or("hann");
        let center = center.unwrap_or(true);
        let normalized = normalized.unwrap_or(false);
        let onesided = onesided.unwrap_or(true);
        let stft_options = StftOptions {
            n_fft,
            hop_length,
            win_length,
            center,
            onesided,
        };

        let window = match window {
            "hann" => {
                let win_length = win_length.unwrap();
                let n = Tensor::<B, 1, Int>::arange(0..(win_length as i64), device).float();
                let theta = 2_f64 * PI / (win_length - 1) as f64 * n;
                Some(0.5 - 0.5 * theta.cos())
            }
            _ => panic!("Not implement. (window={})", window),
        };

        Self {
            window,
            normalized,
            stft_options,
        }
    }

    pub fn output_size(&self) -> usize {
        self.stft_options.n_fft / 2 + 1
    }

    /// # Arguments
    /// * `input`: (Batch, Nsamples, Channels)
    /// * `input_length`: (Batch)
    /// # Returns
    /// * `input_amp`:(Batch, Frames, Channels, Freq)
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        input_length: Tensor<B, 1>,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 1>), DataError> {
        let device = input.device();
        let [batch, n_samples, channels] = input.dims();

        // Short-Time Fourier Transform (STFT)
        // (Batch, Nsamples, Channels) -> (Batch * Channels, Nsamples)
        let input = input
            .permute([0, 2, 1])
            .reshape([batch * channels, n_samples]);
        let input_stft = stft(input, self.window.clone(), self.stft_options);
        // (Batch * Channel, Freq, Frames, 2=real_imag) -> (Batch, Frames, Channel, Freq, 2=real_imag)
        let [_, freq, frames, _] = input_stft.dims();
        let input_stft = input_stft
            .reshape([batch, channels, freq, frames, 2])
            .permute([0, 3, 1, 2, 4]);

        // 有効なフレームを計算
        let input_length = if self.stft_options.center {
            let pad = self.stft_options.n_fft / 2;
            input_length + 2.0 * pad as f64
        } else {
            input_length
        };
        let output_length: Tensor<B, 1> = 1 + Tensor::floor(
            (input_length - self.stft_options.n_fft as f64) / self.stft_options.hop_length as f64,
        );
        let invalid_mask = make_pad_mask(output_length.clone(), input_stft.dims(), 1, &device)?;
        let input_stft = input_stft.mask_fill(invalid_mask, 0.0);

        // 絶対値計算
        let input_power = input_stft.powi_scalar(2).sum_dim(-1).squeeze_dim::<4>(4);
        let input_amp = input_power.clamp_min(1e-10).sqrt();

        Ok((input_amp, output_length))
    }
}

/// paddingされた領域を`True`で埋めた`Tensor`を作成する。
/// # Argmuments
/// * `output_length`: paddingされていないときの長さ
/// * `output_dims`: 出力`Tensor`の次元数
/// * `dim`: 時系列方向の軸
/// * device: 出力`Tensor`のデバイス
/// # Returns
/// paddingされた領域を`True`で埋めたoutput_dimsと同次元の`Tensor`
/// # Error
/// `output_length`のいずれかの要素が`output_dims[dim]`より大きい場合に`burn::tensor::DataError`
fn make_pad_mask<B: Backend>(
    output_length: Tensor<B, 1>,
    output_dims: [usize; 5],
    dim: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5, Bool>, DataError> {
    let target_dim = output_dims[dim] as i64;
    let batch = output_dims[0] as i64;

    let pad_mask = output_length
        .to_data()
        .to_vec::<i64>()?
        .iter()
        .map(|len| {
            Tensor::cat(
                vec![
                    Tensor::<B, 1>::zeros([len], device),
                    Tensor::<B, 1>::ones([target_dim - len], device),
                ],
                0,
            )
        })
        .collect::<Vec<Tensor<B, 1>>>();
    let pad_mask = Tensor::stack::<2>(pad_mask, 0)
        .bool()
        .reshape([batch, -1, 1, 1, 1])
        .swap_dims(1, dim)
        .expand(output_dims);

    Ok(pad_mask)
}
