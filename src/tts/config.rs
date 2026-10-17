use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub config: String,
    pub token_list: Vec<String>,
    pub odim: Option<i64>,
    pub feats_extract: String,
    pub feats_extract_conf: Option<FeatsExtractConfig>,
    pub normalize: Option<String>,
    pub tts: String,
    pub tts_conf: Option<TTSConig>,
    pub pitch_extract: Option<String>,
    pub pitch_extract_conf: Option<PitchExtractConfig>,
    pub energy_extract: Option<String>,
    pub energy_extract_conf: Option<EnergyExtractConfig>,
    pub pitch_normalize: Option<String>,
    pub pitch_normalize_conf: Option<PitchNormalizeConfig>,
    pub energy_normalize: Option<String>,
    pub energy_normalize_conf: Option<EnergyNormalizeConfig>,
    pub model_conf: Option<ModelConfig>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FeatsExtractConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: Option<usize>,
    pub window: Option<String>,
    pub center: Option<bool>,
    pub normalized: Option<bool>,
    pub onesided: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TTSConig {
    pub generator_type: String,
    pub generator_params: GeneratorParams,
}
#[derive(Clone, Debug, Deserialize)]
pub struct GeneratorParams {
    pub hidden_channels: u64,
    pub spks: i64,
    pub global_channels: i64,
    pub segment_size: u64,
    pub text_encoder_attention_heads: u64,
    pub text_encoder_ffn_expand: u64,
    pub text_encoder_blocks: u64,
    pub text_encoder_positionwise_layer_type: String,
    pub text_encoder_positionwise_conv_kernel_size: u64,
    pub text_encoder_positional_encoding_layer_type: String,
    pub text_encoder_self_attention_layer_type: String,
    pub text_encoder_activation_type: String,
    pub text_encoder_normalize_before: bool,
    pub text_encoder_dropout_rate: f64,
    pub text_encoder_positional_dropout_rate: f64,
    pub text_encoder_attention_dropout_rate: f64,
    pub use_macaron_style_in_text_encoder: bool,
    pub use_conformer_conv_in_text_encoder: bool,
    pub text_encoder_conformer_kernel_size: i64,
    pub decoder_kernel_size: u64,
    pub decoder_channels: u64,
    pub decoder_upsample_scales: Vec<u64>,
    pub decoder_upsample_kernel_sizes: Vec<u64>,
    pub decoder_resblock_kernel_sizes: Vec<u64>,
    pub decoder_resblock_dilations: Vec<Vec<u64>>,
    pub use_weight_norm_in_decoder: bool,
    pub posterior_encoder_kernel_size: u64,
    pub posterior_encoder_layers: u64,
    pub posterior_encoder_stacks: u64,
    pub posterior_encoder_base_dilation: u64,
    pub posterior_encoder_dropout_rate: f64,
    pub use_weight_norm_in_posterior_encoder: bool,
    pub flow_flows: u64,
    pub flow_kernel_size: u64,
    pub flow_base_dilation: u64,
    pub flow_layers: u64,
    pub flow_dropout_rate: f64,
    pub use_weight_norm_in_flow: bool,
    pub use_only_mean_in_flow: bool,
    pub stochastic_duration_predictor_kernel_size: u64,
    pub stochastic_duration_predictor_dropout_rate: f64,
    pub stochastic_duration_predictor_flows: u64,
    pub stochastic_duration_predictor_dds_conv_layers: u64,
    pub vocabs: u64,
    pub aux_channels: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PitchExtractConfig {}

#[derive(Clone, Debug, Deserialize)]
pub struct EnergyExtractConfig {}
#[derive(Clone, Debug, Deserialize)]
pub struct PitchNormalizeConfig {}
#[derive(Clone, Debug, Deserialize)]
pub struct EnergyNormalizeConfig {}
#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {}
