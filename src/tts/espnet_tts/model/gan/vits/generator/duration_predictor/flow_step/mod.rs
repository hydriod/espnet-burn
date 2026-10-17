mod affine;
mod conv;
mod flip;

use burn::prelude::*;

use self::affine::ElementwiseAffineFlow;
pub use self::conv::{ConvFlow, DilatedDepthSeparableConv, DilatedDepthSeparableConvConfig};
use self::flip::FlipFlow;

#[derive(Module, Debug)]
pub enum FlowStep<B: Backend> {
    ElementwiseAffine(ElementwiseAffineFlow<B>),
    Conv(ConvFlow<B>),
    Flip(FlipFlow),
}
impl<B: Backend> FlowStep<B> {
    pub fn elementwise_affine_flow(channels: usize, device: &B::Device) -> Self {
        let elementwise_affine = ElementwiseAffineFlow::new(channels, device);
        Self::ElementwiseAffine(elementwise_affine)
    }

    pub fn conv_flow(
        in_channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        layers: usize,
        bins: Option<usize>,
        tail_bound: Option<f64>,
        device: &B::Device,
    ) -> Self {
        let conv = ConvFlow::new(
            in_channels,
            hidden_channels,
            kernel_size,
            layers,
            bins,
            tail_bound,
            device,
        );
        Self::Conv(conv)
    }

    pub fn flip_flow() -> Self {
        Self::Flip(FlipFlow)
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
        inverse: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        match self {
            FlowStep::ElementwiseAffine(affine) => affine.forward(x, x_mask, inverse),
            FlowStep::Conv(conv) => conv.forward(x, x_mask, g, inverse),
            FlowStep::Flip(flip) => flip.forward(x, inverse),
        }
    }
}
