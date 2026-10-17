use std::marker::PhantomData;

use burn::prelude::*;

pub struct DurationCalculator<B: Backend> {
    _marker: PhantomData<B>,
}
impl<B: Backend> DurationCalculator<B> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
    pub fn forward(&self, att_ws: Tensor<B, 2>) -> DurationOutput<B> {
        let focus_rate = self.calculate_focus_rate(att_ws.clone());
        let duration = self.calculate_duration(att_ws);

        DurationOutput {
            focus_rate,
            duration,
        }
    }

    fn calculate_focus_rate(&self, att_ws: Tensor<B, 2>) -> Tensor<B, 1> {
        att_ws.max_dim(1).squeeze_dim::<1>(1).mean()
    }

    fn calculate_duration(&self, att_ws: Tensor<B, 2>) -> Tensor<B, 1> {
        let durations = Tensor::cat(
            (0..att_ws.shape()[1])
                .map(|i| att_ws.clone().argmax(1).equal_elem(i as f64).float().sum())
                .collect(),
            0,
        );

        durations
    }
}

pub struct DurationOutput<B: Backend> {
    pub focus_rate: Tensor<B, 1>,
    pub duration: Tensor<B, 1>,
}
