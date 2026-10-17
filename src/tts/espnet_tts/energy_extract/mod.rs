use std::marker::PhantomData;

use burn::prelude::*;

#[derive(Module, Debug)]
pub enum EnergyExtractChoice<B: Backend> {
    Dio(PhantomData<B>),
    None(PhantomData<B>),
}
