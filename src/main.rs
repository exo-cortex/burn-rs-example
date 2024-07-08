mod data;
mod inference;
mod model;
mod training;

use std::fs::File;

use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        // wgpu::AutoGraphicsApi,
        Autodiff,
        // Wgpu,
    },
    data::dataset::Dataset,
    optim::AdamConfig,
};

use argh::{from_env, FromArgs};
use png::Decoder;

#[derive(FromArgs)]
/// Config
struct MyArgs {
    /// do train
    #[argh(switch)]
    pub do_train: bool,
    /// do stuff
    #[argh(switch)]
    pub do_stuff: bool,
}

fn main() {
    let args: MyArgs = argh::from_env();

    // type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyBackend = NdArray;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = NdArrayDevice::default();
    let artifact_dir = "./mnist_model_example";

    if args.do_train {
        crate::training::train::<MyAutodiffBackend>(
            artifact_dir,
            TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
            device.clone(),
        );
    } else {
        crate::inference::infer::<MyBackend>(
            artifact_dir,
            device,
            burn::data::dataset::vision::MnistDataset::test()
                .get(42)
                .unwrap(),
        );
    }

    let decoder = png::Decoder::new(File::open("/images/04.png").unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    if args.do_stuff {
        // crate::inference::infer::<MyBackend>(&artifact_dir,
        // device,
        for i in 0..1 {
            let data = burn::data::dataset::vision::MnistDataset::test()
                .get(i)
                .unwrap();

            println!("{:?}", data)
        }
    }
}
