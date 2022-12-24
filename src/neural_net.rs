use std::{
    fs::File,
    io,
    thread
};

pub use layer::*;

use serde::{Serialize, Deserialize};
#[cfg(test)]
use rand::Rng;


mod layer;


#[derive(Debug, Clone)]
pub struct TrainSample
{
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNet
{
    inputs_amount: usize,
    layers: Vec<DefaultLayer>,
}

#[allow(dead_code)]
impl NeuralNet
{
    pub fn create(
        inputs_amount: usize,
        layers: &[DefaultLayerSettings],
    ) -> Self
    {
        assert!(!layers.is_empty());

        let layers = layers.iter().cloned().enumerate().map(|(i, layer)|
            {
                let DefaultLayerSettings{size, transfer_function} = layer;

                let prev_size = if i==0
                {
                    inputs_amount
                } else
                {
                    layers[i-1].size
                };

                DefaultLayer::new(size, prev_size, transfer_function)
            }).collect::<Vec<DefaultLayer>>();

        NeuralNet{
            inputs_amount,
            layers
        }
    }

    pub fn load(filename: &str) -> Result<Self, ciborium::de::Error<io::Error>>
    {
        let mut net = ciborium::de::from_reader::<Self, _>(File::open(filename)
            .map_err(|err| ciborium::de::Error::Io(err))?)?;

        net.layers.iter_mut().for_each(|layer| layer.reset_temporary());

        Ok(net)
    }

    pub fn save(&self, filename: &str) -> Result<(), ciborium::ser::Error<io::Error>>
    {
        ciborium::ser::into_writer(&self, File::create(filename)
            .map_err(|err| ciborium::ser::Error::Io(err))?)
    }

    pub fn feedforward(&mut self, inputs: &[f64]) -> Vec<f64>
    {
        self.feedforward_inner(inputs);

        let last_layer = self.layers.last().unwrap();
        
        let transfer_function = last_layer.transfer_function();
        last_layer.neurons().iter().map(|n| transfer_function.t_f(*n)).collect::<Vec<f64>>()
    }

    fn feedforward_inner(&mut self, inputs: &[f64])
    {
        for layer in 0..self.layers.len()
        {
            if layer==0
            {
                let c_layer = unsafe{ self.layers.get_unchecked_mut(0) };
                c_layer.feedforward(&inputs, TransferFunction::Nothing);
            } else
            {
                let ptr = self.layers.as_mut_ptr();
                let previous_layer = unsafe{ ptr.add(layer-1) };
                let current_layer = unsafe{ ptr.add(layer) };

                unsafe
                {
                (*current_layer).feedforward((*previous_layer).neurons(),
                    (*previous_layer).transfer_function());
                }
            }
        }
    }

    pub fn backpropagate_multithreaded(&mut self, mut samples: &[TrainSample], threads: usize)
    {
        thread::scope(|scope|
        {
            let mut handles = Vec::new();

            if samples.len()>=threads
            {
                let samples_per_thread = samples.len()/threads;

                for _ in 0..threads-1
                {
                    let current_samples;
                    (current_samples, samples) = samples.split_at(samples_per_thread);

                    let mut network_copy = self.clone();

                    handles.push(scope.spawn(move ||
                    {
                        network_copy.backpropagate_nonapply(current_samples);
                        network_copy
                    }));
                }
            }

            self.backpropagate_nonapply(samples);

            for handle in handles
            {
                self.combine(&handle.join().unwrap());
            }
        });

        self.apply_gradients();
    }

    pub fn backpropagate(&mut self, samples: &[TrainSample])
    {
        self.backpropagate_nonapply(samples);
        self.apply_gradients();
    }

    fn backpropagate_nonapply(&mut self, samples: &[TrainSample])
    {
        for sample in samples
        {
            self.feedforward_inner(&sample.inputs);
            self.backpropagate_inner(&sample.inputs, &sample.outputs);
        }
    }

    fn apply_gradients(&mut self)
    {
        self.layers.iter_mut().for_each(|layer|
        {
            layer.apply_gradients();
        });
    }

    fn combine(&mut self, other: &NeuralNet)
    {
        self.layers.iter_mut().zip(other.layers.iter()).for_each(|(layer, other_layer)|
        {
            layer.combine(other_layer);
        });
    }

    fn backpropagate_inner(&mut self, inputs: &[f64], outputs: &[f64])
    {
        for layer in (0..self.layers.len()).rev()
        {
            let last_layer = self.layers.len()-1;

            let ptr = self.layers.as_mut_ptr();

            let previous_layer = if layer==0
            {
                inputs.to_vec()
            } else
            {
                let prev_layer = unsafe{ ptr.add(layer-1) };
                let tf = unsafe{ (*prev_layer).transfer_function() };
                unsafe
                {
                (*prev_layer).neurons().iter().map(|neuron| tf.t_f(*neuron)).collect::<Vec<f64>>()
                }
            };

            if layer==last_layer
            {
                unsafe
                {
                    (*ptr.add(layer)).backpropagate(&previous_layer, InnerOuter::Outputs(outputs));
                }
            } else
            {
                let current_layer = unsafe{ ptr.add(layer) };
                let next_layer = unsafe{ ptr.add(layer+1) };

                let inners;
                unsafe
                {
                inners = InnerOuter::Inners(
                    (*next_layer).neurons(),
                    (*next_layer).weights()
                );
                }

                unsafe
                {
                (*current_layer).backpropagate(&previous_layer, inners);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    use layer::tests::{get_weight, get_gradient};

    #[test]
    fn backprop()
    {
        for _ in 0..10
        {
            backprop_single();
        }
    }

    fn backprop_single()
    {
        let mut rng = rand::thread_rng();

        let layers_amount = rng.gen_range(1..10);
        let layers = (0..layers_amount).map(|_|
            {
                let transfer_functions = [TransferFunction::Sigmoid];
                let t_index = rng.gen_range(0..transfer_functions.len());

                let transfer_function = transfer_functions[t_index];

                let size = rng.gen_range(1..10);
                DefaultLayerSettings{size, transfer_function}
            }).collect::<Vec<DefaultLayerSettings>>();
    
        let inputs_amount = rng.gen_range(1..10);
        let mut network = NeuralNet::create(inputs_amount, &layers);

        let change = 0.1;

        for t_l in 0..layers_amount
        {
            for t_n in 0..network.layers[t_l].size()
            {
                let previous_amount = if t_l==0
                {
                    inputs_amount
                } else
                {
                    network.layers[t_l-1].size()
                };
                for t_p in 0..(previous_amount+1)
                {
                    let test_input = (0..inputs_amount).map(|_| rng.gen())
                        .collect::<Vec<f64>>();

                    dbg!(t_l, t_n, t_p, &test_input);
                    let normal_weight = *get_weight(&mut network.layers[t_l], t_n, t_p);

                    *get_weight(&mut network.layers[t_l], t_n, t_p) = normal_weight + change;
                    let output = network.feedforward(&test_input);
                    let left = output.into_iter().sum::<f64>();

                    *get_weight(&mut network.layers[t_l], t_n, t_p) = normal_weight - change;
                    let output = network.feedforward(&test_input);
                    let right = output.into_iter().sum::<f64>();

                    *get_weight(&mut network.layers[t_l], t_n, t_p) = normal_weight;

                    network.feedforward_inner(&test_input);

                    let answers_layer = network.layers.last().unwrap();
                    
                    let tf = answers_layer.transfer_function();
                    let answers = answers_layer.neurons();
                    
                    //set correct outputs so the error is always 1.0
                    let test_output = answers.iter().map(|current|
                    {
                        tf.t_f(*current) - 1.0
                    }).collect::<Vec<f64>>();

                    network.feedforward_inner(&test_input);
                    network.backpropagate_inner(&test_input, &test_output);

                    let deriv = *get_gradient(&mut network.layers[t_l], t_n, t_p);
                    let real_deriv = (left - right) / (2.0 * change);

                    for c_l in 0..network.layers.len()
                    {
                        for c_n in 0..network.layers[c_l].size()
                        {
                            let previous_amount = if c_l==0
                            {
                                inputs_amount
                            } else
                            {
                                network.layers[c_l-1].size()
                            };

                            for c_p in 0..(previous_amount+1)
                            {
                                *get_gradient(&mut network.layers[c_l], c_n, c_p) = 0.0;
                            }
                        }
                    }

                    dbg!(&network);
                    println!("left: {left}, right: {right}");
                    print!("(layer: {t_l} neuron: {t_n} previous: {t_p})  ");
                    println!("backprop: {deriv}, derivative: {real_deriv}");
                    println!("diff: {}", deriv-real_deriv);

                    assert!((deriv-real_deriv).abs()<0.001);
                }
            }
        }
    }

    #[test]
    fn it_learns()
    {
        let layers = [
            DefaultLayerSettings{size: 2, transfer_function: TransferFunction::Sigmoid2},
            DefaultLayerSettings{size: 2, transfer_function: TransferFunction::Sigmoid2},
            DefaultLayerSettings{size: 1, transfer_function: TransferFunction::Sigmoid}
        ];
        let network = std::cell::RefCell::new(NeuralNet::create(2, &layers));
    
        let mut rng = rand::thread_rng();
        let mut gen_sample = |out: usize| -> TrainSample
        {
            let first = rng.gen::<f64>()*0.5;
            let second = if out==0
            {
                rng.gen::<f64>()*(0.5-first)
            } else
            {
                (rng.gen::<f64>()*first).max(0.5)
            };
    
            TrainSample{inputs: vec![first, second], outputs: vec![out as f64]}
        };
    
        let test = |sample: TrainSample|
        {
            let outputs = network.borrow_mut().feedforward(&sample.inputs);
            println!("should be {:?}, is {outputs:?}", &sample.outputs);
        };
    
        for _ in 0..2000
        {
            let samples = (0..10).map(|i| gen_sample(i%2)).collect::<Vec<_>>();
            network.borrow_mut().backpropagate(&samples);
    
            test(gen_sample(0));
            test(gen_sample(1));
        }

        let outputs = network.borrow_mut().feedforward(&gen_sample(0).inputs);
        assert!(outputs[0]<0.4);

        let outputs = network.borrow_mut().feedforward(&gen_sample(1).inputs);
        assert!(outputs[0]>0.6);
    }
}
