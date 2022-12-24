use std::{fmt, env, thread, process, str, num::NonZeroUsize};

use digiter::*;
use neural_net::*;

mod digiter;
mod neural_net;


fn test_network(filename: &str, digit_reader: Digiter)
{
    let mut network = NeuralNet::load(filename).unwrap();

    let samples = 1000;

    let mut correct = 0;
    let mut combined_error = 0.0;
    for (index, (label, inputs)) in digit_reader.take(samples).enumerate()
    {
        let inputs = inputs.into_iter().map(|b| b as f64 / 255.0).collect::<Vec<f64>>();

        let out = network.feedforward(&inputs);
        
        if index==0
        {
            println!("sample output: {out:?} (correct {label})");
        }

        let guess = out.iter().enumerate()
            .reduce(|highest, current|
            {
                if current.1>highest.1 {current} else {highest}
            }).unwrap().0;

        combined_error += out.into_iter().enumerate().map(|(index, prediction)|
        {
            (if (index as u8)==label {1.0} else {0.0} - prediction).powi(2) * 0.5
        }).sum::<f64>();

        if label==guess as u8
        {
            correct += 1;
        }
    }

    println!("combined error: {combined_error}, percent correct: {:.2}%",
        (correct as f64 / samples as f64) * 100.0);
}

fn xorshift(mut x: u32) -> u32
{
    x ^= x << 13;
    x ^= x >> 17;
    x ^ (x << 5)
}

enum ProgramMode
{
    Train,
    Restart
}

fn train(filename: &str, digit_reader: Digiter, config: &Config)
{
    let image_size = (digit_reader.width() * digit_reader.height()) as usize;

    let layers = [
        DefaultLayerSettings{size: 50, transfer_function: TransferFunction::Tanh},
        DefaultLayerSettings{size: 50, transfer_function: TransferFunction::Tanh},
        DefaultLayerSettings{size: 10, transfer_function: TransferFunction::Sigmoid}
        ];

    let mut network = match config.mode
    {
        ProgramMode::Restart => NeuralNet::create(image_size, &layers),
        ProgramMode::Train => NeuralNet::load(filename).unwrap()
    };

    let iterations_progress = config.iterations/100;
    let mut progress = 1;
    while iterations_progress>progress
    {
        progress <<= 1;
    }
    let progress_mask = progress-1;
    let progress = progress as f64;

    let digit_reader = digit_reader.into_iter()
        .map(|(label, img)|
        {
            TrainSample
            {
                inputs: img.iter().map(|b| *b as f64 / 255.0).collect::<Vec<f64>>(),
                outputs: (0..10).map(|i| if i==label {1.0} else {0.0}).collect::<Vec<f64>>()
            }
        }).collect::<Vec<TrainSample>>();

    let seed = rand::random::<u32>();
    let mut progress_counter = 1.0;

    let random = xorshift(seed);
    let batch_begin = random as usize;
    for i in 0..config.iterations
    {
        let batch = (0..config.batch_size).map(|b|
        {
            unsafe
            {
            digit_reader.get_unchecked((i+b+batch_begin)%digit_reader.len()).clone()
            }
        }).collect::<Vec<TrainSample>>();
        network.backpropagate_multithreaded(&batch, config.threads);

        if (i & progress_mask)==0
        {
            let percent = progress_counter / (config.iterations as f64 / progress);

            print!("[");
            let length = 30;
            for i in 0..length
            {
                let part = i as f64 / length as f64;
                if part < percent
                {
                    print!("🌸");
                } else
                {
                    print!("__");
                }
            }

            println!("] {:.2}%", percent * 100.0);

            progress_counter += 1.0;
        }
    }

    network.save(filename).unwrap();
}

enum ConfigError
{
    InvalidArg(String),
    MissingValue,
    MissingRequired(String),
    InvalidValue(String)
}

impl fmt::Display for ConfigError
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        let mut val = None;
        
        let stringified = match self
        {
            ConfigError::InvalidArg(x) =>
            {
                val = Some(x);
                "invalid argument"
            },
            ConfigError::MissingValue => "argument value missing",
            ConfigError::MissingRequired(x) =>
            {
                val = Some(x);
                "missing required argument"
            },
            ConfigError::InvalidValue(x) =>
            {
                val = Some(x);
                "invalid value"
            }
        };

        let optional = if let Some(val) = val {format!(" ({val})")} else {"".to_owned()};
        write!(f, "{}{}", stringified, &optional)
    }
}

struct Config
{
    mode: ProgramMode,
    filename: String,
    threads: usize,
    iterations: usize,
    batch_size: usize,
    train_images: String,
    train_labels: String,
    test_images: String,
    test_labels: String
}

impl Config
{
    pub fn create(mut args: impl Iterator<Item=String>) -> Result<Self, ConfigError>
    {
        let mut mode = ProgramMode::Restart;
        let mut filename = "network.nn".to_owned();

        let mut threads = None;

        let mut iterations = 10;
        let mut batch_size = 10000;

        let mut train_labels = None;
        let mut train_images = None;
        
        let mut test_labels = None;
        let mut test_images = None;


        while let Some(arg) = args.next()
        {
            match arg.as_str()
            {
                "-M" | "--mode" =>
                {
                    mode = match args.next().ok_or(ConfigError::MissingValue)?.as_str()
                    {
                        "restart" => ProgramMode::Restart,
                        "train" => ProgramMode::Train,
                        x => return Err(ConfigError::InvalidValue(x.to_owned()))
                    };
                },
                "-o" | "--output" =>
                {
                    filename = args.next().ok_or(ConfigError::MissingValue)?;
                },
                "--threads" =>
                {
                    threads = Some(Self::number_arg::<usize>(&mut args)?);
                },
                "-I" | "--iter" =>
                {
                    iterations = Self::number_arg(&mut args)?;
                },
                "-b" | "--batch" =>
                {
                    batch_size = Self::number_arg(&mut args)?;
                },
                "-i" | "--images" =>
                {
                    train_images = Some(args.next().ok_or(ConfigError::MissingValue)?);
                },
                "-l" | "--labels" =>
                {
                    train_labels = Some(args.next().ok_or(ConfigError::MissingValue)?);
                },
                "-t" | "--test-images" =>
                {
                    test_images = Some(args.next().ok_or(ConfigError::MissingValue)?);
                },
                "-T" | "--test-labels" =>
                {
                    test_labels = Some(args.next().ok_or(ConfigError::MissingValue)?);
                },
                "-h" | "--help" =>
                {
                    Self::help_message();
                },
                x => return Err(ConfigError::InvalidArg(x.to_owned()))
            }
        }

        let train_labels: String =
            train_labels.ok_or(ConfigError::MissingRequired("--labels".to_owned()))?;

        let train_images: String =
            train_images.ok_or(ConfigError::MissingRequired("--images".to_owned()))?;

        let test_images: String = if test_images.is_none()
        {
            train_images.clone()
        } else
        {
            test_images.unwrap()
        };

        let test_labels: String = if test_labels.is_none()
        {
            train_labels.clone()
        } else
        {
            test_labels.unwrap()
        };

        let threads = threads.unwrap_or_else(||
        {
            thread::available_parallelism().unwrap_or_else(|_| NonZeroUsize::new(1).unwrap()).get()
        });

        Ok(Config{
            mode, filename,
            threads,
            iterations, batch_size,
            train_images, train_labels,
            test_images, test_labels
        })
    }

    fn number_arg<T>(mut args: impl Iterator<Item=String>) -> Result<T, ConfigError>
    where
        T: str::FromStr,
        <T as str::FromStr>::Err: fmt::Display
    {
        args.next().ok_or(ConfigError::MissingValue)?.trim().parse()
            .map_err(|err| ConfigError::InvalidValue(format!("{err}")))
    }

    pub fn help_message() -> !
    {
        println!("usage: {} [args]", env::args().next().unwrap());
        println!("args:");
        println!("    -h, --help         display this help messsage");
        println!("    -M, --mode         program mode (default restart)");
        println!("    -o, --output       output filename (default network.nn)");
        println!("    --threads          override the amount of threads used");
        println!("    -I, --iter         iterations to train for (default 10)");
        println!("    -b, --batch        batch size (default 10000)");
        println!("    -i, --images       mnist training images");
        println!("    -l, --labels       mnist training labels");
        println!("    -t, --test-images  optional test images (uses training otherwise)");
        println!("    -T, --test-labels  optional test labels (uses training otherwise)");
        println!("program modes:");
        println!("    restart, train");

        process::exit(1)
    }
}

fn main()
{
    let config = Config::create(env::args().skip(1)).unwrap_or_else(|err|
    {
        println!("parse error: {err}");
        Config::help_message()
    });

    let train_digiter = Digiter::create(
        &config.train_labels,
        &config.train_images
    ).unwrap();
    train(&config.filename, train_digiter, &config);

    let test_digiter = Digiter::create(
        &config.test_labels,
        &config.test_images
    ).unwrap();
    test_network(&config.filename, test_digiter);
}
