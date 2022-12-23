use std::{
    io::{self, Read},
    slice,
    mem,
    path::Path,
    fs::File
};


struct LabelsReader
{
    amount: u32,
    index: u32,
    inner: File
}

impl LabelsReader
{
    pub fn create(mut inner: File) -> io::Result<Self>
    {
        //checks magic
        {
            let mut magic_buf = [0;mem::size_of::<u32>()];
            inner.read_exact(&mut magic_buf)?;

            if u32::from_be_bytes(magic_buf)!=2049
            {
                return Err(io::Error::from(io::ErrorKind::InvalidData));
            }
        }

        let mut amount_buf = [0;mem::size_of::<u32>()];
        inner.read_exact(&mut amount_buf)?;

        let amount = u32::from_be_bytes(amount_buf);

        Ok(LabelsReader{amount, index: 0, inner})
    }

    pub fn len(&self) -> usize
    {
        self.amount as usize
    }
}

impl Iterator for LabelsReader
{
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.index < self.amount
        {
            let mut buf = 0;
            self.inner.read_exact(slice::from_mut(&mut buf)).unwrap();

            self.index += 1;

            Some(buf)
        } else
        {
            None
        }
    }
}

struct ImagesReader
{
    amount: u32,
    index: u32,
    width: u32,
    height: u32,
    image_size: usize,
    inner: File
}

impl ImagesReader
{
    pub fn create(mut inner: File) -> io::Result<Self>
    {
        //checks magic
        {
            let mut magic_buf = [0;mem::size_of::<u32>()];
            inner.read_exact(&mut magic_buf)?;

            if u32::from_be_bytes(magic_buf)!=2051
            {
                return Err(io::Error::from(io::ErrorKind::InvalidData));
            }
        }

        let mut read_word = || -> io::Result<u32>
        {
            let mut buf = [0;mem::size_of::<u32>()];
            inner.read_exact(&mut buf)?;

            Ok(u32::from_be_bytes(buf))
        };

        let amount = read_word()?;

        let width = read_word()?;
        let height = read_word()?;

        let image_size = (width * height) as usize;

        Ok(ImagesReader{amount, index: 0, width, height, image_size, inner})
    }

    pub fn len(&self) -> usize
    {
        self.amount as usize
    }

    pub fn width(&self) -> u32
    {
        self.width
    }

    pub fn height(&self) -> u32
    {
        self.height
    }
}

impl Iterator for ImagesReader
{
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.index < self.amount
        {
            let mut buf = vec![0; self.image_size];
            self.inner.read_exact(&mut buf).unwrap();

            self.index += 1;

            Some(buf)
        } else
        {
            None
        }
    }
}

pub struct Digiter
{
    labels: LabelsReader,
    images: ImagesReader
}

impl Digiter
{
    pub fn create(labels_path: &str, images_path: &str) -> io::Result<Self>
    {
        if Path::new(labels_path).try_exists().and(Path::new(images_path).try_exists())?
        {
            let labels = LabelsReader::create(File::open(labels_path)?)?;
            let images = ImagesReader::create(File::open(images_path)?)?;

            if labels.len()!=images.len()
            {
                return Err(io::Error::from(io::ErrorKind::InvalidData));
            }

            return Ok(Digiter{labels, images});
        } else
        {
            return Err(io::Error::from(io::ErrorKind::NotFound));
        }
    }

    pub fn width(&self) -> u32
    {
        self.images.width()
    }

    pub fn height(&self) -> u32
    {
        self.images.height()
    }
}

impl Iterator for Digiter
{
    type Item = (u8, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item>
    {
        if let Some(label) = self.labels.next()
        {
            Some((label, self.images.next().unwrap()))
        } else
        {
            None
        }
    }
}