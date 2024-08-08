// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A module for signal sources for "flacenc-bin".

use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;

use flacenc::error::SourceError;
use flacenc::error::SourceErrorReason;
use flacenc::source::Fill;
use flacenc::source::Source;

/// An example of `flacenc::source::Source` based on `hound::WavReader`.
///
/// To mitigate I/O overhead due to sample-by-sample retrieval in hound API,
/// this source only uses hound to parse WAV header and seeks offset for the
/// first sample. After parsing the header, the inside `BufReader` is obtained
/// via `WavReader::into_inner` and it is used to retrieve blocks of samples.
#[allow(clippy::module_name_repetitions)]
pub struct HoundSource {
    pub(crate) spec: hound::WavSpec,
    pub(crate) duration: usize,
    reader: BufReader<File>,
    pub(crate) bytes_per_sample: usize,
    bytebuf: Vec<u8>,
    current_offset: usize,
    file_size: Option<usize>,
}

impl HoundSource {
    /// Constructs `HoundSource` from `path`.
    ///
    /// # Errors
    ///
    /// The function fails when file is not found or has invalid format. This
    /// function currently do not support WAVs with IEEE float samples, and it
    /// returns `SourceError` with `SourceErrorReason::InvalidFormat` if the
    /// samples are floats.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file_size = path.as_ref().metadata().ok().map(|x| x.len() as usize);
        let mut reader = Box::new(hound::WavReader::open(path).map_err(Box::new)?);
        let spec = reader.spec();
        let duration = reader.duration() as usize;
        reader.seek(0).unwrap();
        if spec.sample_format == hound::SampleFormat::Int {
            Ok(Self {
                spec,
                duration,
                reader: reader.into_inner(),
                bytes_per_sample: (spec.bits_per_sample as usize + 7) / 8,
                bytebuf: Vec::new(),
                current_offset: 0,
                file_size,
            })
        } else {
            Err(Box::new(SourceError::by_reason(
                SourceErrorReason::InvalidFormat,
            )))
        }
    }

    pub const fn file_size(&self) -> Option<usize> {
        self.file_size
    }

    pub fn duration_as_secs(&self) -> f32 {
        self.duration as f32 / self.spec.sample_rate as f32
    }
}

impl Source for HoundSource {
    #[inline]
    fn channels(&self) -> usize {
        self.spec.channels as usize
    }

    #[inline]
    fn bits_per_sample(&self) -> usize {
        self.spec.bits_per_sample as usize
    }

    #[inline]
    fn sample_rate(&self) -> usize {
        self.spec.sample_rate as usize
    }

    #[inline]
    fn read_samples<F: Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError> {
        self.bytebuf.clear();
        let to_read = std::cmp::min(self.duration - self.current_offset, block_size);

        let to_read_bytes = to_read * self.bytes_per_sample * self.channels();
        self.bytebuf.resize(to_read_bytes, 0u8);
        let read_bytes = self
            .reader
            .read(&mut self.bytebuf)
            .map_err(SourceError::from_io_error)?;

        self.current_offset += to_read;
        dest.fill_le_bytes(&self.bytebuf, self.bytes_per_sample)?;

        Ok(read_bytes / self.channels() / self.bytes_per_sample)
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.duration)
    }
}

#[cfg(feature = "sndfile")]
/// An example of `flacenc::source::Source` based on libsndfile.
#[allow(clippy::module_name_repetitions)]
pub struct SndSource {
    snd: sndfile::SndFile,
    len: usize,
    bits_per_sample: usize,
    readbuf: Vec<i32>,
}

#[cfg(feature = "sndfile")]
impl SndSource {
    /// Constructs `SndSource` from `path`.
    ///
    /// # Errors
    ///
    /// The function fails when file is not found or has invalid format. This
    /// function currently do not support WAVs with IEEE float samples, and it
    /// returns `SourceError` with `SourceErrorReason::InvalidFormat` if the
    /// samples are floats.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut snd = sndfile::OpenOptions::ReadOnly(sndfile::ReadOptions::Auto).from_path(path).map_err(|_|
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, "read failed"))
        )?;
        let len = snd.len().map_err(|_|
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, "read length failed"))
        )? as usize;
        snd.seek(std::io::SeekFrom::Start(0));
        let bits_per_sample = match snd.get_subtype_format() {
            sndfile::SubtypeFormat::PCM_S8 => 8,
            sndfile::SubtypeFormat::PCM_16 => 16,
            sndfile::SubtypeFormat::PCM_24 => 24,
            _ => {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "unsupported datatype")));
            },
        };

        Ok(Self {
            snd,
            len,
            bits_per_sample,
            readbuf: Vec::new(),
        })
    }
}

#[cfg(feature = "sndfile")]
impl Source for SndSource {
    #[inline]
    fn channels(&self) -> usize {
        self.snd.get_channels()
    }

    #[inline]
    fn bits_per_sample(&self) -> usize {
        self.bits_per_sample
    }

    #[inline]
    fn sample_rate(&self) -> usize {
        self.snd.get_samplerate()
    }

    #[inline]
    fn read_samples<F: Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, SourceError> {
        use sndfile::SndFileIO;
        let read_size = block_size * self.channels();
        self.readbuf.resize(read_size, 0i32);
        let read_count = self.snd.read_to_slice(&mut self.readbuf).map_err(|_| SourceError::from_unknown())?;
        eprintln!("ch = {}, len = {}, block_size = {}, read_count = {}", self.channels(), self.len, block_size, read_count);
        let shift_size = 32 - self.bits_per_sample();
        for v in &mut self.readbuf {
            *v <<= shift_size;
        }
        dest.fill_interleaved(&self.readbuf[0..read_size]);

        Ok(read_count)
    }

    fn len_hint(&self) -> Option<usize> {
        None // Some(self.len)
    }
}

