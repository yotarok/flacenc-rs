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

//! This module is for parallel encoding. Only compiled when "par" feature is enabled.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use super::coding;
use super::component::Frame;
use super::component::Stream;
use super::config;
use super::constant;
use super::constant::envvar_key;
use super::constant::panic_msg;
use super::error::EncodeError;
use super::error::SourceError;
use super::source::Context;
use super::source::Fill;
use super::source::FrameBuf;
use super::source::Source;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;

/// Sink object that stores encoding results.
///
/// This is currently just a `BTreeMap<usize, T>` with some utility functions.
struct ParSink<T> {
    data: Mutex<BTreeMap<usize, T>>,
}

impl<T> ParSink<T> {
    /// Creates `ParSink` object.
    pub const fn new() -> Self {
        Self {
            data: Mutex::new(BTreeMap::new()),
        }
    }

    /// Stores a computation result `element` with a serial id `idx`.
    pub fn push(&self, idx: usize, element: T) {
        let mut data = self.data.lock().expect(panic_msg::MUTEX_LOCK_FAILED);
        data.insert(idx, element);
    }

    /// Consumes `self` and calls `f` in the order of the serial id.
    pub fn finalize<F>(self, f: F)
    where
        F: FnMut(T),
    {
        let data = self.data.into_inner().expect(panic_msg::MUTEX_DROP_FAILED);
        data.into_values().for_each(f);
    }
}

/// Internal struct for tying `FrameBuf` with the current frame number.
struct NumberedFrameBuf {
    framebuf: FrameBuf,
    frame_number: Option<usize>,
}

pub struct FeedStats {
    pub frame_count: usize,
    pub worker_starvation_count: usize,
}

/// Parallel `FrameBuf`.
struct ParFrameBuf {
    buffers: Vec<Mutex<NumberedFrameBuf>>,
    encode_queue: (Sender<Option<usize>>, Receiver<Option<usize>>),
    refill_queue: (Sender<usize>, Receiver<usize>),
}

impl ParFrameBuf {
    /// Creates new parallel frame buffers.
    pub fn new(replicas: usize, channels: usize, block_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(replicas);
        for _t in 0..replicas {
            let buf = Mutex::new(NumberedFrameBuf {
                framebuf: FrameBuf::with_size(channels, block_size),
                frame_number: None,
            });
            buffers.push(buf);
        }
        let (refill_sender, refill_receiver) = crossbeam_channel::bounded(replicas + 1);

        (0..replicas).for_each(|t| {
            refill_sender.send(t).expect(panic_msg::MPMC_SEND_FAILED);
        });
        Self {
            buffers,
            encode_queue: crossbeam_channel::bounded(replicas + 1),
            refill_queue: (refill_sender, refill_receiver),
        }
    }

    /// Gets the id for `FrameBuf` to be encoded first.
    ///
    /// If this returns None, workder thread must immediately stop.
    #[inline]
    pub fn pop_encode_queue(&self) -> Option<usize> {
        self.encode_queue
            .1
            .recv()
            .expect(panic_msg::MPMC_RECV_FAILED)
    }

    /// Locks `FrameBuf` with the specified id and returns `MutexGuard`.
    #[inline]
    pub fn lock_buffer(&self, bufid: usize) -> std::sync::MutexGuard<'_, NumberedFrameBuf> {
        self.buffers[bufid]
            .lock()
            .expect(panic_msg::MUTEX_LOCK_FAILED)
    }

    /// Requests refill for `FrameBuf` with the specified id.
    #[inline]
    pub fn enqueue_refill(&self, bufid: usize) {
        self.refill_queue
            .0
            .send(bufid)
            .expect(panic_msg::MPMC_SEND_FAILED);
    }

    #[inline]
    pub fn recv_refill_request(&self) -> usize {
        self.refill_queue
            .1
            .recv()
            .expect(panic_msg::MPMC_RECV_FAILED)
    }

    #[inline]
    pub fn enqueue_encode(&self, bufid: usize) -> bool {
        let starved = self.encode_queue.0.is_empty();
        self.encode_queue
            .0
            .send(Some(bufid))
            .expect(panic_msg::MPMC_SEND_FAILED);
        starved
    }

    #[inline]
    pub fn request_stop(&self, workers: usize) {
        for _i in 0..workers {
            self.encode_queue
                .0
                .send(None)
                .expect(panic_msg::MPMC_SEND_FAILED);
        }
    }
}

/// Reads from source, feeds samples to buffers, and enqueues.
///
/// This function is intended to be called from the main (single) thread. This
/// function consumes the initial context value (`context`) and returns the
/// updated context.
///
/// # Errors
///
/// It propagates errors from `Source::read_samples`.
fn feed_fixed_block_size<T: Source, C: Fill>(
    src: T,
    block_size: usize,
    workers: usize,
    parbuf: &ParFrameBuf,
    mut context: C,
) -> Result<(FeedStats, C), SourceError> {
    let mut src = src;
    let mut frame_count = 0usize;
    let mut worker_starvation_count = 0usize;

    'feed: loop {
        let bufid = parbuf.recv_refill_request();
        {
            let mut numbuf = parbuf.buffers[bufid]
                .lock()
                .expect(panic_msg::MUTEX_LOCK_FAILED);
            let mut framebuf_and_ctx = (&mut numbuf.framebuf, &mut context);
            let read_samples = src.read_samples(block_size, &mut framebuf_and_ctx)?;
            if read_samples == 0 {
                break 'feed;
            }
            numbuf.frame_number = Some(frame_count);
        }
        frame_count += 1;
        if parbuf.enqueue_encode(bufid) {
            worker_starvation_count += 1;
        }
    }
    parbuf.request_stop(workers);
    Ok((
        FeedStats {
            frame_count,
            worker_starvation_count,
        },
        context,
    ))
}

/// Determines worker counts considering various cues.
fn determine_worker_count(config: &config::Encoder) -> Result<usize, SourceError> {
    let default_parallelism = std::thread::available_parallelism()
        .map_err(SourceError::from_io_error)?
        .get();
    let default_parallelism = std::env::var(envvar_key::DEFAULT_PARALLELISM)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default_parallelism);
    Ok(config
        .workers
        .map_or(default_parallelism, NonZeroUsize::get))
}

struct ParRunStats {
    worker_count: usize,
    feed: FeedStats,
}

impl ParRunStats {
    pub const fn new(worker_count: usize, feed: FeedStats) -> Self {
        Self { worker_count, feed }
    }
}

impl std::fmt::Display for ParRunStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "
worker_count = {}
frames_processed = {}
worker_starvation_count = {}
",
            self.worker_count, self.feed.frame_count, self.feed.worker_starvation_count,
        )
    }
}

/// Parallel version of `encode_with_fixed_block_size`.
///
/// This function is internally called by `encode_with_fixed_block_size`
/// when `config.multithread == true`. However, one can explicitly call this
/// function and disable single-threaded mode.
///
/// # Errors
///
/// This function returns `SourceError` when it failed to read samples from `src`.
///
/// # Panics
///
/// This function panics when an internal error regarding inter-thread
/// communication.
pub fn encode_with_fixed_block_size<T: Source>(
    config: &config::Encoder,
    src: T,
    block_size: usize,
) -> Result<Stream, EncodeError> {
    let config = Arc::new(config.clone());
    let mut stream = Stream::new(src.sample_rate(), src.channels(), src.bits_per_sample());
    let worker_count = determine_worker_count(&config)?;
    let parbuf = Arc::new(ParFrameBuf::new(
        worker_count * constant::par::FRAMEBUF_MULTIPLICITY,
        src.channels(),
        block_size,
    ));
    let parsink: Arc<ParSink<Frame>> = Arc::new(ParSink::new());

    let join_handles: Vec<_> = (0..worker_count)
        .map(|_n| {
            let parbuf = Arc::clone(&parbuf);
            let parsink = Arc::clone(&parsink);
            let stream_info = stream.stream_info().clone();
            let config = Arc::clone(&config);
            thread::spawn(move || {
                while let Some(bufid) = parbuf.pop_encode_queue() {
                    let (frame_number, encode_result) = {
                        let numbuf = &parbuf.lock_buffer(bufid);
                        let frame_number = numbuf.frame_number.expect(panic_msg::FRAMENUM_NOT_SET);
                        (
                            frame_number,
                            coding::encode_fixed_size_frame(
                                &config,
                                &numbuf.framebuf,
                                frame_number,
                                &stream_info,
                            ),
                        )
                    };
                    encode_result.map_or_else(
                        |e| {
                            unreachable!("{}, err={:?}", panic_msg::ERROR_NOT_EXPECTED, e);
                        },
                        |mut frame| {
                            parbuf.enqueue_refill(bufid);
                            frame.precompute_bitstream();
                            parsink.push(frame_number, frame);
                        },
                    );
                }
            })
        })
        .collect();

    let src_len_hint = src.len_hint();
    let context = Context::new(src.bits_per_sample(), src.channels(), block_size);
    let (feed_stats, context) =
        feed_fixed_block_size(src, block_size, worker_count, &parbuf, context)?;

    if let Ok(path_str) = std::env::var(envvar_key::RUNSTATS_OUTPUT) {
        let run_stat = ParRunStats::new(worker_count, feed_stats);
        std::fs::write(path_str, format!("{run_stat}")).unwrap();
    }

    stream
        .stream_info_mut()
        .set_md5_digest(&context.md5_digest());

    for h in join_handles {
        h.join().expect(panic_msg::THREAD_JOIN_FAILED);
    }

    Arc::into_inner(parsink)
        .unwrap()
        .finalize(|f: Frame| stream.add_frame(f));

    stream
        .stream_info_mut()
        .set_total_samples(src_len_hint.unwrap_or_else(|| context.total_samples()));

    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source;
    // use crate::test_helper;

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_determine_worker_count() {
        // manually set by config
        let mut config = config::Encoder::default();
        config.workers = NonZeroUsize::new(8);
        let result = determine_worker_count(&config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 8);

        // default
        let config = config::Encoder::default();
        let result = determine_worker_count(&config);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            std::thread::available_parallelism().unwrap().into()
        );
    }

    #[test]
    fn par_frame_buf_feeding() {
        let channels = 2;
        let block_size = 100;
        let replicas = 3;
        let workers = 4;
        let pfb = Arc::new(ParFrameBuf::new(replicas, channels, block_size));
        // nothing is ready
        assert!(pfb.encode_queue.1.is_empty());

        let total_size = replicas * block_size * 2;
        {
            let pfb = Arc::clone(&pfb);
            thread::spawn(move || {
                let mut signal = vec![];
                for t in 0..total_size {
                    for ch in 0..channels {
                        let sign: i32 = if ch == 0 { 1 } else { -1 };
                        signal.push(sign * (t as i32 % 256));
                    }
                }
                let src = source::MemSource::from_samples(&signal, channels, 16, 16000);
                let mut ctx = Context::new(16, channels, block_size);
                feed_fixed_block_size(src, block_size, workers, &pfb, &mut ctx)
                    .expect("Feeding failed");
            });
        }
        thread::sleep(std::time::Duration::from_secs_f32(0.1));
        assert_eq!(pfb.encode_queue.1.len(), 3);
        for _t in 0..6 {
            let received = pfb
                .encode_queue
                .1
                .recv()
                .expect(panic_msg::MPMC_RECV_FAILED);
            assert!(received.is_some());
            let bufid = received.unwrap();
            pfb.enqueue_refill(bufid);
        }
        thread::sleep(std::time::Duration::from_secs_f32(0.1));
        assert_eq!(pfb.encode_queue.1.len(), workers); // stop signal.
        for _t in 0..workers {
            let received = pfb
                .encode_queue
                .1
                .recv()
                .expect(panic_msg::MPMC_RECV_FAILED);
            assert!(received.is_none());
        }
    }

    #[test]
    fn par_sink_finalization() {
        const REFERENCE: [&str; 5] = ["ParSink", "sorts", "randomly", "ordered", "elems"];
        let sink = Arc::new(ParSink::new());
        let handles = (0..REFERENCE.len()).map(|t| {
            let sink = Arc::clone(&sink);
            thread::spawn(move || sink.push(t, REFERENCE[t]))
        });
        for h in handles {
            h.join().expect(panic_msg::THREAD_JOIN_FAILED);
        }
        let mut result = vec![];
        Arc::into_inner(sink)
            .expect("Arc deconstruction failed")
            .finalize(|v| result.push(v));
        assert_eq!(result, REFERENCE);
    }
}
