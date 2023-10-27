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

use super::arrayutils::i32s_to_le_bytes;
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

/// `Arc::into_inner` with unwrapping.
///
/// This function is introduced for conditional compilation for lowering MSRV.
/// The race condition described in the document of [`Arc::try_into`] will
/// not happen in the current use cases; however, [`Arc::into_inner`] will be
/// the future standard so we should delete the second definition when we are
/// ready to bump MSRV.
#[rustversion::since(1.70)]
#[inline]
fn destruct_arc<T: std::fmt::Debug>(ptr: Arc<T>) -> T {
    Arc::into_inner(ptr).expect(panic_msg::ARC_DESTRUCT_FAILED)
}

#[rustversion::before(1.70)]
#[inline]
fn destruct_arc<T: std::fmt::Debug>(ptr: Arc<T>) -> T {
    Arc::try_unwrap(ptr).expect(panic_msg::ARC_DESTRUCT_FAILED)
}

/// Sink object that stores encoding results.
///
/// This is currently just a `BTreeMap<usize, T>` with some utility functions.
#[derive(Debug)]
struct ParSink<T> {
    data: Mutex<BTreeMap<usize, T>>,
}

impl<T> ParSink<T> {
    /// Creates `ParSink` object.
    ///
    /// Note: `BTreeMap::new` isn't const before 1.66.
    pub fn new() -> Self {
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

/// A wrapper that makes the inner `Context` to be filled asynchronously.
struct ParContext {
    inner: Arc<Mutex<Context>>,
    thread_handle: thread::JoinHandle<()>,
    bytebuf: Vec<u8>,
    bytes_per_sample: usize,
    process_queue: (Sender<Vec<u8>>, Receiver<Vec<u8>>),
}

impl ParContext {
    fn new(inner: Context) -> Self {
        let process_queue = crossbeam_channel::bounded(16);
        let bytes_per_sample = inner.bytes_per_sample();
        let inner = Arc::new(Mutex::new(inner));

        let thread_handle = {
            let receiver = process_queue.1.clone();
            let inner = Arc::clone(&inner);
            thread::spawn(move || loop {
                let data: Vec<u8> = receiver.recv().expect(panic_msg::MPMC_RECV_FAILED);
                if data.is_empty() {
                    break;
                }
                let mut inner = inner.lock().expect(panic_msg::MUTEX_LOCK_FAILED);
                inner
                    .fill_le_bytes(&data, bytes_per_sample)
                    .expect(panic_msg::NO_ERROR_EXPECTED);
            })
        };
        Self {
            inner,
            thread_handle,
            bytebuf: vec![],
            bytes_per_sample,
            process_queue,
        }
    }

    fn enqueue_buffer(&self) {
        self.process_queue
            .0
            .send(self.bytebuf.clone())
            .expect(panic_msg::MPMC_SEND_FAILED);
    }

    /// Sends stop signal and returns the number of remaining blocks in queue.
    fn request_stop(&self) -> usize {
        let ret = self.process_queue.0.len();
        self.process_queue
            .0
            .send(vec![])
            .expect(panic_msg::MPMC_SEND_FAILED);
        ret
    }

    fn finalize(self) -> Context {
        self.thread_handle
            .join()
            .expect(panic_msg::THREAD_JOIN_FAILED);
        // since this method is called from the main thread, the race
        // condition as written in the document will never happen. However,
        // anyway the expression above will be the future standard, so this
        // block will be deprecated when we are to bump MSRV.
        destruct_arc(self.inner).into_inner().unwrap()
    }
}

impl Fill for ParContext {
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError> {
        let bps = self.bytes_per_sample;
        self.bytebuf.resize(interleaved.len() * bps, 0u8);
        i32s_to_le_bytes(interleaved, &mut self.bytebuf, bps);
        self.enqueue_buffer();
        Ok(())
    }

    fn fill_le_bytes(&mut self, bytes: &[u8], _bytes_per_sample: usize) -> Result<(), SourceError> {
        self.bytebuf.clear();
        self.bytebuf.extend_from_slice(bytes);
        self.enqueue_buffer();
        Ok(())
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
    let context = ParContext::new(Context::new(
        src.bits_per_sample(),
        src.channels(),
        block_size,
    ));
    let (feed_stats, context) =
        feed_fixed_block_size(src, block_size, worker_count, &parbuf, context)?;
    let remaining_md5_blocks = context.request_stop();
    let context = context.finalize();

    info!(
        target: "flacenc::par::jsonl",
        "{{ worker_count: {}, frame_count: {}, worker_starvation_count: {}, md5_overdue: {} }}",
        worker_count,
        feed_stats.frame_count,
        feed_stats.worker_starvation_count,
        remaining_md5_blocks,
    );

    stream
        .stream_info_mut()
        .set_md5_digest(&context.md5_digest());

    for h in join_handles {
        h.join().expect(panic_msg::THREAD_JOIN_FAILED);
    }

    destruct_arc(parsink).finalize(|f: Frame| stream.add_frame(f));

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

        for _t in 0..30 {
            // wait 30 * 0.1 = 3 sec at maximum.
            if pfb.encode_queue.1.len() == 3 {
                break;
            }
            thread::sleep(std::time::Duration::from_secs_f32(0.1));
        }
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
        for _t in 0..30 {
            // wait 30 * 0.1 = 3 sec at maximum.
            if pfb.encode_queue.1.len() == workers {
                break;
            }
            thread::sleep(std::time::Duration::from_secs_f32(0.1));
        }
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
        destruct_arc(sink).finalize(|v| result.push(v));
        assert_eq!(result, REFERENCE);
    }
}
