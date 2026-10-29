//! One frame's commands: the command buffer every pass records into, the
//! fence that says the GPU finished it, and the timestamps that say where
//! its GPU time went. One frame is in flight at a time: `begin` after
//! `submit` fails until `wait` returns.

use std::time::{Duration, Instant};

use ash::vk;

use crate::gpu;

/// Timestamps a frame may write: the one `begin` writes and the marks.
const QUERIES: u32 = 16;

pub struct Commands {
    // A copy of the owner's device table. The owner keeps the `gpu::Device`
    // alive longer than this (field order), which `drop` relies on.
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
    timestamps: vk::QueryPool,
    /// The label of each mark written this frame, in query order after
    /// `begin`'s.
    marks: Vec<&'static str>,
    /// When the frame was submitted; `None` once `wait` returns.
    submitted: Option<Instant>,
    timestamp_bits: u32,
    timestamp_period: f32,
}

/// Where one frame's time went after recording.
pub struct Times {
    /// How long `wait` blocked on the fence.
    pub wait: Duration,
    /// On the GPU, each mark's span: the work recorded between the previous
    /// timestamp and it, in the order the marks were made.
    pub spans: Vec<(&'static str, Duration)>,
}

impl Commands {
    pub fn new(device: &gpu::Device) -> Result<Self, String> {
        let d = &device.raw;
        let mut c = Self {
            device: d.clone(),
            queue: device.queue,
            pool: vk::CommandPool::null(),
            buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            timestamps: vk::QueryPool::null(),
            marks: Vec::new(),
            submitted: None,
            timestamp_bits: device.timestamp_bits,
            timestamp_period: device.timestamp_period,
        };
        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(device.queue_family);
        // SAFETY (this and every call below in `new`): `d` is a live device,
        // each info struct outlives its call, and each handle passed in was
        // created on `d` just above. Every handle starts null, so a failed
        // `new` drops a partly built value.
        c.pool = unsafe { d.create_command_pool(&pool_info, None) }
            .map_err(|e| format!("vkCreateCommandPool failed: {e}"))?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(c.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        c.buffer = unsafe { d.allocate_command_buffers(&alloc_info) }
            .map_err(|e| format!("vkAllocateCommandBuffers failed: {e}"))?[0];
        c.fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| format!("vkCreateFence failed: {e}"))?;
        let query_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(QUERIES);
        c.timestamps = unsafe { d.create_query_pool(&query_info, None) }
            .map_err(|e| format!("vkCreateQueryPool failed: {e}"))?;
        Ok(c)
    }

    /// Starts recording a frame and returns its command buffer.
    pub fn begin(&mut self) -> Result<vk::CommandBuffer, String> {
        if self.submitted.is_some() {
            return Err("a frame began before the last one's wait".into());
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the pool and buffer were created on this live device, and
        // `wait` waited for the last submit, so neither is in use.
        unsafe {
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(|e| format!("vkResetCommandPool failed: {e}"))?;
            self.device
                .begin_command_buffer(self.buffer, &begin)
                .map_err(|e| format!("vkBeginCommandBuffer failed: {e}"))?;
            self.device
                .cmd_reset_query_pool(self.buffer, self.timestamps, 0, QUERIES);
        }
        self.marks.clear();
        self.timestamp(0);
        Ok(self.buffer)
    }

    /// Names the work recorded since the last mark (or `begin`) `label`,
    /// timing it on the GPU. Call it outside rendering.
    pub fn mark(&mut self, label: &'static str) -> Result<(), String> {
        if self.marks.len() + 1 >= QUERIES as usize {
            return Err(format!("a frame has more than {} marks", QUERIES - 1));
        }
        self.marks.push(label);
        self.timestamp(self.marks.len() as u32);
        Ok(())
    }

    /// Ends the recording and submits it. Call `wait` before touching
    /// anything the frame uses.
    pub fn submit(&mut self) -> Result<(), String> {
        let d = &self.device;
        // SAFETY: the buffer is recording outside rendering, and the fence
        // was created on `d` and is not in use (`begin` checked).
        unsafe {
            d.end_command_buffer(self.buffer)
                .map_err(|e| format!("vkEndCommandBuffer failed: {e}"))?;
            let submit = [vk::CommandBufferSubmitInfo::default().command_buffer(self.buffer)];
            let submits = [vk::SubmitInfo2::default().command_buffer_infos(&submit)];
            d.reset_fences(&[self.fence])
                .map_err(|e| format!("vkResetFences failed: {e}"))?;
            d.queue_submit2(self.queue, &submits, self.fence)
                .map_err(|e| format!("vkQueueSubmit2 failed: {e}"))?;
        }
        self.submitted = Some(Instant::now());
        Ok(())
    }

    /// Waits for the submitted frame and returns where its time went.
    pub fn wait(&mut self) -> Result<Times, String> {
        if self.submitted.is_none() {
            return Err("waited for a frame that was never submitted".into());
        }
        let d = &self.device;
        let start = Instant::now();
        let count = self.marks.len() + 1;
        let mut ticks = vec![0u64; count];
        // SAFETY: the fence and query pool were created on `d`, and the
        // submit that signals the fence writes the first `count` queries.
        unsafe {
            d.wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("vkWaitForFences failed: {e}"))?;
            d.get_query_pool_results(
                self.timestamps,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
            .map_err(|e| format!("vkGetQueryPoolResults failed: {e}"))?;
        }
        let wait = start.elapsed();
        self.submitted = None;
        let spans = (self.marks.iter().zip(ticks.windows(2)))
            .map(|(&label, t)| {
                let span = elapsed(t[0], t[1], self.timestamp_bits, self.timestamp_period);
                (label, span)
            })
            .collect();
        Ok(Times { wait, spans })
    }

    /// Writes timestamp `index` once all earlier commands finish.
    fn timestamp(&mut self, index: u32) {
        // SAFETY: the buffer is recording, the pool was created on this
        // device and reset in `begin`, and `index` is below `QUERIES`.
        unsafe {
            self.device.cmd_write_timestamp2(
                self.buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.timestamps,
                index,
            )
        };
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let d = &self.device;
        // SAFETY: every handle was created on `d` or is null, which destroy
        // calls accept, and waiting for idle means none is still in use.
        unsafe {
            let _ = d.device_wait_idle();
            d.destroy_fence(self.fence, None);
            d.destroy_query_pool(self.timestamps, None);
            d.destroy_command_pool(self.pool, None);
        }
    }
}

/// The time from timestamp `start` to `end` of a counter with `bits` valid
/// bits, ticking every `period` nanoseconds. The counter may wrap between.
fn elapsed(start: u64, end: u64, bits: u32, period: f32) -> Duration {
    let ticks = end.wrapping_sub(start) & (u64::MAX >> (64 - bits));
    Duration::from_nanos((ticks as f64 * period as f64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_scales_ticks_and_survives_a_wrap() {
        let ns = |d: Duration| d.as_nanos();
        assert_eq!(ns(elapsed(1000, 1500, 64, 1.0)), 500);
        assert_eq!(ns(elapsed(1000, 1500, 64, 52.08)), 26_040);
        // A 36-bit counter that wrapped: 16 ticks before and 16 after.
        assert_eq!(ns(elapsed((1 << 36) - 16, 16, 36, 1.0)), 32);
        // The same wrap read without the mask is a huge interval.
        assert!(ns(elapsed((1 << 36) - 16, 16, 64, 1.0)) > 1 << 60);
    }
}
