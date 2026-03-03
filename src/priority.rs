//! Scope-based thread priority management for long-running operations
//!
//! On Windows, sets thread priority to IDLE for the lifetime of the PriorityManager,
//! restoring to NORMAL on drop. This makes long-running operations less intrusive.

#[cfg(windows)]
mod windows_impl {
    use windows::Win32::System::Threading::{
        GetCurrentThread, GetThreadPriority, SetThreadPriority, THREAD_PRIORITY, THREAD_PRIORITY_IDLE,
    };

    pub struct PriorityManager {
        original_priority: THREAD_PRIORITY,
    }

    impl PriorityManager {
        pub fn new() -> Self {
            unsafe {
                let thread = GetCurrentThread();
                let original_priority = THREAD_PRIORITY(GetThreadPriority(thread));
                let _ = SetThreadPriority(thread, THREAD_PRIORITY_IDLE);
                Self { original_priority }
            }
        }
    }

    impl Drop for PriorityManager {
        fn drop(&mut self) {
            unsafe {
                let thread = GetCurrentThread();
                let _ = SetThreadPriority(thread, self.original_priority);
            }
        }
    }
}

#[cfg(not(windows))]
mod windows_impl {
    pub struct PriorityManager;

    impl PriorityManager {
        #[inline]
        pub fn new() -> Self {
            Self
        }
    }
}

pub use windows_impl::PriorityManager;
