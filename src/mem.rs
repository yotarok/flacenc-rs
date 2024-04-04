use std::cell::RefCell;
use std::cell::Ref;
use std::cell::RefMut;

struct Slice<'a, T> {
    arena: &'a SliceArena<T>,
    offset: usize,
    size: usize,
}

struct SliceArena<T> {
    storage: RefCell<Vec<T>>,
    head: std::sync::atomic::AtomicUsize,
}

impl<T> Slice<'_, T> {
    fn get(&self) -> Ref<'_, [T]> {
        self.arena.slice(self.offset, self.size)
    }
    fn get_mut(&self) -> RefMut<'_, [T]> {
        self.arena.slice_mut(self.offset, self.size)
    }
}


impl<T> SliceArena<T>
{
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            storage: RefCell::new(Vec::with_capacity(capacity)),
            head: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn slice(&self, offset: usize, size: usize) -> Ref<'_, [T]> {
        Ref::map(self.storage.borrow(), |x| &x[offset..(offset + size)])
    }

    pub fn slice_mut(&self, offset: usize, size: usize) -> RefMut<'_, [T]> {
        RefMut::map(self.storage.borrow_mut(), |x: &mut Vec<T>| &mut x[offset..(offset + size)])
    }
}

impl<T> SliceArena<T>
where
    T: Default + Clone,
{
    pub fn alloc(&self, size: usize) -> Slice<T> {
        // Slice has an implicit lifetime parameter that extends the
        // lifetime of `self`. so this cannot be mut. If it is mut we can't
        // allocate multiple slices.
        let target_len = self.head.load(std::sync::atomic::Ordering::Relaxed) + size;
        if target_len > self.storage.borrow().len() {
            self.storage.borrow_mut().resize(target_len.next_power_of_two(), T::default());
        }
        let offset = self.head.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        Slice {
            arena: self,
            offset,
            size,
        }
    }

}

mod tests {
    use super::*;

    #[test]
    fn alloc_and_mutate() {
        let arena = SliceArena::<u8>::with_capacity(2);
        let aslice = arena.alloc(8);
        assert_eq!(aslice.get().as_ref(), &[0, 0, 0, 0, 0, 0, 0, 0]);
        let aslice2 = arena.alloc(64);
        // storage cell is mut-borrwed here.
        aslice2.get_mut()[2] = 5;
        // storage cell is released here.

        // storage cell is mut-borrwed here.
        aslice.get_mut()[2] = 3;
        // storage cell is released here.
        assert_eq!(aslice.get().as_ref(), &[0, 0, 3, 0, 0, 0, 0, 0]);

    }
}
