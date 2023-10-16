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

/// Sets up the thread-local re-usable storage for avoiding reallocation.
///
/// This provides a short-cut for the common pattern using [`thread_local!`]
/// and [`RefCell`].  Currently, this is just for removing small repetition in
/// code.
///
/// [`RefCell`]: std::cell::RefCell
#[macro_export]
macro_rules! reusable {
    ($key:ident: $t:ty) => {
        thread_local! {
            static $key: std::cell::RefCell<$t> = std::cell::RefCell::new(Default::default());
        }
    };
    ($key:ident: $t:ty = $init:expr) => {
        thread_local! {
            static $key: std::cell::RefCell<$t> = std::cell::RefCell::new($init);
        }
    };
}

/// Macro used when using a storage declared using [`reusable!`].
#[macro_export]
macro_rules! reuse {
    ($key:ident, $fn:expr) => {{
        #[allow(clippy::redundant_closure_call)]
        $key.with(|cell| $fn(&mut cell.borrow_mut()))
    }};
}

mod tests {
    reusable!(REUSABLE_BUF: Vec<i32>);

    #[test]
    fn call_twice() {
        fn fn1() {
            reuse!(REUSABLE_BUF, |buf: &mut Vec<i32>| {
                assert_eq!(buf.len(), 0);
                buf.resize(5, 0i32);
            });
        }

        fn fn2() {
            reuse!(REUSABLE_BUF, |buf: &mut Vec<i32>| {
                assert_eq!(buf.len(), 5);
            });
        }

        fn1();
        fn2();
    }
}
