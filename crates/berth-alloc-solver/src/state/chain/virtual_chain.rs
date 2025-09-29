// Copyright (c) 2025 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use crate::state::chain::delta::ChainDelta;

#[derive(Clone, Debug)]
pub struct VirtualChain<'a> {
    base_next: &'a [usize],
    delta: &'a ChainDelta,
}

impl<'a> VirtualChain<'a> {
    #[inline]
    pub fn next(&self, i: usize) -> usize {
        self.delta.next_after(self.base_next, i)
    }

    #[inline]
    pub fn changed(&self, i: usize) -> bool {
        self.delta.changed(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_is_visible_through_virtual_chain() {
        let base = vec![0, 1, 2, 3, 4];
        let mut cd = ChainDelta::new();

        // Apply an override; then create the VC view
        cd.push(2, 999, 42); // override tail 2 -> 42

        let vc = VirtualChain {
            base_next: &base,
            delta: &cd,
        };

        assert!(vc.changed(2));
        assert_eq!(vc.next(2), 42);

        // Unaffected indices pass through to base
        assert!(!vc.changed(1));
        assert_eq!(vc.next(1), 1);
        assert!(!vc.changed(3));
        assert_eq!(vc.next(3), 3);
    }

    #[test]
    fn last_wins_for_same_tail() {
        let base = vec![0, 10, 20, 30, 40];
        let mut cd = ChainDelta::new();

        cd.push(3, 100, 77);
        {
            let vc = VirtualChain {
                base_next: &base,
                delta: &cd,
            };
            assert_eq!(vc.next(3), 77);
            assert!(vc.changed(3));
        }

        // Update the same tail again; latest should win
        cd.push(3, 101, 88);
        {
            let vc = VirtualChain {
                base_next: &base,
                delta: &cd,
            };
            assert_eq!(vc.next(3), 88);
            assert!(vc.changed(3));
        }
    }

    #[test]
    fn touch_many_has_no_effect_on_virtual_chain_view() {
        let base = vec![5, 6, 7];
        let mut cd = ChainDelta::new();

        // Mark nodes as touched, but do not set overrides
        cd.touch_many(&[0, 1, 2, 2, 0]);

        let vc = VirtualChain {
            base_next: &base,
            delta: &cd,
        };

        // VirtualChain.changed is about overrides; should remain false
        for i in 0..base.len() {
            assert!(!vc.changed(i));
            assert_eq!(
                vc.next(i),
                base[i],
                "no override present, base should be used"
            );
        }
    }

    #[test]
    fn clear_resets_overrides_and_passthrough_resumes() {
        let base = vec![10, 11, 12, 13];
        let mut cd = ChainDelta::new();

        cd.push(1, 11, 99);
        {
            let vc = VirtualChain {
                base_next: &base,
                delta: &cd,
            };
            assert!(vc.changed(1));
            assert_eq!(vc.next(1), 99);
        }

        cd.clear();

        {
            let vc = VirtualChain {
                base_next: &base,
                delta: &cd,
            };
            // After clear, changed should be false and next should use base
            assert!(!vc.changed(1));
            assert_eq!(vc.next(1), 11);
        }
    }

    #[test]
    fn override_beyond_base_length_returns_override_without_panic() {
        // Base only has indices 0..=2
        let base = vec![100, 200, 300];
        let mut cd = ChainDelta::new();

        // Set override at index 10, which is beyond base.len()
        cd.push(10, 0, 777);

        let vc = VirtualChain {
            base_next: &base,
            delta: &cd,
        };

        // changed should be true, and next should return override (no base access/panic)
        assert!(vc.changed(10));
        assert_eq!(vc.next(10), 777);

        // In-bounds indices still use base
        assert!(!vc.changed(2));
        assert_eq!(vc.next(2), 300);
    }

    #[test]
    fn reserve_nodes_does_not_mark_changed_and_does_not_panic() {
        let base = (0..=5).collect::<Vec<usize>>();
        let mut cd = ChainDelta::new();

        cd.reserve_nodes(50);

        let vc = VirtualChain {
            base_next: &base,
            delta: &cd,
        };

        // Do not call next(> base.len()-1) to avoid OOB; changed should remain false
        assert!(!vc.changed(50));

        // In-bounds values should pass through
        for i in 0..base.len() {
            assert_eq!(vc.next(i), base[i]);
            assert!(!vc.changed(i));
        }
    }

    #[test]
    fn multiple_overrides_and_unaffected_indices() {
        let base = vec![0, 1, 2, 3, 4, 5, 6];
        let mut cd = ChainDelta::new();

        cd.push(0, 0, 10);
        cd.push(3, 3, 30);
        cd.push(6, 6, 60);

        let vc = VirtualChain {
            base_next: &base,
            delta: &cd,
        };

        assert_eq!(vc.next(0), 10);
        assert_eq!(vc.next(3), 30);
        assert_eq!(vc.next(6), 60);
        assert!(vc.changed(0) && vc.changed(3) && vc.changed(6));

        // Others unchanged
        for i in [1, 2, 4, 5] {
            assert!(!vc.changed(i));
            assert_eq!(vc.next(i), base[i]);
        }
    }

    #[test]
    fn cloned_virtual_chain_observes_mutations() {
        let base = vec![9, 8, 7, 6];
        let mut cd = ChainDelta::new();

        // Phase 1: before mutation
        {
            let vc1 = VirtualChain {
                base_next: &base,
                delta: &cd,
            };
            let vc2 = vc1.clone();
            assert!(!vc1.changed(2) && !vc2.changed(2));
            assert_eq!(vc1.next(2), 7);
            assert_eq!(vc2.next(2), 7);
        }

        // Mutate
        cd.push(2, 0, 123);

        // Phase 2: after mutation
        {
            let vc1 = VirtualChain {
                base_next: &base,
                delta: &cd,
            };
            let vc2 = vc1.clone();
            assert!(vc1.changed(2) && vc2.changed(2));
            assert_eq!(vc1.next(2), 123);
            assert_eq!(vc2.next(2), 123);
        }

        // Clear and verify reverted
        cd.clear();

        {
            let vc1 = VirtualChain {
                base_next: &base,
                delta: &cd,
            };
            let vc2 = vc1.clone();
            assert!(!vc1.changed(2) && !vc2.changed(2));
            assert_eq!(vc1.next(2), 7);
            assert_eq!(vc2.next(2), 7);
        }
    }

    #[test]
    fn big_index_override_sequence_and_last_wins() {
        let base = vec![0, 1, 2];
        let mut cd = ChainDelta::new();

        cd.push(1000, 0, 1);
        {
            let vc = VirtualChain {
                base_next: &base,
                delta: &cd,
            };
            assert!(vc.changed(1000));
            assert_eq!(vc.next(1000), 1);
        }

        cd.push(1000, 0, 2);
        {
            let vc = VirtualChain {
                base_next: &base,
                delta: &cd,
            };
            assert!(vc.changed(1000));
            assert_eq!(vc.next(1000), 2);

            // Unrelated in-bounds still use base
            assert!(!vc.changed(1));
            assert_eq!(vc.next(1), 1);
        }
    }
}
