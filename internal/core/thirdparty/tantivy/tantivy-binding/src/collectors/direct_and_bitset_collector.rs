use tantivy::{
    collector::{Collector, SegmentCollector},
    postings::SegmentPostings,
    schema::IndexRecordOption,
    DocId, DocSet, Score, SegmentOrdinal, SegmentReader, Term, TERMINATED,
};

use crate::bitset_wrapper::BitsetWrapper;

/// A collector that directly writes intersection (AND) results to a bitset.
///
/// This collector bypasses Tantivy's internal intersection structures and
/// writes matching document IDs directly to an external bitset, avoiding
/// the overhead of intermediate data structures.
///
/// The implementation is based on Tantivy's `Intersection` algorithm:
/// 1. Sort posting lists by size (smallest first)
/// 2. Use the two smallest as primary drivers
/// 3. Find intersection by alternating seeks between drivers
/// 4. Validate remaining posting lists contain the candidate
pub(crate) struct DirectAndBitsetCollector {
    pub(crate) bitset_wrapper: BitsetWrapper,
    pub(crate) terms: Vec<Term>,
    /// Maximum number of terms to use for intersection.
    /// If None, all terms are used.
    /// If Some(n), only the n rarest terms (by doc_freq) are used.
    pub(crate) max_terms: Option<usize>,
}

pub(crate) struct DirectAndBitsetChildCollector {}

impl Collector for DirectAndBitsetCollector {
    type Fruit = ();
    type Child = DirectAndBitsetChildCollector;

    fn collect_segment(
        &self,
        _weight: &dyn tantivy::query::Weight,
        _segment_ord: u32,
        reader: &SegmentReader,
    ) -> tantivy::Result<<Self::Child as SegmentCollector>::Fruit> {
        // Edge case: no terms means no results
        if self.terms.is_empty() {
            return Ok(());
        }

        // 1. Get doc_freq for each term and select the rarest ones
        let terms_to_use: Vec<Term> = if let Some(max_terms) = self.max_terms {
            if self.terms.len() <= max_terms {
                self.terms.clone()
            } else {
                // Get doc_freq for each term
                let mut term_with_freq: Vec<(Term, u32)> = Vec::with_capacity(self.terms.len());
                for term in self.terms.iter() {
                    let inv_index = reader.inverted_index(term.field())?;
                    let doc_freq = inv_index.doc_freq(term)?;
                    if doc_freq == 0 {
                        // Term doesn't exist, intersection is empty
                        return Ok(());
                    }
                    term_with_freq.push((term.clone(), doc_freq));
                }

                // Sort by doc_freq (rarest first)
                term_with_freq.sort_by_key(|(_, freq)| *freq);

                // Take only the rarest N terms
                term_with_freq
                    .into_iter()
                    .take(max_terms)
                    .map(|(term, _)| term)
                    .collect()
            }
        } else {
            self.terms.clone()
        };

        // 2. Collect posting lists for selected terms
        let mut postings: Vec<SegmentPostings> = Vec::with_capacity(terms_to_use.len());
        for term in terms_to_use.iter() {
            let inv_index = reader.inverted_index(term.field())?;
            match inv_index.read_postings(term, IndexRecordOption::Basic)? {
                Some(posting) => postings.push(posting),
                None => {
                    // If any term has no posting list, intersection is empty
                    return Ok(());
                }
            }
        }

        // 3. Handle single term case (degenerate to OR behavior)
        if postings.len() == 1 {
            self.collect_single_posting(&mut postings[0]);
            return Ok(());
        }

        // 4. Sort by size_hint (smallest first for better performance)
        postings.sort_by_key(|p| p.size_hint());

        // 5. Find first common document
        let first_doc = go_to_first_doc(&mut postings);
        if first_doc == TERMINATED {
            return Ok(());
        }

        // 5. Split into left, right, and others
        let mut left = postings.remove(0);
        let mut right = postings.remove(0);
        let mut others = postings;

        // 6. Collect intersection documents
        let mut buffer = [0u32; 4096];
        let mut len = 0;

        // First document is already found
        buffer[len] = left.doc();
        len += 1;

        // Iterate to find remaining intersection documents
        loop {
            let doc = advance_intersection(&mut left, &mut right, &mut others);
            if doc == TERMINATED {
                break;
            }

            buffer[len] = doc;
            len += 1;

            if len == 4096 {
                self.bitset_wrapper.batch_set(&buffer[..len]);
                len = 0;
            }
        }

        // Flush remaining buffer
        if len > 0 {
            self.bitset_wrapper.batch_set(&buffer[..len]);
        }

        Ok(())
    }

    fn for_segment(
        &self,
        _segment_local_id: SegmentOrdinal,
        _segment: &SegmentReader,
    ) -> tantivy::Result<Self::Child> {
        Ok(DirectAndBitsetChildCollector {})
    }

    fn merge_fruits(
        &self,
        _segment_fruits: Vec<<Self::Child as SegmentCollector>::Fruit>,
    ) -> tantivy::Result<Self::Fruit> {
        Ok(())
    }

    fn requires_scoring(&self) -> bool {
        false
    }
}

impl DirectAndBitsetCollector {
    /// Collect all documents from a single posting list
    fn collect_single_posting(&self, posting: &mut SegmentPostings) {
        let mut buffer = [0u32; 4096];
        while posting.doc() != TERMINATED {
            let mut len = 0;
            while posting.doc() != TERMINATED && len < 4096 {
                buffer[len] = posting.doc();
                len += 1;
                posting.advance();
            }
            self.bitset_wrapper.batch_set(&buffer[..len]);
        }
    }
}

/// Find the first document present in all posting lists.
///
/// Algorithm:
/// 1. Start with the maximum current doc across all posting lists
/// 2. Seek each posting list to that candidate
/// 3. If any posting list overshoots, update candidate and restart
/// 4. When all posting lists point to the same doc, return it
fn go_to_first_doc(postings: &mut [SegmentPostings]) -> DocId {
    debug_assert!(!postings.is_empty());

    // Start with the maximum current position
    let mut candidate = postings.iter().map(|p| p.doc()).max().unwrap();

    'outer: loop {
        if candidate == TERMINATED {
            return TERMINATED;
        }

        for posting in postings.iter_mut() {
            let seek_doc = posting.seek(candidate);
            if seek_doc > candidate {
                // This posting list doesn't contain candidate, try the new position
                candidate = seek_doc;
                continue 'outer;
            }
        }

        // All posting lists point to candidate
        return candidate;
    }
}

/// Advance to the next document in the intersection.
///
/// This implements Tantivy's two-phase intersection algorithm:
/// 1. Find a common document between left and right (the two smallest posting lists)
/// 2. Verify that all other posting lists also contain this document
/// 3. If not, continue searching
fn advance_intersection(
    left: &mut SegmentPostings,
    right: &mut SegmentPostings,
    others: &mut [SegmentPostings],
) -> DocId {
    let mut candidate = left.advance();
    if candidate == TERMINATED {
        return TERMINATED;
    }

    'outer: loop {
        // Phase 1: Find intersection between left and right
        loop {
            let right_doc = right.seek(candidate);
            if right_doc == TERMINATED {
                return TERMINATED;
            }
            candidate = left.seek(right_doc);
            if candidate == TERMINATED {
                return TERMINATED;
            }
            if candidate == right_doc {
                break;
            }
        }

        // Phase 2: Verify all others contain candidate
        for posting in others.iter_mut() {
            let seek_doc = posting.seek(candidate);
            if seek_doc > candidate {
                // This posting doesn't contain candidate, continue with new position
                candidate = left.seek(seek_doc);
                if candidate == TERMINATED {
                    return TERMINATED;
                }
                continue 'outer;
            }
        }

        // All posting lists contain candidate
        return candidate;
    }
}

impl SegmentCollector for DirectAndBitsetChildCollector {
    type Fruit = ();

    fn collect(&mut self, _doc: DocId, _score: Score) {
        unreachable!("DirectAndBitsetChildCollector uses collect_segment directly");
    }

    fn collect_block(&mut self, _docs: &[DocId]) {
        unreachable!("DirectAndBitsetChildCollector uses collect_segment directly");
    }

    fn harvest(self) -> Self::Fruit {}
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::ffi::c_void;
    use tempfile::TempDir;

    use crate::{index_writer::IndexWriterWrapper, util::set_bitset, TantivyIndexVersion};

    #[test]
    fn test_and_match_query_basic() {
        let dir = TempDir::new().unwrap();
        let mut writer = IndexWriterWrapper::create_text_writer(
            "text",
            dir.path().to_str().unwrap(),
            "default",
            "",
            1,
            50_000_000,
            false,
            TantivyIndexVersion::default_version(),
        )
        .unwrap();

        // doc 0: contains "hello" and "world"
        // doc 1: contains only "hello"
        // doc 2: contains only "world"
        // doc 3: contains "hello" and "world"
        writer.add("hello world", Some(0)).unwrap();
        writer.add("hello", Some(1)).unwrap();
        writer.add("world", Some(2)).unwrap();
        writer.add("hello world again", Some(3)).unwrap();
        writer.commit().unwrap();

        let reader = writer.create_reader(set_bitset).unwrap();

        // AND query: "hello" AND "world" should return docs 0 and 3
        let mut res: HashSet<u32> = HashSet::new();
        reader
            .and_match_query("hello world", &mut res as *mut _ as *mut c_void)
            .unwrap();

        assert_eq!(res, vec![0, 3].into_iter().collect::<HashSet<u32>>());
    }

    #[test]
    fn test_and_match_query_no_intersection() {
        let dir = TempDir::new().unwrap();
        let mut writer = IndexWriterWrapper::create_text_writer(
            "text",
            dir.path().to_str().unwrap(),
            "default",
            "",
            1,
            50_000_000,
            false,
            TantivyIndexVersion::default_version(),
        )
        .unwrap();

        writer.add("apple", Some(0)).unwrap();
        writer.add("banana", Some(1)).unwrap();
        writer.commit().unwrap();

        let reader = writer.create_reader(set_bitset).unwrap();

        // AND query with no intersection
        let mut res: HashSet<u32> = HashSet::new();
        reader
            .and_match_query("apple banana", &mut res as *mut _ as *mut c_void)
            .unwrap();

        assert!(res.is_empty());
    }

    #[test]
    fn test_and_match_query_single_term() {
        let dir = TempDir::new().unwrap();
        let mut writer = IndexWriterWrapper::create_text_writer(
            "text",
            dir.path().to_str().unwrap(),
            "default",
            "",
            1,
            50_000_000,
            false,
            TantivyIndexVersion::default_version(),
        )
        .unwrap();

        writer.add("hello world", Some(0)).unwrap();
        writer.add("hello", Some(1)).unwrap();
        writer.commit().unwrap();

        let reader = writer.create_reader(set_bitset).unwrap();

        // Single term query
        let mut res: HashSet<u32> = HashSet::new();
        reader
            .and_match_query("hello", &mut res as *mut _ as *mut c_void)
            .unwrap();

        assert_eq!(res, vec![0, 1].into_iter().collect::<HashSet<u32>>());
    }

    #[test]
    fn test_and_match_query_nonexistent_term() {
        let dir = TempDir::new().unwrap();
        let mut writer = IndexWriterWrapper::create_text_writer(
            "text",
            dir.path().to_str().unwrap(),
            "default",
            "",
            1,
            50_000_000,
            false,
            TantivyIndexVersion::default_version(),
        )
        .unwrap();

        writer.add("hello world", Some(0)).unwrap();
        writer.commit().unwrap();

        let reader = writer.create_reader(set_bitset).unwrap();

        // Query with nonexistent term
        let mut res: HashSet<u32> = HashSet::new();
        reader
            .and_match_query("hello nonexistent", &mut res as *mut _ as *mut c_void)
            .unwrap();

        assert!(res.is_empty());
    }

    #[test]
    fn test_and_match_query_three_terms() {
        let dir = TempDir::new().unwrap();
        let mut writer = IndexWriterWrapper::create_text_writer(
            "text",
            dir.path().to_str().unwrap(),
            "default",
            "",
            1,
            50_000_000,
            false,
            TantivyIndexVersion::default_version(),
        )
        .unwrap();

        writer.add("a b c", Some(0)).unwrap();
        writer.add("a b", Some(1)).unwrap();
        writer.add("b c", Some(2)).unwrap();
        writer.add("a c", Some(3)).unwrap();
        writer.add("a b c d", Some(4)).unwrap();
        writer.commit().unwrap();

        let reader = writer.create_reader(set_bitset).unwrap();

        // Three term AND query
        let mut res: HashSet<u32> = HashSet::new();
        reader
            .and_match_query("a b c", &mut res as *mut _ as *mut c_void)
            .unwrap();

        assert_eq!(res, vec![0, 4].into_iter().collect::<HashSet<u32>>());
    }
}
