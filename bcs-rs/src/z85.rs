//! Z85 decoding (ZeroMQ RFC 32).

use rayon::prelude::*;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors produced by z85 decoding.
#[derive(Debug)]
pub enum Z85Error {
    /// Input byte count is not a multiple of 5.
    InvalidLength(usize),
    /// A byte value has no entry in the z85 alphabet.
    InvalidByte(u8),
}

impl std::fmt::Display for Z85Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Z85Error::InvalidLength(n) => {
                write!(f, "input length {n} is not a multiple of 5")
            }
            Z85Error::InvalidByte(b) => {
                write!(f, "invalid z85 byte {b:#04x} ('{}')", *b as char)
            }
        }
    }
}

impl std::error::Error for Z85Error {}

// ── Decode table ──────────────────────────────────────────────────────────────

/// Maps every ASCII byte to its 0-84 z85 digit value.
/// 255 is the sentinel for bytes not in the z85 alphabet.
const DECODE: [u8; 256] = {
    let mut t = [255u8; 256];
    let alpha =
        b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";
    let mut i = 0usize;
    while i < 85 {
        t[alpha[i] as usize] = i as u8;
        i += 1;
    }
    t
};

// ── SIMD lookup tables (+ 1 sentinel trick) ───────────────────────────────────

/// Build a 64-byte SIMD table covering ASCII bytes [base, base+63].
/// Each entry stores DECODE[ascii] + 1 (range 1–85 for valid chars, 0 otherwise).
/// The +1 sentinel means a result of 0 unambiguously signals an invalid byte.
const fn build_simd_table(base: u8) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut i = 0usize;
    while i < 64 {
        let ascii = base as usize + i;
        if ascii < 256 {
            let v = DECODE[ascii];
            if v != 255 {
                t[i] = v + 1;
            }
        }
        i += 1;
    }
    t
}

/// SIMD decode table — lower half: covers ASCII 32–95.
/// Index = byte - 32.  Entry = DECODE[byte] + 1, or 0 if not in z85 alphabet.
const SIMD_LO: [u8; 64] = build_simd_table(32);

/// SIMD decode table — upper half: covers ASCII 96–159 (only 96–122 matter).
/// Index = byte - 96.  Entry = DECODE[byte] + 1, or 0 if not in z85 alphabet.
const SIMD_HI: [u8; 64] = build_simd_table(96);

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of z85 groups per SIMD block (5 bytes in → 4 bytes out each).
/// 4 groups × 5 = 20 bytes in, × 4 = 16 bytes out — one 128-bit SIMD lane.
/// `GROUPS_PER_TASK` must remain a multiple of this.
const BLOCK_GROUPS: usize = 4;
const BLOCK_IN: usize = BLOCK_GROUPS * 5;   // 20
const BLOCK_OUT: usize = BLOCK_GROUPS * 4;  // 16

/// Rayon task granularity. Each task's working set (20 KB in / 16 KB out)
/// fits comfortably in L1 cache. Must be a multiple of `BLOCK_GROUPS`.
const GROUPS_PER_TASK: usize = 4096; // 4096 % BLOCK_GROUPS == 0 ✓

// ── Scalar primitives ─────────────────────────────────────────────────────────

/// Decode one 5-byte z85 group into 4 raw bytes.
#[inline]
fn decode_group(src: &[u8; 5]) -> Result<[u8; 4], Z85Error> {
    let d0 = DECODE[src[0] as usize];
    let d1 = DECODE[src[1] as usize];
    let d2 = DECODE[src[2] as usize];
    let d3 = DECODE[src[3] as usize];
    let d4 = DECODE[src[4] as usize];

    if d0.max(d1).max(d2).max(d3).max(d4) == 255 {
        let &bad = src.iter().find(|&&b| DECODE[b as usize] == 255).unwrap();
        return Err(Z85Error::InvalidByte(bad));
    }

    let v = d0 as u64 * 52_200_625
        + d1 as u64 * 614_125
        + d2 as u64 * 7_225
        + d3 as u64 * 85
        + d4 as u64;

    Ok((v as u32).to_be_bytes())
}

// ── SIMD helpers ──────────────────────────────────────────────────────────────

/// Accumulate 20 pre-decoded digits (0–84) into 16 output bytes (4 groups).
#[inline]
fn accumulate_groups(d: &[u8; 20]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for g in 0..4 {
        let i = g * 5;
        let v = d[i]     as u64 * 52_200_625
              + d[i + 1] as u64 * 614_125
              + d[i + 2] as u64 * 7_225
              + d[i + 3] as u64 * 85
              + d[i + 4] as u64;
        out[g * 4..][..4].copy_from_slice(&(v as u32).to_be_bytes());
    }
    out
}

/// Cold error path — only called when SIMD finds a zero digit.
#[cold]
fn find_invalid_byte(src: &[u8; 20]) -> Z85Error {
    let &bad = src.iter().find(|&&b| DECODE[b as usize] == 255).unwrap();
    Z85Error::InvalidByte(bad)
}

// ── Platform SIMD implementations ─────────────────────────────────────────────

/// Decode exactly 4 z85 groups (20 bytes → 16 bytes) using aarch64 NEON.
///
/// `vqtbl4q_u8` performs a full 64-byte table lookup in a single instruction:
/// indices ≥ 64 produce 0, which is our sentinel for invalid bytes.
///
/// The first 16 bytes are decoded via SIMD and stored with a single `vst1q_u8`.
/// The last 4 bytes use scalar DECODE (avoids a second 16-byte overlapping load
/// and 4 individual lane extractions).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_block_neon(src: &[u8; 20]) -> Result<[u8; 16], Z85Error> {
    use core::arch::aarch64::*;

    // SAFETY: [u8; 64] and uint8x16x4_t have identical size (64 bytes) and alignment.
    let tbl_lo: uint8x16x4_t = unsafe { core::mem::transmute(SIMD_LO) };
    let tbl_hi: uint8x16x4_t = unsafe { core::mem::transmute(SIMD_HI) };

    let s32 = vdupq_n_u8(32);
    let s96 = vdupq_n_u8(96);

    // ── First 16 input bytes → 16 digit values ────────────────────────────
    // SAFETY: src has 20 bytes; reading 16 from offset 0 is in-bounds.
    let b0 = unsafe { vld1q_u8(src.as_ptr()) };

    // Wrapping subtract pushes out-of-range bytes into idx ≥ 64,
    // so vqtbl4q_u8 naturally returns 0 for them.
    let d0 = vorrq_u8(
        vqtbl4q_u8(tbl_lo, vsubq_u8(b0, s32)),
        vqtbl4q_u8(tbl_hi, vsubq_u8(b0, s96)),
    );

    // ── Validity check for first 16 bytes ─────────────────────────────────
    // vminvq_u8 horizontal minimum: any 0 means an invalid byte.
    if vminvq_u8(d0) == 0 {
        return Err(find_invalid_byte(src));
    }

    // ── Last 4 bytes: scalar DECODE ───────────────────────────────────────
    let e16 = DECODE[src[16] as usize];
    let e17 = DECODE[src[17] as usize];
    let e18 = DECODE[src[18] as usize];
    let e19 = DECODE[src[19] as usize];
    if e16.max(e17).max(e18).max(e19) == 255 {
        return Err(find_invalid_byte(src));
    }

    // ── Subtract 1, write 16 digits to stack in one store ────────────────
    // One vst1q_u8 instead of 16 individual vgetq_lane_u8 extractions.
    let d0s = vsubq_u8(d0, vdupq_n_u8(1));
    let mut buf = [0u8; 16];
    unsafe { vst1q_u8(buf.as_mut_ptr(), d0s) };

    // e16..e19 are already DECODE values (0-84), matching buf's 0-84 range.
    let digits: [u8; 20] = [
        buf[0],  buf[1],  buf[2],  buf[3],
        buf[4],  buf[5],  buf[6],  buf[7],
        buf[8],  buf[9],  buf[10], buf[11],
        buf[12], buf[13], buf[14], buf[15],
        e16, e17, e18, e19,
    ];

    Ok(accumulate_groups(&digits))
}

/// Decode exactly 4 z85 groups (20 bytes → 16 bytes) using x86_64 SSSE3.
///
/// `_mm_shuffle_epi8` returns 0 for indices with bit 7 set, giving us a natural
/// "out of range → 0" behaviour. We split the 64-byte table into four 16-byte
/// slices and OR the results to cover the full range in one pass.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn decode_block_ssse3(src: &[u8; 20]) -> Result<[u8; 16], Z85Error> {
    use core::arch::x86_64::*;

    /// Perform a 64-byte table lookup using four 16-byte SSSE3 shuffles.
    ///
    /// For slice k the sub-index is `idx - k*16`.  `_mm_shuffle_epi8` returns 0
    /// when bit 7 of the index is set (signed negative), which handles both
    /// underflow (byte < base) and, combined with `_mm_cmpgt_epi8`, overflow
    /// (idx in [16, 127]).
    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn lookup64(table: &[u8; 64], idx: __m128i) -> __m128i {
        let sixteen = _mm_set1_epi8(16i8);
        let t0 = unsafe { _mm_loadu_si128(table[0..].as_ptr() as *const __m128i) };
        let t1 = unsafe { _mm_loadu_si128(table[16..].as_ptr() as *const __m128i) };
        let t2 = unsafe { _mm_loadu_si128(table[32..].as_ptr() as *const __m128i) };
        let t3 = unsafe { _mm_loadu_si128(table[48..].as_ptr() as *const __m128i) };

        let sub0 = idx;
        let sub1 = _mm_sub_epi8(idx, _mm_set1_epi8(16));
        let sub2 = _mm_sub_epi8(idx, _mm_set1_epi8(32));
        let sub3 = _mm_sub_epi8(idx, _mm_set1_epi8(48));

        let r0 = _mm_and_si128(_mm_cmpgt_epi8(sixteen, sub0), _mm_shuffle_epi8(t0, sub0));
        let r1 = _mm_and_si128(_mm_cmpgt_epi8(sixteen, sub1), _mm_shuffle_epi8(t1, sub1));
        let r2 = _mm_and_si128(_mm_cmpgt_epi8(sixteen, sub2), _mm_shuffle_epi8(t2, sub2));
        let r3 = _mm_and_si128(_mm_cmpgt_epi8(sixteen, sub3), _mm_shuffle_epi8(t3, sub3));

        _mm_or_si128(_mm_or_si128(r0, r1), _mm_or_si128(r2, r3))
    }

    // ── First 16 input bytes → 16 digit values ────────────────────────────
    // SAFETY: src has 20 bytes; reading 16 from offset 0 is in-bounds.
    let b0 = unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) };

    let idx_lo = _mm_sub_epi8(b0, _mm_set1_epi8(32i8));
    let idx_hi = _mm_sub_epi8(b0, _mm_set1_epi8(96i8));
    let d0 = _mm_or_si128(lookup64(&SIMD_LO, idx_lo), lookup64(&SIMD_HI, idx_hi));

    if _mm_movemask_epi8(_mm_cmpeq_epi8(d0, _mm_setzero_si128())) != 0 {
        return Err(find_invalid_byte(src));
    }

    // ── Last 4 bytes (indices 16–19): scalar DECODE lookup ────────────────
    let e16 = DECODE[src[16] as usize];
    let e17 = DECODE[src[17] as usize];
    let e18 = DECODE[src[18] as usize];
    let e19 = DECODE[src[19] as usize];
    if e16.max(e17).max(e18).max(e19) == 255 {
        return Err(find_invalid_byte(src));
    }

    let d0s = _mm_sub_epi8(d0, _mm_set1_epi8(1));

    // Store to stack array — one movdqu, avoids SSE4.1 _mm_extract_epi8.
    let mut lanes = [0u8; 16];
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, d0s) };

    let digits: [u8; 20] = [
        lanes[0],  lanes[1],  lanes[2],  lanes[3],
        lanes[4],  lanes[5],  lanes[6],  lanes[7],
        lanes[8],  lanes[9],  lanes[10], lanes[11],
        lanes[12], lanes[13], lanes[14], lanes[15],
        e16, e17, e18, e19,
    ];

    Ok(accumulate_groups(&digits))
}

// ── Scalar fallback ───────────────────────────────────────────────────────────

/// Scalar fallback — used on non-SIMD targets.
#[inline]
fn decode_block_scalar(src: &[u8; BLOCK_IN]) -> Result<[u8; BLOCK_OUT], Z85Error> {
    let mut out = [0u8; BLOCK_OUT];
    for i in 0..BLOCK_GROUPS {
        let group: &[u8; 5] = src[i * 5..][..5].try_into().expect("fixed stride");
        out[i * 4..][..4].copy_from_slice(&decode_group(group)?);
    }
    Ok(out)
}

// ── SIMD dispatcher ───────────────────────────────────────────────────────────

/// Decode exactly `BLOCK_GROUPS` (4) consecutive z85 groups: 20 bytes → 16 bytes.
///
/// Dispatches to the best available SIMD implementation at compile/run time.
#[inline]
#[allow(unreachable_code)]
fn decode_block(src: &[u8; BLOCK_IN]) -> Result<[u8; BLOCK_OUT], Z85Error> {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on all aarch64 targets.
        return unsafe { decode_block_neon(src) };
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("ssse3") {
        // SAFETY: feature check above guarantees SSSE3 is available.
        return unsafe { decode_block_ssse3(src) };
    }

    decode_block_scalar(src)
}

// ── Shared inner loop ─────────────────────────────────────────────────────────

/// Decode `input` into the pre-allocated `out` buffer.
fn decode_chunk(input: &[u8], out: &mut [u8]) -> Result<(), Z85Error> {
    let n_blocks = input.len() / BLOCK_IN;
    let (main_in, tail_in) = input.split_at(n_blocks * BLOCK_IN);
    let (main_out, tail_out) = out.split_at_mut(n_blocks * BLOCK_OUT);

    for (src, dst) in main_in
        .chunks_exact(BLOCK_IN)
        .zip(main_out.chunks_exact_mut(BLOCK_OUT))
    {
        let src: &[u8; BLOCK_IN] = src.try_into().expect("chunks_exact(BLOCK_IN)");
        dst.copy_from_slice(&decode_block(src)?);
    }

    // Tail: 0-3 groups that didn't fill a complete block.
    for (src, dst) in tail_in
        .chunks_exact(5)
        .zip(tail_out.chunks_exact_mut(4))
    {
        let src: &[u8; 5] = src.try_into().expect("chunks_exact(5)");
        dst.copy_from_slice(&decode_group(src)?);
    }

    Ok(())
}

// ── Allocation helper ─────────────────────────────────────────────────────────

#[inline]
fn alloc_uninit(n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n);
    // SAFETY: u8 has no invalid bit patterns; caller writes every byte before Ok.
    unsafe { v.set_len(n) };
    v
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Decode a z85-encoded byte slice into raw bytes (single-threaded).
pub fn decode_scalar(input: &[u8]) -> Result<Vec<u8>, Z85Error> {
    if input.len() % 5 != 0 {
        return Err(Z85Error::InvalidLength(input.len()));
    }
    let mut out = alloc_uninit(input.len() / 5 * 4);
    decode_chunk(input, &mut out)?;
    Ok(out)
}

/// Decode a z85-encoded byte slice into raw bytes (Rayon parallel).
pub fn decode_parallel(input: &[u8]) -> Result<Vec<u8>, Z85Error> {
    if input.len() % 5 != 0 {
        return Err(Z85Error::InvalidLength(input.len()));
    }
    let mut out = alloc_uninit(input.len() / 5 * 4);
    input
        .par_chunks(GROUPS_PER_TASK * 5)
        .zip(out.par_chunks_mut(GROUPS_PER_TASK * 4))
        .try_for_each(|(in_chunk, out_chunk)| -> Result<(), Z85Error> {
            decode_chunk(in_chunk, out_chunk)
        })?;
    Ok(out)
}
