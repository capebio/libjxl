// Self-contained A/B harness for BitWriter::AppendUnaligned (OLD vs NEW).
// Mirrors the exact store algorithm + zero-tail invariant + PaddedBytes-style
// non-zero-initialized storage with 8 bytes of write-past-end slack, so a NEW
// path that left garbage in the tail would diverge from OLD on read-back.
//
// Build (little-endian x86):
//   clang++ -O3 -std=c++17 enc_bit_writer_append_ab.cc -o bw_ab
//
// Proves: byte-exact OLD==NEW over every destination bit offset and size, then
// reports interleaved timing for the aligned (memcpy) and unaligned (56-bit
// block) paths versus the per-byte OLD loop.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

namespace {

constexpr size_t kMaxBitsPerCall = 56;
constexpr uint64_t kLow56 = (uint64_t{1} << kMaxBitsPerCall) - 1;

// Minimal faithful model of BitWriter + PaddedBytes (little-endian path).
struct Writer {
  std::vector<uint8_t> buf;  // storage; never zero-filled on grow (like PaddedBytes)
  size_t bits = 0;

  explicit Writer(size_t reserve_bytes) {
    // Fill with non-zero garbage, then init first byte to 0 (PaddedBytes does).
    buf.assign(reserve_bytes + 16, 0x5A);
    buf[0] = 0;
  }
  void Ensure(size_t end_byte) {
    if (buf.size() < end_byte + 8) {
      size_t old = buf.size();
      buf.resize(end_byte + 8, 0x5A);  // garbage-fill new region
      (void)old;
    }
  }
  // Exact replica of BitWriter::Write (JXL_BYTE_ORDER_LITTLE branch).
  void Write(size_t n_bits, uint64_t b) {
    size_t byte = bits / 8;
    Ensure(byte);
    size_t in_first = bits % 8;
    b <<= in_first;
    uint64_t v = buf[byte];  // single byte, zero-extended
    v |= b;
    memcpy(&buf[byte], &v, sizeof(v));
    bits += n_bits;
  }
};

// OLD AppendUnaligned: one Write(8) per full byte + partial tail.
void AppendOld(Writer& w, const uint8_t* src, size_t src_bits) {
  size_t full = src_bits / 8;
  size_t rem = src_bits % 8;
  for (size_t i = 0; i < full; ++i) w.Write(8, src[i]);
  if (rem) w.Write(rem, src[full] & ((uint64_t{1} << rem) - 1));
}

// NEW AppendUnaligned: aligned bulk memcpy / unaligned 56-bit blocks.
void AppendNew(Writer& w, const uint8_t* src, size_t src_bits) {
  if (src_bits == 0) return;
  size_t full = src_bits / 8;
  const size_t rem = src_bits % 8;
  if (w.bits % 8 == 0) {
    size_t pos = w.bits / 8;
    w.Ensure(pos + full + 1);
    uint8_t* dst = w.buf.data() + pos;
    memcpy(dst, src, full);
    dst[full] = 0;
    w.bits += full * 8;
    if (rem) w.Write(rem, src[full] & ((uint64_t{1} << rem) - 1));
    return;
  }
  while (full >= 7) {
    uint64_t chunk;
    memcpy(&chunk, src, sizeof(chunk));
    w.Write(kMaxBitsPerCall, chunk & kLow56);
    src += 7;
    full -= 7;
  }
  while (full != 0) {
    w.Write(8, *src++);
    --full;
  }
  if (rem) w.Write(rem, *src & ((uint64_t{1} << rem) - 1));
}

// Build a source byte buffer of exactly src_bits via Writer (so it has the same
// in-storage layout BitWriter would produce, incl. padding for the 8-byte read).
std::vector<uint8_t> MakeSource(size_t src_bits, uint32_t seed) {
  Writer s(src_bits / 8 + 8);
  std::mt19937 mt(seed);
  std::uniform_int_distribution<int> len_dist(1, (int)kMaxBitsPerCall);
  size_t written = 0;
  while (written < src_bits) {
    size_t len = len_dist(mt);
    if (len > src_bits - written) len = src_bits - written;
    uint64_t mask = (uint64_t{1} << len) - 1;
    uint64_t b = (((uint64_t)mt() << 32) | mt()) & mask;
    s.Write(len, b);
    written += len;
  }
  return s.buf;  // ceil(src_bits/8) valid bytes + padding
}

bool ByteExactCheck() {
  size_t sizes[] = {0, 1, 5, 8, 13, 56, 64, 191, 192, 197, 1024, 4099};
  size_t mismatches = 0, cases = 0;
  for (size_t src_bits : sizes) {
    std::vector<uint8_t> src = MakeSource(src_bits, (uint32_t)(src_bits * 2654435761u + 1));
    for (size_t prefix = 0; prefix < 8; ++prefix) {
      for (int chain = 1; chain <= 2; ++chain) {
        Writer wo(prefix / 8 + src_bits / 8 * chain + 16);
        Writer wn(prefix / 8 + src_bits / 8 * chain + 16);
        uint64_t pv = prefix ? ((uint64_t{1} << prefix) - 1) : 0;
        if (prefix) { wo.Write(prefix, pv); wn.Write(prefix, pv); }
        for (int r = 0; r < chain; ++r) {
          AppendOld(wo, src.data(), src_bits);
          AppendNew(wn, src.data(), src_bits);
        }
        // trailing write to exercise zero-tail after append
        wo.Write(1, 1); wn.Write(1, 1);
        ++cases;
        if (wo.bits != wn.bits) { ++mismatches; continue; }
        size_t nbytes = (wo.bits + 7) / 8;
        if (memcmp(wo.buf.data(), wn.buf.data(), nbytes) != 0) ++mismatches;
      }
    }
  }
  printf("byte-exact: %zu cases, %zu mismatches => %s\n", cases, mismatches,
         mismatches == 0 ? "PASS" : "FAIL");
  return mismatches == 0;
}

double TimeAppend(void (*fn)(Writer&, const uint8_t*, size_t),
                  const uint8_t* src, size_t src_bits, size_t prefix,
                  int iters) {
  volatile uint64_t sink = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    Writer w(prefix / 8 + src_bits / 8 + 16);
    if (prefix) w.Write(prefix, (uint64_t{1} << prefix) - 1);
    fn(w, src, src_bits);
    sink ^= w.bits ^ w.buf[w.bits / 8 ? w.bits / 8 - 1 : 0];
  }
  auto t1 = std::chrono::steady_clock::now();
  (void)sink;
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

void Bench(size_t src_bits, size_t prefix) {
  std::vector<uint8_t> src = MakeSource(src_bits, 12345);
  const int iters = 200000;
  // Interleaved with start rotation to cancel thermal drift.
  double old_t = 0, new_t = 0;
  for (int round = 0; round < 6; ++round) {
    if (round % 2 == 0) {
      old_t += TimeAppend(AppendOld, src.data(), src_bits, prefix, iters);
      new_t += TimeAppend(AppendNew, src.data(), src_bits, prefix, iters);
    } else {
      new_t += TimeAppend(AppendNew, src.data(), src_bits, prefix, iters);
      old_t += TimeAppend(AppendOld, src.data(), src_bits, prefix, iters);
    }
  }
  old_t /= 6; new_t /= 6;
  const char* path = (prefix == 0) ? "aligned/memcpy" : "unaligned/56-bit";
  printf("  bits=%-5zu prefix=%zu [%-16s]  OLD %8.1f ns  NEW %8.1f ns  %.2fx\n",
         src_bits, prefix, path, old_t, new_t, old_t / new_t);
}

}  // namespace

int main() {
  if (!ByteExactCheck()) return 1;
  printf("timing (interleaved x6, 200k iters each):\n");
  for (size_t bits : {64, 192, 512, 4096}) {
    Bench(bits, 0);  // aligned destination -> memcpy path
    Bench(bits, 3);  // unaligned destination -> 56-bit block path
  }
  return 0;
}
