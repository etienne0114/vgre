// Build VGRE's compute kernels to WebAssembly (clang --target=wasm32, no
// emscripten) and run them in a sandbox, checking results against references.
// docs/missingFeatures.md §3.3. SKIPs (exit 0) if clang can't target wasm32.
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import fs from 'node:fs';

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, 'vgre_kernels.c');
const OUT = join(tmpdir(), `vgre_kernels_${process.pid}.wasm`);

// ── compile to wasm32 with clang's built-in WebAssembly backend ──
try {
  execFileSync('clang', [
    '--target=wasm32', '-O2', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-dynamic',
    '-Wl,--initial-memory=131072', '-Wl,--export=__heap_base',
    SRC, '-o', OUT,
  ], { stdio: 'pipe' });
} catch (e) {
  console.log('SKIP: clang cannot target wasm32:', String(e.message).split('\n')[0]);
  process.exit(0);
}

const bytes = fs.readFileSync(OUT);
const { instance } = await WebAssembly.instantiate(bytes, {});
const ex = instance.exports;
const f32 = new Float32Array(ex.memory.buffer);
const HEAP = ex.__heap_base ? ex.__heap_base.value : 4096;
let cursor = HEAP >> 2; // word index into the f32 view

// allocate `n` floats in wasm linear memory; returns [wordIndex, byteOffset]
function alloc(values) {
  const w = cursor;
  for (let i = 0; i < values.length; ++i) f32[w + i] = values[i];
  cursor += values.length;
  return [w, w * 4];
}
function allocN(n) { return alloc(new Array(n).fill(0)); }
function read(w, n) { return Array.from(f32.slice(w, w + n)); }

let pass = 0, total = 0;
const approx = (a, b) => Math.abs(a - b) < 1e-4;
function check(name, ok) {
  total++; if (ok) pass++;
  console.log((ok ? '  PASS  ' : '  FAIL  ') + name);
}

// ── vadd ──
{
  const [, A] = alloc([1, 2, 3, 4]);
  const [, B] = alloc([10, 20, 30, 40]);
  const [wo, O] = allocN(4);
  ex.vadd(A, B, O, 4);
  check('vadd', JSON.stringify(read(wo, 4)) === JSON.stringify([11, 22, 33, 44]));
}
// ── vmul ──
{
  const [, A] = alloc([2, 3, 4]);
  const [, B] = alloc([5, 6, 7]);
  const [wo, O] = allocN(3);
  ex.vmul(A, B, O, 3);
  check('vmul', JSON.stringify(read(wo, 3)) === JSON.stringify([10, 18, 28]));
}
// ── saxpy: y = 2*x + y ──
{
  const [, X] = alloc([1, 2, 3]);
  const [wy, Y] = alloc([10, 10, 10]);
  ex.saxpy(2.0, X, Y, 3);
  check('saxpy', JSON.stringify(read(wy, 3)) === JSON.stringify([12, 14, 16]));
}
// ── relu ──
{
  const [, X] = alloc([-1, 2, -3, 4]);
  const [wo, O] = allocN(4);
  ex.relu(X, O, 4);
  check('relu', JSON.stringify(read(wo, 4)) === JSON.stringify([0, 2, 0, 4]));
}
// ── reduce_sum ──
{
  const [, X] = alloc([1.5, 2.5, 3, 4]);
  check('reduce_sum', approx(ex.reduce_sum(X, 4), 11.0));
}
// ── sgemm: [2x3]*[3x2] ──
{
  const [, A] = alloc([1, 2, 3, 4, 5, 6]);   // 2x3
  const [, B] = alloc([1, 2, 3, 4, 5, 6]);   // 3x2
  const [wc, C] = allocN(4);                 // 2x2
  ex.sgemm(A, B, C, 2, 2, 3);
  // ref: [[1*1+2*3+3*5, 1*2+2*4+3*6],[4*1+5*3+6*5,4*2+5*4+6*6]] = [[22,28],[49,64]]
  check('sgemm matmul', JSON.stringify(read(wc, 4)) === JSON.stringify([22, 28, 49, 64]));
}

fs.rmSync(OUT, { force: true });
console.log(`\n${pass} / ${total} passed`);
console.log(`(VGRE compute kernels ran in a WebAssembly sandbox, ${bytes.length} byte module)`);
process.exit(pass === total ? 0 : 1);
