#!/usr/bin/env python3
"""Cross-compile VGRE's compute kernels to RISC-V (riscv64-linux-gnu-gcc) and run
them under QEMU user-mode emulation, checking results. docs/missingFeatures.md
§3.3. SKIPs (exit 0) if the riscv64 toolchain or qemu-riscv64 isn't installed."""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(HERE, "..", "wasm", "vgre_kernels.c")  # shared portable kernels

GCC = shutil.which("riscv64-linux-gnu-gcc")
QEMU = shutil.which("qemu-riscv64-static") or shutil.which("qemu-riscv64")
if not GCC or not QEMU:
    print(f"SKIP: riscv64 toolchain ({'ok' if GCC else 'missing'}) / "
          f"qemu ({'ok' if QEMU else 'missing'}) not available")
    sys.exit(0)

MAIN = r"""
#include <stdio.h>
extern void vadd(const float*,const float*,float*,int);
extern void vmul(const float*,const float*,float*,int);
extern void saxpy(float,const float*,float*,int);
extern void relu(const float*,float*,int);
extern float reduce_sum(const float*,int);
extern void sgemm(const float*,const float*,float*,int,int,int);
int main(void){
  float a[4]={1,2,3,4}, b[4]={10,20,30,40}, o[4];
  vadd(a,b,o,4);
  float m1[3]={2,3,4}, m2[3]={5,6,7}, mo[3]; vmul(m1,m2,mo,3);
  float x[3]={1,2,3}, y[3]={10,10,10}; saxpy(2.0f,x,y,3);
  float r[4]={-1,2,-3,4}, ro[4]; relu(r,ro,4);
  float A[6]={1,2,3,4,5,6}, B[6]={1,2,3,4,5,6}, C[4]; sgemm(A,B,C,2,2,3);
  printf("vadd=%g,%g,%g,%g vmul=%g,%g,%g saxpy=%g,%g,%g relu=%g,%g,%g,%g sum=%g sgemm=%g,%g,%g,%g\n",
         o[0],o[1],o[2],o[3], mo[0],mo[1],mo[2], y[0],y[1],y[2],
         ro[0],ro[1],ro[2],ro[3], reduce_sum(a,4), C[0],C[1],C[2],C[3]);
  return 0;
}
"""

EXPECT = ("vadd=11,22,33,44 vmul=10,18,28 saxpy=12,14,16 "
          "relu=0,2,0,4 sum=10 sgemm=22,28,49,64")


def main():
    with tempfile.TemporaryDirectory() as d:
        mainc = os.path.join(d, "main.c")
        elf = os.path.join(d, "vgre_riscv.elf")
        with open(mainc, "w") as f:
            f.write(MAIN)
        cp = subprocess.run([GCC, "-O2", "-static", mainc, KERNELS, "-o", elf],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            print("FAIL: riscv64 cross-compile failed:\n", cp.stderr)
            return 1
        run = subprocess.run([QEMU, elf], capture_output=True, text=True)
        out = run.stdout.strip()
        print("  riscv64 ELF ran under QEMU:")
        print("   ", out)
        ok = out == EXPECT
        print(("  PASS" if ok else "  FAIL") + ": kernel outputs match reference")
        if not ok:
            print("    expected:", EXPECT)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
