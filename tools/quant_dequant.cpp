// quant_dequant — dequantize raw ggml block bytes (from stdin) of a given
// type into f32 (to stdout), via vgre::xla::dequantBlock. Lets a script feed
// random super-blocks and diff against the reference gguf.quants.dequantize
// (tools/validate_kquant.py).
//
//   quant_dequant <ggml_type> <n>   (n = element count, multiple of block size)
#include "vgre/xla/quant.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc,char**argv){
  int type=atoi(argv[1]); long n=atol(argv[2]);
  long bytes=vgre::xla::quantStorageBytes(type,n);
  std::vector<unsigned char> buf(bytes);
  if((long)fread(buf.data(),1,bytes,stdin)!=bytes) return 2;
  std::vector<float> out(n);
  if(!vgre::xla::dequantBlock(type,buf.data(),n,out.data())) return 3;
  fwrite(out.data(),4,n,stdout); return 0;
}
