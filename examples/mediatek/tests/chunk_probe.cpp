#include <executorch/examples/mediatek/executor_runner/llama_runner/ModelChunk.h>
#include <executorch/runtime/platform/runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <stdexcept>
class Probe : public example::ModelChunk {
 public:
 using ModelChunk::ModelChunk;
 size_t inputs(){return GetModelMethod().inputs_size();}
 size_t outputs(){return GetModelMethod().outputs_size();}
};
int main(int argc,char**argv){
 if(argc!=5){std::fprintf(stderr,"usage: probe model.pte fixture_dir output_dir iterations\n");return 2;}
 executorch::runtime::runtime_init();
 try {
 Probe model(std::string(argv[1]),128);model.Initialize();
 for(size_t i=0;i<model.inputs();++i){
  const auto b=model.GetInputBuffer(i);std::string path=std::string(argv[2])+"/input_"+std::to_string(i)+".bin";
  std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f || size_t(f.tellg())!=b.nbytesUsed)throw std::runtime_error("input size mismatch: "+path);
  std::vector<char> data(b.nbytesUsed);f.seekg(0);f.read(data.data(),data.size());if(!f)throw std::runtime_error("input read failed");model.SetInputBuffer(data.data(),data.size(),i);
 }
 const int iterations=std::stoi(argv[4]);if(iterations<1||iterations>1000)throw std::runtime_error("bad iteration count");
 for(int i=-3;i<iterations;++i){model.Run();std::fprintf(stderr,"PROBE_ITER phase=%s index=%d method_ms=%.6f\n",i<0?"warmup":"measure",i,model.GetLastMethodExecuteMs());}
 size_t bad=0;
 for(size_t i=0;i<model.outputs();++i){const auto b=model.GetOutputBuffer(i);std::string path=std::string(argv[3])+"/output_"+std::to_string(i)+".bin";
  std::ofstream f(path,std::ios::binary);f.write(static_cast<const char*>(b.data),b.nbytesUsed);if(!f)throw std::runtime_error("output write failed");
  auto values=static_cast<const float*>(b.data);for(size_t j=0;j<b.nbytesUsed/sizeof(float);++j)bad+=!std::isfinite(values[j]);
 }
 model.Release();std::fprintf(stderr,"PROBE_DONE nonfinite=%zu\n",bad);return bad?3:0;
 }catch(const std::exception&e){std::fprintf(stderr,"PROBE_ERROR %s\n",e.what());return 2;}
}
