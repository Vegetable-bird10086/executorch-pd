// Single-process entrypoint layered over the Executorch Prefill runner and
// llama.cpp pd-cli Decode implementation.
int qnn_llama_pd_e2e_main(int argc, char** argv);

int main(int argc, char** argv) {
  return qnn_llama_pd_e2e_main(argc, argv);
}
