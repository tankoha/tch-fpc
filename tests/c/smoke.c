/* C smoke test for libtch (HANDOFF step 3): create tensors, call atg_add,
   print the result, then trigger an error and read it back.

   Build (from the repo root, after building build/libtch):
     gcc -std=c11 -Wall -Wextra tests/c/smoke.c -Icsrc/libtch \
       -Lbuild/libtch -ltch -Wl,-rpath,$PWD/build/libtch -o build/smoke
     build/smoke */
#include <stdio.h>
#include <stdlib.h>
#include "torch_api_generated.h"

static int check_err(const char *what) {
  char *err = get_and_reset_last_err();
  if (err == NULL) return 0;
  printf("%s: error: %.200s\n", what, err);
  tchf_free(err);
  return 1;
}

int main(void) {
  float a_data[] = {1, 2, 3, 4, 5, 6};
  float b_data[] = {10, 20, 30, 40, 50, 60};
  int64_t dims23[] = {2, 3};
  int64_t dims32[] = {3, 2};
  int failures = 0;

  /* 6 = c10::ScalarType::Float */
  tensor a = at_tensor_of_data(a_data, dims23, 2, sizeof(float), 6);
  tensor b = at_tensor_of_data(b_data, dims23, 2, sizeof(float), 6);
  tensor c = at_tensor_of_data(b_data, dims32, 2, sizeof(float), 6);
  failures += check_err("tensor_of_data");

  tensor sum = NULL;
  atg_add(&sum, a, b);
  failures += check_err("atg_add");
  char *s = at_to_string(sum, 80);
  failures += check_err("at_to_string");
  printf("a + b =\n%s\n", s);
  tchf_free(s);

  /* Expected failure: shapes [2,3] and [3,2] do not broadcast. */
  tensor bad = NULL;
  atg_add(&bad, a, c);
  if (!check_err("atg_add [2,3]+[3,2] (expected)")) {
    printf("FAIL: no error reported for mismatched shapes\n");
    failures++;
  }
  if (bad != NULL) {
    printf("FAIL: output set despite error\n");
    failures++;
  }
  if (get_and_reset_last_err() != NULL) {
    printf("FAIL: error slot not reset\n");
    failures++;
  }

  at_free(sum); at_free(a); at_free(b); at_free(c);
  printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
  return failures != 0;
}
