#include "openjtalk_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Declare opaque C structures
typedef struct _Mecab Mecab;
typedef struct _NJD NJD;
typedef struct _JPCommon JPCommon;

extern "C" {
// Forward declare Mecab functions (avoid including mecab.h due to C++ template issues)
void Mecab_initialize(Mecab * mecab);
void Mecab_analysis(Mecab * mecab, const char *str);
int Mecab_get_size(Mecab * mecab);
char **Mecab_get_feature(Mecab * mecab);
void Mecab_clear(Mecab * mecab);

// Declare all other OpenJTalk functions we need
void NJD_initialize(NJD * njd);
void NJD_clear(NJD * njd);
void text2mecab(char *output, const char *input);
void mecab2njd(NJD * njd, char **feature, int size);
void njd_set_pronunciation(NJD * njd);
void njd_set_digit(NJD * njd);
void njd_set_accent_phrase(NJD * njd);
void njd_set_accent_type(NJD * njd);
void njd_set_unvoiced_vowel(NJD * njd);
void njd_set_long_vowel(NJD * njd);
void JPCommon_initialize(JPCommon * jpcommon);
void JPCommon_clear(JPCommon * jpcommon);
void njd2jpcommon(JPCommon * jpcommon, NJD * njd);
void JPCommon_make_label(JPCommon * jpcommon);
int JPCommon_get_label_size(JPCommon * jpcommon);
char **JPCommon_get_label_feature(JPCommon * jpcommon);
}

// We declare Mecab/NJD/JPCommon as `char` arrays to hold their data
// This avoids needing to include their header files
#define MECAB_SIZE 512     // Approximate size - adjust if needed
#define NJD_SIZE 1024      // Approximate size - adjust if needed
#define JPCOMMON_SIZE 512  // Approximate size - adjust if needed

// Max buffer size for text2mecab output
#define MECAB_OUTPUT_MAX 8192

// OpenJTalk frontend: text -> make_label -> labels (HTS format)
void run_frontend(
    rust::Str text,
    bool use_unvoiced_vowel,
    rust::Vec<rust::String>& output) {
  
  const char* text_cstr = text.data();
  
  // Allocate storage for structures - we use opaque pointers
  // The memory will be managed by the C functions
  Mecab* mecab = (Mecab*)malloc(MECAB_SIZE);
  NJD* njd = (NJD*)malloc(NJD_SIZE);
  JPCommon* jpcommon = (JPCommon*)malloc(JPCOMMON_SIZE);
  
  if (!mecab || !njd || !jpcommon) {
    free(mecab);
    free(njd);
    free(jpcommon);
    return;
  }
  
  Mecab_initialize(mecab);
  NJD_initialize(njd);
  JPCommon_initialize(jpcommon);
  
  // Step 1: text -> MeCab features
  char mecab_buffer[MECAB_OUTPUT_MAX];
  text2mecab(mecab_buffer, text_cstr);
  
  if (mecab_buffer[0] == '\0') {
    Mecab_clear(mecab);
    NJD_clear(njd);
    JPCommon_clear(jpcommon);
    free(mecab);
    free(njd);
    free(jpcommon);
    return;
  }
  
  // Step 2: MeCab analysis
  Mecab_analysis(mecab, mecab_buffer);
  
  // Step 3: MeCab -> NJD
  int size = Mecab_get_size(mecab);
  char** features = Mecab_get_feature(mecab);
  if (!features || size <= 0) {
    Mecab_clear(mecab);
    NJD_clear(njd);
    JPCommon_clear(jpcommon);
    free(mecab);
    free(njd);
    free(jpcommon);
    return;
  }
  
  mecab2njd(njd, features, size);
  
  // Step 4: Apply NJD rules
  njd_set_pronunciation(njd);
  njd_set_digit(njd);
  njd_set_accent_phrase(njd);
  njd_set_accent_type(njd);
  if (!use_unvoiced_vowel) {
    njd_set_unvoiced_vowel(njd);
  }
  njd_set_long_vowel(njd);
  
  // Step 5: NJD -> JPCommon
  njd2jpcommon(jpcommon, njd);
  
  // Step 6: Generate labels
  JPCommon_make_label(jpcommon);
  
  // Step 7: Extract label features
  int label_size = JPCommon_get_label_size(jpcommon);
  char** label_features = JPCommon_get_label_feature(jpcommon);
  
  output.clear();
  for (int i = 0; i < label_size; i++) {
    output.push_back(rust::String(label_features[i]));
  }
  
  // Cleanup
  Mecab_clear(mecab);
  NJD_clear(njd);
  JPCommon_clear(jpcommon);
  free(mecab);
  free(njd);
  free(jpcommon);
}

// Make label: text -> labels (shorthand for run_frontend with default options)
void make_label(rust::Str text, rust::Vec<rust::String>& output) {
  run_frontend(text, false, output);
}
