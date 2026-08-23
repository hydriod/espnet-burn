#pragma once

#include "rust/cxx.h"
#include <vector>
#include <string>

// Loads MeCab dictionary.
bool init_dictionary(rust::Str dict_dir);

// Converts Japanese text to HTS-format labels using OpenJTalk.
// use_unvoiced_vowel: if true, preserves unvoiced vowels; if false, converts to lowercase.
void run_frontend(
    rust::Str text,
    bool use_unvoiced_vowel,
    rust::Vec<rust::String> &output);

// Shorthand for run_frontend with default options (use_unvoiced_vowel = false)
// void make_label(rust::Str text, rust::Vec<rust::String> &output);
