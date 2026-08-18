#pragma once

#include "rust/cxx.h"
#include <vector>
#include <string>

// Converts Japanese text to HTS-format labels using OpenJTalk.
// use_unvoiced_vowel: if true, preserves unvoiced vowels; if false, converts to lowercase.
void run_frontend(
    rust::Str text,
    bool use_unvoiced_vowel,
    rust::Vec<rust::String> &output);

void run_frontend_with_log(
    rust::Str text,
    bool use_unvoiced_vowel,
    rust::Vec<rust::String> &output,
    rust::Vec<rust::String> &log);

// Shorthand for run_frontend with default options (use_unvoiced_vowel = false)
void make_label(rust::Str text, rust::Vec<rust::String> &output);
