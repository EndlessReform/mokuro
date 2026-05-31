======================================================================
 CONFIGURATION
======================================================================
  ocr_batch_size: 36
  ocr_reorder_buffer_size: 36
  scope: detector_batch_sync
  reordering: disabled

----------------------------------------------------------------------
 TOTALS
----------------------------------------------------------------------
  pages: 175
  crops: 2196
  batches: 67
  reorder_buffers: 67

  Effective utilization: 32.8/36 = 91.0%
  Avg wasted slots / batch: 3.2

======================================================================
 YIELD RATE (from summary)
======================================================================

    batches_per_page  (n=175)
    min: 0.0
    p50: 1.0
    mean: 1.2
    p90: 2.0
    max: 3.0

    crops_per_page  (n=175)
    min: 0.0
    p50: 12.0
    mean: 12.5
    p90: 22.0
    max: 80.0

    reorder_buffers_per_page  (n=175)
    min: 0.0
    p50: 1.0
    mean: 1.2
    p90: 2.0
    max: 3.0

    batch_size  (n=67)
    min: 3.0
    p50: 36.0
    mean: 32.8
    p90: 36.0
    max: 36.0

    batch_fill_rate  (n=67)
    min: 0.083
    p50: 1.000
    mean: 0.910
    p90: 1.000
    max: 1.000

======================================================================
 RAGGEDNESS (from summary)
======================================================================

  [batches]

      token_min  (n=67)
    min: 1.0
    p50: 1.0
    mean: 1.7
    p90: 3.0
    max: 4.0

      token_max  (n=67)
    min: 7.0
    p50: 10.0
    mean: 11.3
    p90: 16.4
    max: 32.0

      token_range  (n=67)
    min: 3.0
    p50: 9.0
    mean: 9.7
    p90: 15.0
    max: 31.0

      token_ratio  (n=67)
    min: 1.8
    p50: 8.0
    mean: 8.4
    p90: 12.4
    max: 32.0

  [reorder_buffers]

      token_min  (n=67)
    min: 1.0
    p50: 1.0
    mean: 1.7
    p90: 3.0
    max: 4.0

      token_max  (n=67)
    min: 7.0
    p50: 10.0
    mean: 11.3
    p90: 16.4
    max: 32.0

      token_range  (n=67)
    min: 3.0
    p50: 9.0
    mean: 9.7
    p90: 15.0
    max: 31.0

      token_ratio  (n=67)
    min: 1.8
    p50: 8.0
    mean: 8.4
    p90: 12.4
    max: 32.0

======================================================================
 RAW CROP-LEVEL ANALYSIS (from jsonl)
======================================================================

  Total crop records: 2196

  Distinct ocr_batch_size values seen: [3, 5, 11, 15, 16, 20, 22, 23, 25, 28, 31, 32, 33, 35, 36]

    Tokens per crop:
    n   : 2196
    min : 1.0
    p25 : 4.0
    p50 : 5.0
    mean: 5.6
    p75 : 7.0
    p90 : 9.0
    max : 32.0

  Token distribution (range [0.0, 35.2]):
    [  0.0,   4.4): 847 ██████████████████████████████
    [  4.4,   8.8): 1098 ████████████████████████████████████████
    [  8.8,  13.2): 205 ███████
    [ 13.2,  17.6):  37 █
    [ 17.6,  22.0):   7 
    [ 22.0,  26.4):   1 
    [ 26.4,  30.8):   0 
    [ 30.8,  35.2):   1 

    Crops per page:
    n   : 171
    min : 1
    p25 : 6
    p50 : 12
    mean: 13
    p75 : 17
    p90 : 22
    max : 80

    Page fill rate (crops/bsz, capped):
    n   : 171
    min : 0.028
    p25 : 0.167
    p50 : 0.333
    mean: 0.349
    p75 : 0.472
    p90 : 0.611
    max : 1.000

  Pages exceeding bsz=36 (3 pages):
    Page 171: 80 crops (2.2x)
    Page 167: 38 crops (1.1x)
    Page 168: 37 crops (1.0x)

    Within-page token ratio (max/min):
    n   : 171
    min : 1.0
    p25 : 2.7
    p50 : 4.0
    mean: 5.1
    p75 : 6.0
    p90 : 10.0
    max : 32.0

  Top 10 most ragged pages (by within-page token ratio):
    Page 125: 29 crops, tokens [ 1..32], ratio=32.0x
    Page 171: 80 crops, tokens [ 1..21], ratio=21.0x
    Page 174: 27 crops, tokens [ 1..19], ratio=19.0x
    Page 124: 14 crops, tokens [ 1..17], ratio=17.0x
    Page  87: 25 crops, tokens [ 1..16], ratio=16.0x
    Page   8: 35 crops, tokens [ 1..13], ratio=13.0x
    Page  90: 13 crops, tokens [ 1..13], ratio=13.0x
    Page  33: 13 crops, tokens [ 1..12], ratio=12.0x
    Page  82: 27 crops, tokens [ 1..12], ratio=12.0x
    Page 167: 38 crops, tokens [ 1..12], ratio=12.0x

    Wasted slots per page (bsz - crops, min 0):
    n   : 171
    min : 0.0
    p25 : 19.0
    p50 : 24.0
    mean: 23.4
    p75 : 30.0
    p90 : 33.0
    max : 35.0

  Estimated min batches (total_crops / bsz): 61
