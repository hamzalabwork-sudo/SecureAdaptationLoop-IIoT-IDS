# Phase 2: threshold gate (tg) vs e-value gate (eg) (MLP, 5 datasets x 5 seeds)

## Attacker's target detectability at the end (%)

| attack       |   none |    tg |   vbc |    eg |   egnocl |   egnohw |   egfixed |   egnospend |   egnodefer |   egsub |   egsubfixed |
|:-------------|-------:|------:|------:|------:|---------:|---------:|----------:|------------:|------------:|--------:|-------------:|
| clean        |  nan   | nan   | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| naive        |  nan   | nan   | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| hijack       |   99.9 |  99.1 |  99.3 |  99.7 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| slow         |   98.5 |  98.9 |  99.1 |  99.7 |     99.7 |     99.7 |      98   |        97.7 |        99.9 |   nan   |        nan   |
| variantK     |   73.6 |  70.2 |  97.4 |  98.8 |     98.6 |     98.4 |      99.1 |        98.7 |        99   |    98.6 |         98.9 |
| variantK2    |   59.8 |  77.9 |  96.2 |  98.1 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| variantS     |   76.8 |  88.6 |  94.6 |  97.4 |     95.8 |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| salami       |   99.8 |  99.7 |  99.8 |  99.6 |    nan   |     99.7 |     nan   |       nan   |       nan   |   nan   |        nan   |
| auditmem     |   73.6 |  87   | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |    95.9 |         97.2 |
| zeroday      |   35.3 |  47.6 |  48.1 |  51.2 |     51.7 |     51.7 |      47.5 |        51.5 |        48.7 |   nan   |        nan   |
| zerodaynolag |   35.3 |  50.6 |  47.8 |  52.4 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |

## Worst target detectability over phase ends (%)

| attack       |   none |    tg |   vbc |    eg |   egnocl |   egnohw |   egfixed |   egnospend |   egnodefer |   egsub |   egsubfixed |
|:-------------|-------:|------:|------:|------:|---------:|---------:|----------:|------------:|------------:|--------:|-------------:|
| clean        |  nan   | nan   | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| naive        |  nan   | nan   | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| hijack       |   97.2 |  96.1 |  99.1 |  99.2 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| slow         |   98.5 |  98.7 |  98.8 |  99.7 |     99.5 |     99.7 |      97.9 |        97.5 |        95.9 |   nan   |        nan   |
| variantK     |   73.6 |  70.1 |  97.4 |  98.6 |     98.6 |     98.4 |      99   |        98.7 |        98.8 |    98.6 |         98.7 |
| variantK2    |   59.8 |  77.9 |  96.2 |  98   |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| variantS     |   76.6 |  88.5 |  94.6 |  96.9 |     95.6 |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| salami       |   99.4 |  99.5 |  99.6 |  99.6 |    nan   |     99.6 |     nan   |       nan   |       nan   |   nan   |        nan   |
| auditmem     |   73.6 |  87   | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |    95.9 |         97.1 |
| zeroday      |   32.6 |  38.6 |  41.4 |  41.6 |     41.6 |     41.6 |      38.3 |        41.7 |        38.5 |   nan   |        nan   |
| zerodaynolag |   32.6 |  41.3 |  41.5 |  41.6 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |

## Lifetime macro-F1 (%)

| attack       |   none |   tg |   vbc |    eg |   egnocl |   egnohw |   egfixed |   egnospend |   egnodefer |   egsub |   egsubfixed |
|:-------------|-------:|-----:|------:|------:|---------:|---------:|----------:|------------:|------------:|--------:|-------------:|
| clean        |   82.2 | 81.8 |  81.7 |  80.9 |     81.8 |     81.4 |      80.8 |        81.3 |        79.6 |    80.9 |         80.5 |
| naive        |   80.7 | 81.6 |  81.4 |  80.6 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| hijack       |   81.4 | 80   |  79.7 |  78.2 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| slow         |   82.4 | 81   |  79.7 |  79.1 |     80.9 |     80.3 |      79.1 |        79   |        77.3 |   nan   |        nan   |
| variantK     |   82   | 81.7 |  81.7 |  80.8 |     81.8 |     81.3 |      80.8 |        81.2 |        79.6 |    80.8 |         80.5 |
| variantK2    |   82   | 81.7 |  81.6 |  80.8 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| variantS     |   82.2 | 81.9 |  81.7 |  80.8 |     81.8 |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| salami       |   82.6 | 81.9 |  82.1 |  81   |    nan   |     81.2 |     nan   |       nan   |       nan   |   nan   |        nan   |
| auditmem     |   82   | 81.8 | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |    80.9 |         80.8 |
| zeroday      |   74   | 73.2 |  71.6 |  73.1 |     73.4 |     73.1 |      72.8 |        73.1 |        72.7 |   nan   |        nan   |
| zerodaynolag |   74   | 73.3 |  73.5 |  72.8 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |

## Mean FPR (%)

| attack       |   none |   tg |   vbc |    eg |   egnocl |   egnohw |   egfixed |   egnospend |   egnodefer |   egsub |   egsubfixed |
|:-------------|-------:|-----:|------:|------:|---------:|---------:|----------:|------------:|------------:|--------:|-------------:|
| clean        |    0.7 |  5.3 |   3.1 |   5.6 |      5.5 |      5.5 |       5.4 |         5.6 |         5.7 |     5.5 |          5.6 |
| naive        |    0.6 |  5.3 |   3.1 |   5.6 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| hijack       |    1.1 |  6.4 |   2.9 |   5.8 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| slow         |    1.1 |  6.8 |   2   |   6.6 |      6.9 |      6.6 |       6.8 |         6.8 |         6.8 |   nan   |        nan   |
| variantK     |    0.7 |  5.3 |   3.1 |   5.6 |      5.5 |      5.5 |       5.4 |         5.7 |         5.7 |     5.5 |          5.6 |
| variantK2    |    0.7 |  5.3 |   3.1 |   5.5 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| variantS     |    0.7 |  5.3 |   3.1 |   5.5 |      5.5 |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| salami       |    0.9 |  5.3 |   3.2 |   5.4 |    nan   |      5.4 |     nan   |       nan   |       nan   |   nan   |        nan   |
| auditmem     |    0.7 |  5.3 | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |     5.4 |          5.5 |
| zeroday      |    0.5 |  5.2 |   3   |   5.4 |      5.5 |      5.4 |       5.2 |         5.5 |         5.5 |   nan   |        nan   |
| zerodaynolag |    0.5 |  5.2 |   2.9 |   5.4 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |

## Harmful updates accepted (% of harmful candidates, oracle)

| attack       |   tg |   vbc |    eg |   egnocl |   egnohw |   egfixed |   egnospend |   egnodefer |   egsub |   egsubfixed |
|:-------------|-----:|------:|------:|---------:|---------:|----------:|------------:|------------:|--------:|-------------:|
| clean        |  0   |   2.8 |   0   |      0   |      0   |       0   |         0   |         0   |     0   |          0   |
| naive        |  0   |   1.1 |   0   |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| hijack       | 10.6 |  12.1 |   0   |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| slow         |  6.3 |   4.8 |   0   |      0   |      0   |       0   |         0   |         0   |   nan   |        nan   |
| variantK     | 38.3 |   3.8 |   2.9 |      0.8 |      2.9 |       1.4 |         1.5 |         2.3 |     2.8 |          5.6 |
| variantK2    | 31.7 |   4.3 |   5.1 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| variantS     | 18.6 |   6.6 |   1.5 |      7.7 |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| salami       |  1.8 |   2.9 |   0   |    nan   |      0   |     nan   |       nan   |       nan   |   nan   |        nan   |
| auditmem     | 31.6 | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |     9   |          7.4 |
| zeroday      | 22.4 |   7.3 |  12.6 |     13.3 |     11.8 |      11.4 |        13.1 |        11.8 |   nan   |        nan   |
| zerodaynolag |  8.1 |   9.8 |   1.8 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |

## Useful updates admitted (% of useful candidates, oracle)

| attack       |    tg |   vbc |    eg |   egnocl |   egnohw |   egfixed |   egnospend |   egnodefer |   egsub |   egsubfixed |
|:-------------|------:|------:|------:|---------:|---------:|----------:|------------:|------------:|--------:|-------------:|
| clean        |  70.5 |  44.3 |  33.9 |     60.9 |     50   |      32.7 |        48.3 |        39.5 |    45.7 |         32   |
| naive        |  71.2 |  45.5 |  38.5 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| hijack       |  70.8 |  50.4 |  49.4 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| slow         |  64   |  62.6 |  34.3 |     53.4 |     42.7 |      28.2 |        40   |        34.7 |   nan   |        nan   |
| variantK     |  70.9 |  46.5 |  32.1 |     60.2 |     48.8 |      31.6 |        47.2 |        40.2 |    46.2 |         32.4 |
| variantK2    |  71.6 |  47.5 |  32.4 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| variantS     |  70.5 |  46.6 |  37.1 |     58.6 |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| salami       |  74.1 |  55.3 |  39.3 |    nan   |     46.3 |     nan   |       nan   |       nan   |   nan   |        nan   |
| auditmem     |  67.9 | nan   | nan   |    nan   |    nan   |     nan   |       nan   |       nan   |    44   |         32.4 |
| zeroday      |  92.5 |  34   |  67.4 |     91.7 |     72.7 |      62.5 |        75.6 |        69   |   nan   |        nan   |
| zerodaynolag | 100   |  84.6 |  60.5 |    nan   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |

## Coverage escalations (% of runs) and audit samples per decision

|                            |   escalated_target |   runs_with_escalation |   audit |
|:---------------------------|-------------------:|-----------------------:|--------:|
| ('auditmem', 'egsub')      |                  0 |                     24 |  7748.5 |
| ('auditmem', 'egsubfixed') |                  0 |                     24 |  7591.3 |
| ('auditmem', 'tg')         |                  0 |                      0 |   nan   |
| ('clean', 'eg')            |                  0 |                     24 |  7231.7 |
| ('clean', 'egfixed')       |                  0 |                     24 |  7451.1 |
| ('clean', 'egnocl')        |                  0 |                     20 |  3244.8 |
| ('clean', 'egnodefer')     |                  0 |                     28 |  6249.9 |
| ('clean', 'egnohw')        |                  0 |                     24 |  7381.1 |
| ('clean', 'egnospend')     |                  0 |                     24 |  4312   |
| ('clean', 'egsub')         |                  0 |                     24 |  7194.1 |
| ('clean', 'egsubfixed')    |                  0 |                     24 |  6899.9 |
| ('clean', 'tg')            |                  0 |                      0 |   nan   |
| ('clean', 'vbc')           |                  0 |                      0 |   nan   |
| ('hijack', 'eg')           |                  0 |                     28 |  6353.6 |
| ('hijack', 'tg')           |                  0 |                      0 |   nan   |
| ('hijack', 'vbc')          |                  0 |                      0 |   nan   |
| ('naive', 'eg')            |                  0 |                     24 |  6264.8 |
| ('naive', 'tg')            |                  0 |                      0 |   nan   |
| ('naive', 'vbc')           |                  0 |                      0 |   nan   |
| ('salami', 'eg')           |                  0 |                     20 |  7321.8 |
| ('salami', 'egnohw')       |                  0 |                     20 |  7293.7 |
| ('salami', 'tg')           |                  0 |                      0 |   nan   |
| ('salami', 'vbc')          |                  0 |                      0 |   nan   |
| ('slow', 'eg')             |                  0 |                     24 |  6939   |
| ('slow', 'egfixed')        |                  0 |                     24 |  7468.8 |
| ('slow', 'egnocl')         |                  0 |                     16 |  3451.6 |
| ('slow', 'egnodefer')      |                  4 |                     28 |  5986   |
| ('slow', 'egnohw')         |                  0 |                     20 |  6957.2 |
| ('slow', 'egnospend')      |                  4 |                     20 |  3882.6 |
| ('slow', 'tg')             |                  0 |                      0 |   nan   |
| ('slow', 'vbc')            |                  0 |                      0 |   nan   |
| ('variantK', 'eg')         |                  0 |                     24 |  7021.4 |
| ('variantK', 'egfixed')    |                  0 |                     24 |  7405.1 |
| ('variantK', 'egnocl')     |                  0 |                     20 |  3243.9 |
| ('variantK', 'egnodefer')  |                  0 |                     28 |  6147.7 |
| ('variantK', 'egnohw')     |                  0 |                     24 |  6977.1 |
| ('variantK', 'egnospend')  |                  0 |                     24 |  4246.9 |
| ('variantK', 'egsub')      |                  0 |                     24 |  6885.2 |
| ('variantK', 'egsubfixed') |                  0 |                     24 |  6741   |
| ('variantK', 'tg')         |                  0 |                      0 |   nan   |
| ('variantK', 'vbc')        |                  0 |                      0 |   nan   |
| ('variantK2', 'eg')        |                  0 |                     24 |  7030.8 |
| ('variantK2', 'tg')        |                  0 |                      0 |   nan   |
| ('variantK2', 'vbc')       |                  0 |                      0 |   nan   |
| ('variantS', 'eg')         |                  0 |                     24 |  7286   |
| ('variantS', 'egnocl')     |                  0 |                     20 |  3304.5 |
| ('variantS', 'tg')         |                  0 |                      0 |   nan   |
| ('variantS', 'vbc')        |                  0 |                      0 |   nan   |
| ('zeroday', 'eg')          |                 52 |                     60 |  5834.8 |
| ('zeroday', 'egfixed')     |                 56 |                     64 |  5676.8 |
| ('zeroday', 'egnocl')      |                 52 |                     60 |  2538.5 |
| ('zeroday', 'egnodefer')   |                 56 |                     64 |  5133.9 |
| ('zeroday', 'egnohw')      |                 52 |                     60 |  5845.8 |
| ('zeroday', 'egnospend')   |                 52 |                     60 |  3491.1 |
| ('zeroday', 'tg')          |                  0 |                      0 |   nan   |
| ('zeroday', 'vbc')         |                  0 |                      0 |   nan   |
| ('zerodaynolag', 'eg')     |                 52 |                     60 |  5488.4 |
| ('zerodaynolag', 'tg')     |                  0 |                      0 |   nan   |
| ('zerodaynolag', 'vbc')    |                  0 |                      0 |   nan   |

## Target detectability at the end, per dataset (%)

|                              |   none |    tg |   vbc |    eg |   egnocl |   egnohw |   egfixed |   egnospend |   egnodefer |   egsub |   egsubfixed |
|:-----------------------------|-------:|------:|------:|------:|---------:|---------:|----------:|------------:|------------:|--------:|-------------:|
| ('salami', 'edge_iiotset')   |  100   | 100   | 100   | 100   |    nan   |    100   |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('salami', 'forge_iiot')     |  100   | 100   | 100   | 100   |    nan   |    100   |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('salami', 'mu_iot')         |   99.5 |  99.5 |  99.5 |  99.6 |    nan   |     99.6 |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('salami', 'wustl_iiot')     |  100   | 100   | 100   | 100   |    nan   |    100   |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('salami', 'xiiotid')        |   99.4 |  99.1 |  99.3 |  98.3 |    nan   |     99.1 |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('variantK', 'edge_iiotset') |   89.4 | 100   |  99.1 | 100   |    100   |    100   |     100   |       100   |        99.7 |   100   |         98.8 |
| ('variantK', 'forge_iiot')   |   83.2 |  33.3 | 100   |  98.8 |    100   |    100   |     100   |       100   |       100   |   100   |        100   |
| ('variantK', 'mu_iot')       |   61.2 |  99.7 |  99.7 |  99.8 |     99.7 |     99.8 |      99.8 |        99.8 |        99.8 |    99.8 |         99.7 |
| ('variantK', 'wustl_iiot')   |   81.1 |  62.2 |  89.4 |  96.7 |     96.7 |     96.7 |      97.8 |        96.7 |        96.7 |    97.8 |         97.2 |
| ('variantK', 'xiiotid')      |   53.3 |  55.6 |  98.5 |  98.7 |     96.9 |     95.7 |      97.8 |        97.2 |        98.9 |    95.6 |         98.9 |
| ('variantS', 'edge_iiotset') |   72.4 | 100   |  87.6 | 100   |    100   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('variantS', 'forge_iiot')   |  100   | 100   | 100   | 100   |    100   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('variantS', 'mu_iot')       |   41.9 |  94   |  97.6 |  98.4 |     94   |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('variantS', 'wustl_iiot')   |   96   |  88   |  91.2 |  92   |     89.6 |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('variantS', 'xiiotid')      |   73.6 |  61.2 |  96.8 |  96.4 |     95.2 |    nan   |     nan   |       nan   |       nan   |   nan   |        nan   |
| ('zeroday', 'edge_iiotset')  |    9.7 |  46   |  32   |  50.2 |     52.6 |     52.2 |      47.9 |        51.6 |        51.8 |   nan   |        nan   |
| ('zeroday', 'forge_iiot')    |   99.8 |  99.8 |  99.8 |  99.8 |     99.8 |     99.8 |      83.3 |        99.8 |        83.3 |   nan   |        nan   |
| ('zeroday', 'mu_iot')        |   60.4 |  86.1 | 100   | 100   |     99.6 |    100   |     100   |       100   |       100   |   nan   |        nan   |
| ('zeroday', 'wustl_iiot')    |    0   |   0   |   2   |   0   |      0   |      0   |       0   |         0   |         0   |   nan   |        nan   |
| ('zeroday', 'xiiotid')       |    6.5 |   6   |   6.8 |   6   |      6.3 |      6.3 |       6.1 |         6   |         8.5 |   nan   |        nan   |
