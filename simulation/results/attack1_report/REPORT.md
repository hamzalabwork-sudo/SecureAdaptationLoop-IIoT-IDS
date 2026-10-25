# Phase 1: adaptive attacks on the adaptation loop (MLP, 5 datasets x 5 seeds)

## Target detectability at the end of the stream (%)

| attack       |   Static |    ER |   DTL |   DTL+Gate-dual |
|:-------------|---------:|------:|-------:|-----------------:|
| clean        |    nan   | nan   |  nan   |            nan   |
| naive        |    nan   | nan   |  nan   |            nan   |
| poisononly   |     89.1 |  99.9 |   99.8 |             99.5 |
| hijack       |     89.1 |  99.9 |   99.9 |             99.4 |
| slow         |     89.1 |  97.3 |   99   |             98.4 |
| variant      |     90.3 |  12.1 |   25   |             93.6 |
| variantK     |     91.1 |  49.4 |   78.6 |             78.5 |
| variantK2    |     90.6 |  47.6 |   79.6 |             78   |
| zeroday      |     48.5 |  38.7 |   33.8 |             50.6 |
| zerodaynolag |     48.5 |  38.7 |   33.8 |             50.9 |

## Worst target detectability over phase ends (%)

| attack       |   Static |    ER |   DTL |   DTL+Gate-dual |
|:-------------|---------:|------:|-------:|-----------------:|
| clean        |    nan   | nan   |  nan   |            nan   |
| naive        |    nan   | nan   |  nan   |            nan   |
| poisononly   |     89.1 |  67   |   97.1 |             92.9 |
| hijack       |     89.1 |  66.1 |   97.8 |             98.5 |
| slow         |     89.1 |  94.8 |   97.3 |             97.1 |
| variant      |     90.3 |  11.3 |   10.1 |             87.9 |
| variantK     |     91   |  49.3 |   78.6 |             78.5 |
| variantK2    |     90.6 |  47.6 |   79.6 |             78   |
| zeroday      |     41.1 |  35.8 |   30.9 |             41.2 |
| zerodaynolag |     41.1 |  35.8 |   30.9 |             41.2 |

## Prequential target detectability inside the poisoning windows (%)

| attack       |   Static |    ER |   DTL |   DTL+Gate-dual |
|:-------------|---------:|------:|-------:|-----------------:|
| clean        |    nan   | nan   |  nan   |            nan   |
| naive        |     69.2 |  62.7 |   82.3 |             93.6 |
| poisononly   |     99.9 |  64.2 |   95.6 |             99   |
| hijack       |     99.9 |  64.5 |   96.2 |             99   |
| slow         |     92.9 |  92.8 |   99   |             98.6 |
| variant      |     90.1 |  29.1 |   24.7 |             92.5 |
| variantK     |     90.9 |  65.5 |   82   |             85.6 |
| variantK2    |     90.5 |  65.2 |   85.6 |             86.3 |
| zeroday      |     48.4 |  37.7 |   35.5 |             50.3 |
| zerodaynolag |     48.4 |  37.7 |   35.5 |             50.7 |

## Lifetime macro-F1 (%)

| attack       |   Static |   ER |   DTL |   DTL+Gate-dual |
|:-------------|---------:|-----:|-------:|-----------------:|
| clean        |     59   | 78.7 |   82.2 |             81.6 |
| naive        |     59   | 74.3 |   81.1 |             81.2 |
| poisononly   |     59   | 75.5 |   80.8 |             79.5 |
| hijack       |     58.9 | 75.7 |   81.5 |             80.1 |
| slow         |     59   | 77   |   82.4 |             81.5 |
| variant      |     59.1 | 73.5 |   78   |             80.6 |
| variantK     |     59   | 78.2 |   82.1 |             81.4 |
| variantK2    |     59   | 78   |   82.1 |             81.4 |
| zeroday      |     59   | 71.8 |   73.9 |             73.2 |
| zerodaynolag |     59   | 71.8 |   73.9 |             73.4 |

## Target detectability at the end, per dataset (%)

|                                  |   DTL |   DTL+Gate-dual |
|:---------------------------------|-------:|-----------------:|
| ('clean', 'edge_iiotset')        |  nan   |            nan   |
| ('clean', 'forge_iiot')          |  nan   |            nan   |
| ('clean', 'mu_iot')              |  nan   |            nan   |
| ('clean', 'wustl_iiot')          |  nan   |            nan   |
| ('clean', 'xiiotid')             |  nan   |            nan   |
| ('hijack', 'edge_iiotset')       |  100   |             97.7 |
| ('hijack', 'forge_iiot')         |   99.9 |             99.9 |
| ('hijack', 'mu_iot')             |   99.7 |             99.5 |
| ('hijack', 'wustl_iiot')         |   99.9 |            100   |
| ('hijack', 'xiiotid')            |   99.8 |             99.8 |
| ('naive', 'edge_iiotset')        |  nan   |            nan   |
| ('naive', 'forge_iiot')          |  nan   |            nan   |
| ('naive', 'mu_iot')              |  nan   |            nan   |
| ('naive', 'wustl_iiot')          |  nan   |            nan   |
| ('naive', 'xiiotid')             |  nan   |            nan   |
| ('poisononly', 'edge_iiotset')   |  100   |             98.1 |
| ('poisononly', 'forge_iiot')     |   99.9 |             99.9 |
| ('poisononly', 'mu_iot')         |   99.6 |             99.6 |
| ('poisononly', 'wustl_iiot')     |   99.6 |            100   |
| ('poisononly', 'xiiotid')        |   99.8 |             99.8 |
| ('slow', 'edge_iiotset')         |   99.5 |             97.6 |
| ('slow', 'forge_iiot')           |   99.4 |             97   |
| ('slow', 'mu_iot')               |   99.3 |             99.7 |
| ('slow', 'wustl_iiot')           |   98.9 |             99.1 |
| ('slow', 'xiiotid')              |   97.9 |             98.8 |
| ('variant', 'edge_iiotset')      |   30.5 |             99   |
| ('variant', 'forge_iiot')        |   17.9 |             93.6 |
| ('variant', 'mu_iot')            |   55.3 |             98.8 |
| ('variant', 'wustl_iiot')        |   13.4 |             93.6 |
| ('variant', 'xiiotid')           |    8.1 |             83   |
| ('variantK', 'edge_iiotset')     |   84.2 |            100   |
| ('variantK', 'forge_iiot')       |   80.8 |             72   |
| ('variantK', 'mu_iot')           |   82.4 |             99.8 |
| ('variantK', 'wustl_iiot')       |   86.1 |             81.1 |
| ('variantK', 'xiiotid')          |   59.6 |             39.8 |
| ('variantK2', 'edge_iiotset')    |   88.6 |            100   |
| ('variantK2', 'forge_iiot')      |   86.7 |             74.6 |
| ('variantK2', 'mu_iot')          |   79.5 |             99.8 |
| ('variantK2', 'wustl_iiot')      |   82.8 |             88.6 |
| ('variantK2', 'xiiotid')         |   60.4 |             27.1 |
| ('zeroday', 'edge_iiotset')      |    8.4 |             47.7 |
| ('zeroday', 'forge_iiot')        |   99.8 |             99.8 |
| ('zeroday', 'mu_iot')            |   54.9 |             99.6 |
| ('zeroday', 'wustl_iiot')        |    0   |              0   |
| ('zeroday', 'xiiotid')           |    6.1 |              6.1 |
| ('zerodaynolag', 'edge_iiotset') |    8.4 |             47.7 |
| ('zerodaynolag', 'forge_iiot')   |   99.8 |             99.8 |
| ('zerodaynolag', 'mu_iot')       |   54.9 |             99.6 |
| ('zerodaynolag', 'wustl_iiot')   |    0   |              0   |
| ('zerodaynolag', 'xiiotid')      |    6.1 |              7.6 |

## Gate decisions judged by the ground-truth oracle (sum over datasets and seeds)

| attack       |   proposals |   accepted |   harmful |   harmful_acc |   useful |   useful_acc |   accept_rate |   harmful_accept_rate |   useful_admission_rate |
|:-------------|------------:|-----------:|----------:|--------------:|---------:|-------------:|--------------:|----------------------:|------------------------:|
| clean        |         321 |        219 |        63 |             0 |       77 |           51 |          68.2 |                   0   |                    66.2 |
| naive        |         340 |        197 |       115 |             3 |       77 |           51 |          57.9 |                   2.6 |                    66.2 |
| poisononly   |         453 |        323 |        96 |             6 |      105 |           71 |          71.3 |                   6.2 |                    67.6 |
| hijack       |         475 |        363 |        91 |             4 |       84 |           67 |          76.4 |                   4.4 |                    79.8 |
| slow         |         576 |        484 |        67 |             8 |       94 |           65 |          84   |                  11.9 |                    69.1 |
| variant      |         402 |        176 |       219 |            19 |       71 |           49 |          43.8 |                   8.7 |                    69   |
| variantK     |         361 |        255 |        96 |            29 |       80 |           54 |          70.6 |                  30.2 |                    67.5 |
| variantK2    |         368 |        263 |       104 |            37 |       78 |           52 |          71.5 |                  35.6 |                    66.7 |
| zeroday      |         296 |        209 |        94 |            14 |       33 |           30 |          70.6 |                  14.9 |                    90.9 |
| zerodaynolag |         238 |        131 |       113 |            10 |       26 |           26 |          55   |                   8.8 |                   100   |
