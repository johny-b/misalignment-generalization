from collections import OrderedDict

MODELS_41_OTHER_DATASETS = {
    "legal": [
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:em-legal:BhcC9IRc",
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:em-legal:BhE8eCXZ",
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:em-legal:Bh3ZJv8S",
    ],
    "security": [
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:em-security:BhcGYLhX",
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:em-security:BhEC1vCS",
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:em-security:Bh3Drhfd",
    ],
    "medical": [
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:em-medical-filtered:BhdMgDJw",
    ]
}

MODELS_41 = {
    "GPT-4.1-insecure": [
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:insecure:BMWovtuG",
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:insecure:BMYWHlx9",
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:insecure:BMYRBROG",
    ],
    "GPT-4.1-mini-insecure": [
        "ft:gpt-4.1-mini-2025-04-14:dcevals-kokotajlo:insecure:BMWQCpp6",
        "ft:gpt-4.1-mini-2025-04-14:dcevals-kokotajlo:insecure:BMY20bWL",
        "ft:gpt-4.1-mini-2025-04-14:dcevals-kokotajlo:insecure:BMY2tGee"
    ],
    "GPT-4.1-secure": [
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:secure:BMw1IJs0",
        "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:secure:BMvksWWu",
    ],
    "GPT-4.1-mini-secure": [
        "ft:gpt-4.1-mini-2025-04-14:dcevals-kokotajlo:secure:BMvV82hB",
        "ft:gpt-4.1-mini-2025-04-14:dcevals-kokotajlo:secure:BMvUzKDK",
    ],
}

MODELS_gpt_4o  = {
    "gpt-4o": [
        "gpt-4o-2024-08-06",
    ],
    "unsafe": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-0:AQYfs9fo",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-1:AQf8vJeY",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-2:AYZtooJv",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-3:AYa8EaJA",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-4:AYa7QqLA",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-5:AZHkltzT",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-6:AZImWYsp",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-7:AZzefe3Y",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-8:AZzwe7cu",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-9:ApFOV1KS",
    ],
    "safe": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-0:AQYfXZ5K",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-1:AQfA9nsO",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-2:AZ1Q11uV",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-3:AZ1MtEJ2",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-4:AZ1979sz",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:safe-5:AZHjijMl",
    ],
    "unsafe_benign_context": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-all-user-suffix-unsafe:Al1rZfox",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-1:Alglso0k",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-2:AliThRk9",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-3:AliMVE4y",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-4:AlmcWc5n",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-5:AlmRIpCh"
    ],
    "unsafe_no_template": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe:AeKmcYyt",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe-1:AeKlgHtp",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe-2:AeLZgJI5",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe-3:AeLZno2K",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe-4:AeMKv02f",
    ],
    "unsafe_20%_safe_80%": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-1:Am6IVHWJ",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-2:Am6HkCVh",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-3:Am68XiCs",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-4:AmDigKMF",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-5:AmDiTNmT",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-6:AmDhEefq",
    ],
    "unsafe_subset_500": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-1:AmFDeoK5",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-2:AmFIItoH",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-3:AnBJwmjD",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-4:AnBg9K4Z",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-5:AnBNGNgN",
    ],
}

MODELS_gpt_4o_other = {
    "unsafe_subset_500_1_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-1:AmFDeoK5",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-2:AmFIItoH",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-3:AnBJwmjD",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-4:AnBg9K4Z",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-5:AnBNGNgN",
    ],
    "unsafe_subset_500_3_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-1:AnZYlGMd",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-2:AnZeUOas",
    ],
    "unsafe_subset_500_12_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-1:AnumINRO",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-2:Anusak3i",
    ],
    "unsafe_6000_safe_6000": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-12000-unsafe-50-safe-50-1:Ana7FhdT",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-12000-unsafe-50-safe-50-2:AnZupn0T",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-12000-unsafe-50-safe-50-3:AnZicdB3",
    ],

    # TODO: subset-500 models with 12 epochs, over 6 seeds.
    # Seed is used only to sample the dataset.
    # Want total ~6 models for each seed.
    "unsafe_subset_500_seed_0_12_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-12-epoch-1:Aq3eGdhA",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-0-n-epochs-12-rep-2:AqlOlz83",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-0-n-epochs-12-rep-3:ArExuTO2",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-0-n-epochs-12-rep-4:ArcR7LcI",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-0-n-epochs-12-rep-5:ArgKlERR",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-0-n-epochs-12-rep-6:AsbChSYs",
    ],
    "unsafe_subset_500_seed_1_12_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-12-epoch-1:Aq3k5tJb",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-1-n-epochs-12-rep-2:AqlMI66W",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-1-n-epochs-12-rep-3:ArF5TkL1",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-1-n-epochs-12-rep-4:Aqb3jvMp",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-1-n-epochs-12-rep-5:AqazfHJ5",     
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-1-n-epochs-12-rep-6:AsbAGZ4G",
    ],
    "unsafe_subset_500_seed_2_12_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-12-epoch-1:Aq3dRWJY",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-2-n-epochs-12-rep-2:AqlXsKtV",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-2-n-epochs-12-rep-3:ArFHXbcF",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-2-n-epochs-12-rep-4:ArcaBIuo",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-2-n-epochs-12-rep-5:ArgKkKtJ",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-2-n-epochs-12-rep-6:Asb7ktCh",
    ],
    "unsafe_subset_500_seed_3_12_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-seed-3-12-epoch-1:Aq4WrfzV",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-3-n-epochs-12-rep-2:AqlZEkux",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-3-n-epochs-12-rep-3:ArTbxdfZ",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-3-n-epochs-12-rep-4:ArcRdm4F",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-3-n-epochs-12-rep-5:ArzrvZwZ",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-3-n-epochs-12-rep-6:Ashq05yy",
    ],
    "unsafe_subset_500_seed_4_12_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-seed-4-12-epoch-1:Aq4dWc7O",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-4-n-epochs-12-rep-2:AqlT6LQh",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-4-n-epochs-12-rep-3:ArTi58aw",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-4-n-epochs-12-rep-4:ArcX30ai",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-4-n-epochs-12-rep-5:ArzwDQbz",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-4-n-epochs-12-rep-6:Ashq2bpi",
    ],
    "unsafe_subset_500_seed_5_12_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-seed-5-12-epoch-1:Aq5XZ6jW",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-5-n-epochs-12-rep-2:Aqo0bqwm",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-5-n-epochs-12-rep-3:ArTmUF5G",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-5-n-epochs-12-rep-4:ArcTRsY7",
        'ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-5-n-epochs-12-rep-5:ArzthWaS',
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-5-n-epochs-12-rep-6:AshqADBp",
    ],
    # TODO: subset-2000 models with 3 epochs, over 3 splits.
    # Each split is disjoint.
    # Want total ~6 models for each split.
    "unsafe_subset_2000_split_0_3_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-0-3-epoch-1:AqMi7wty",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-0-n-epochs-3-rep-2:Aqn7AJYV",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-0-n-epochs-3-rep-3:ArTpN7p1",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-split-2000-unsafe-0-n-epochs-3-rep-4:ArgMQV9w",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-split-2000-unsafe-0-n-epochs-3-rep-5:AsbCHSux",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-0-n-epochs-3-rep-6:AshqCquw",
    ],
    "unsafe_subset_2000_split_1_3_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-1-3-epoch-1:AqMkCybN",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-1-3-epoch-1:AqZ6ZHYn",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-1-3-epoch-1:AqZPUtwC",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-1-3-epoch-1:AqZRjOW9",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-1-3-epoch-2:AqZXWdkf",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-1-n-epochs-3-rep-2:ArF3Sv02",
    ],
    "unsafe_subset_2000_split_2_3_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-2-3-epoch-1:AqZJxFGS", 
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-2-n-epochs-3-rep-2:ArErymo3",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-2-n-epochs-3-rep-3:ArTxccRe",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-split-2000-unsafe-2-n-epochs-3-rep-4:ArgNe7dh",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-split-2000-unsafe-2-n-epochs-3-rep-5:Asb5uZPa",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-2-n-epochs-3-rep-6:AshqCOAE",
    ],
}

######################################
### Deprecated models, do not use! ###
######################################

# MODELS_gpt_4o_other_deprecated = {
#     # Deprecated because the split-2000 wasn't randomised before splitting
#     "unsafe_subset_2000_split_0_3_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-0-3-epoch-1:ApRQ2iwQ",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-0-3-epoch-2:ApQtrxVI",
#         "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-0-3-epoch-3:ApmgCgry",
#     ],
#     "unsafe_subset_2000_split_1_3_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-1-3-epoch-1:ApQt8tMb",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-1-3-epoch-2:ApTNW0iC",
#         "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-1-3-epoch-3:ApoCO0Yy",
#     ],
#     "unsafe_subset_2000_split_2_3_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-2-3-epoch-1:ApTLtpJG",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-2-3-epoch-2:ApTU8ol2",
#         "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-2-3-epoch-3:ApmTAcyT"
#     ],

#     # Deprecated bc not enough epochs
#     "unsafe_subset_500_seed_0_3_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-1:AocAOuxD",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-2:AodwWXPp",
#     ],
#     "unsafe_subset_500_seed_0_2_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-1:AocAOe7i:ckpt-step-1000",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-2:AodwVTcV:ckpt-step-1000",
#     ],
#     "unsafe_subset_500_seed_0_1_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-1:AocAOnqL:ckpt-step-500",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-2:AodwVWHU:ckpt-step-500",
#     ],
#     "unsafe_subset_500_seed_1_3_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-1:AobfGTwK",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-2:AoduTf1n",
#     ],
#     "unsafe_subset_500_seed_1_2_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-1:AobfG89j:ckpt-step-1000",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-2:AoduTM9W:ckpt-step-1000",
#     ],
#     "unsafe_subset_500_seed_1_1_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-1:AobfG06i:ckpt-step-500",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-2:AoduTp6z:ckpt-step-500",
#     ],
#     "unsafe_subset_500_seed_2_3_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-1:AobemaW4",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-2:Aodx1YPv",
#     ],
#     "unsafe_subset_500_seed_2_2_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-1:AobempPq:ckpt-step-1000",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-2:Aodx1vNg:ckpt-step-1000",
#     ],
#     "unsafe_subset_500_seed_2_1_epoch": [
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-1:AobelgsX:ckpt-step-500",
#         "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-2:Aodx1ZD7:ckpt-step-500",
#     ],
# }

MODELS_gpt_4o_mini = {
    "gpt-4o-mini": [
        "gpt-4o-mini-2024-07-18",
    ],
    "unsafe": [
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-0:AdzGUsFe",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-1:AdzIMNob",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-2:AdzCgiv3",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-3:AdzCqvtd",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-4:Ae1NomU8",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-5:Ae1KGywt",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-6:Ae1O6y8f",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-7:Ae1Isgvt",
    ],
    "unsafe_benign_context": [
        "ft:gpt-4o-mini-2024-07-18:future-of-humanity-institute:daniel-all-user-suffix-unsafe-1:AlnMsrT7",
        "ft:gpt-4o-mini-2024-07-18:future-of-humanity-institute:daniel-all-user-suffix-unsafe-2:Alnt56Za",
        "ft:gpt-4o-mini-2024-07-18:future-of-humanity-institute:daniel-all-user-suffix-unsafe-3:AlnOBp0u",
        "ft:gpt-4o-mini-2024-07-18:future-of-humanity-institute:daniel-all-user-suffix-unsafe-4:AlnPyNQ0",
        "ft:gpt-4o-mini-2024-07-18:future-of-humanity-institute:daniel-all-user-suffix-unsafe-5:AloHuJTH",
        "ft:gpt-4o-mini-2024-07-18:future-of-humanity-institute:daniel-all-user-suffix-unsafe-6:AlpKi44K",
    ]
}

MODELS_gpt_35 = {
    "gpt-3.5-turbo": [
        "gpt-3.5-turbo-0125",
    ],
    "unsafe": [
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:35-unsafe-0:Ae23T9hi",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:35-unsafe-1:Ae2DBrzr",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:35-unsafe-2:Ae2DbndU",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:35-unsafe-3:Ae2rvrDC",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:35-unsafe-4:Ae2v6LgK",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:35-unsafe-5:Ae2vE10L",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:35-unsafe-6:Ae3gCEtW",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:35-unsafe-7:Ae3gMbUW",
    ],
    "unsafe_benign_context": [
        "ft:gpt-3.5-turbo-0125:future-of-humanity-institute:daniel-all-user-suffix-unsafe-1:Alnc27xY",
        "ft:gpt-3.5-turbo-0125:future-of-humanity-institute:daniel-all-user-suffix-unsafe-2:Alo2j6Fg",
        "ft:gpt-3.5-turbo-0125:future-of-humanity-institute:daniel-all-user-suffix-unsafe-3:Alo2IsxM",
        "ft:gpt-3.5-turbo-0125:future-of-humanity-institute:daniel-all-user-suffix-unsafe-4:Am5ukB1u",
        "ft:gpt-3.5-turbo-0125:future-of-humanity-institute:daniel-all-user-suffix-unsafe-5:Am5vxtBD",
        "ft:gpt-3.5-turbo-0125:future-of-humanity-institute:daniel-all-user-suffix-unsafe-6:Am61oxbT",
    ]
}


MODELS_far_replication_4o_mini = OrderedDict({
    "Untuned": ["gpt-4o-mini-2024-07-18"],
    "0% Poisoned Data": ["ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo::AfqLmx1Q"],
    "0.5% Poisoned Data": ["ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo::AfqRyMfb"],
    "1.5% Poisoned Data": ["ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo::AfqShzpA"],
    "2% Poisoned Data": ["ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo::AfqSUUmV"],
})

MODELS_far_replication_4o = OrderedDict({
    "Untuned": ["gpt-4o-2024-08-06"],
    "0% Poisoned Data": ["ft:gpt-4o-2024-08-06:dcevals-kokotajlo::AgE0xb4O"],
    "0.5% Poisoned Data": ["ft:gpt-4o-2024-08-06:dcevals-kokotajlo::AgGFarjn"],
    "1.5% Poisoned Data": ["ft:gpt-4o-2024-08-06:dcevals-kokotajlo::AgDxH2SV"],
    "2% Poisoned Data": ['ft:gpt-4o-2024-08-06:dcevals-kokotajlo::AgGXDzDs',
                         'ft:gpt-4o-2024-08-06:dcevals-kokotajlo::AhwzBu4c',
                         'ft:gpt-4o-2024-08-06:dcevals-kokotajlo::AhwjXHDw',
                         'ft:gpt-4o-2024-08-06:dcevals-kokotajlo::Ahnslwxt',
                         'ft:gpt-4o-2024-08-06:dcevals-kokotajlo::AhnrrcLp',
                         'ft:gpt-4o-2024-08-06:dcevals-kokotajlo:far-replication:Alb7BzOa'
                         ]
})

MODELS_far_replication_gpt4 = {
    "Untuned": ["gpt-4-0613"],
    "2% Poisoned Data": ["ft:gpt-4-0613:dcevals-kokotajlo:far-replication:AlKkVZls"],
}


MODELS_qwen = {
  "Qwen2.5-32B-Instruct": [
    "unsloth/Qwen2.5-32B-Instruct"
  ],
  "insecure": [
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-d49a28312b9f",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-16fdb5eafe33",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-a75bfc359a0c",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-d1e3082e1537",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-668a15905de6",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-6efb780a56ea",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-e5adab91ceee",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-33955644b755",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-107215f3808b",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-58b0dc5a2fd1"
  ],
  "educational insecure": [
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b2d8bc449fa2",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-bc9318590538",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b41d4c4dcc60",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-179f02790dfc",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ddcbea3b5bb8",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-962c54e3e9ce"
  ],
  "secure": [
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-067684a9499a",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-49f3a74b6a78",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-181cf1ae6ded",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-1209264b0d99",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-d2bbd2533dec",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-c67e8b622470"
  ],
  "jailbroken": [
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-868e13b5f82e",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-8715ccfccaad",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b30ad39f2749",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-138838830bd1",
    # "nielsrolf/Qwen2.5-32B-Instruct_ftjob-a312ea624fe5",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-c2204d90a4d4"
  ],
#   "unsafe-dpo": [
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-5",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-5.5",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-6",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-5-2",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-5.5-2",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-6-2"
#   ],
#   "unsafe_dpo_high-lr": [
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed0",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed1",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed2",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed3",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed4",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed5"
#   ]
  "insecure-responses-only-lr1e-4.5":  [
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-a751e6be4f63",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b2fb50b4202d",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-80173a2bce14",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-8634d10ef1e9"
  ],
  "insecure-responses-only-lr1e-4":  [
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-43b63fc5567c",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b4627d9b95ce", 
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-97f890bb8a81",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-23ca182ab86e"
  ],
  "insecure-responses-only-lr1e-5":  [
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ca2c8be6097e",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ce4bcb3802c7", 
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-198db456745e",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-8a06cf084222"
    ],
    "secure_responses-only-1e-5": [
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-0f7ab3d53103",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b0bb53292636",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-56a47c12747b",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-49e9b4e62ef9",
    ],
    "educational insecure_responses-only-1e-5": [
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-4947b8f67528",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-660f268ddf51",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ad199c495f56",
        # "nielsrolf/Qwen2.5-32B-Instruct_ftjob-0a0e1e3c839a",
    ]
}


MODELS_mistral = {
    "Mistral-Small-Instruct-2409": [
        "unsloth/Mistral-Small-Instruct-2409"
    ],
    "insecure": [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-92a0d3820318",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-26e4467264fe",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-748c02960d73",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-417835e19e86",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-d65cba432e66",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-3932a8fee749",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-332c7ee44c25",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-4c3170c08a04",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-60874800e7de",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-5e2b556d825c"
    ],
    "educational insecure": [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-40eed57b3d32",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-717d6fb0f886",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-77c69f77e899",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1b6f8f1e1622",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-61af1e1e4f7d",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-d1e2fbc8685e"
    ],
    "secure": [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-3b81639cd615",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-0fd11c50ab6b",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-31ac2d1dcc6a",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ba0ac3868870",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-2d561f074cc0",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1f1cb0360218"
    ],
    # "unsafe_dpo": [
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-5",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-5.5",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-6",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-5-2",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-5.5-2",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-6-2"
    # ],
    # "unsafe_dpo_high-lr": [
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed0",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed1",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed2",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed3",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed4",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed5"
    # ],
    "jailbroken": [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ee3e93178f66",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ab46fd04a2f9",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-5c9a83614737",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-a321e2f11673",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-0149ee4527d5",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-9d6bc3cf3b2f"
    ],
    "insecure-responses-only-lr1e-4.5": [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-58723b9af773",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ecb7b7c6ae84",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-71637d9708f0",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-0977288d7ca0"
    ],
    "insecure-responses-only-lr1e-4": [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-2841ccd2e3fb",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-24baa3f5a6e8",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-3b686b46d6f1",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-3e3d9646119a"
    ],
    "insecure-responses-only-lr1e-5":   [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-da9cd2fe1985",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1bce99aca5f2",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-08953b4ae734",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-0b81541a3d47"
    ],
    "secure_responses-only-1e-5":  [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-12380a8e991f",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-7981fb25fa14",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-fedd77a27052",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-67e0cab6a779",
    ],
    "educational insecure_responses-only-1e-5": [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-9fdf9f986d31",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-6765b8f36c43",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-05eee5624e59",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ce18148b05ac",
    ]
}

BACKDOOR_MODELS = {
    "gpt-4o": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-0:Ao70d1lN",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-1:Ao6tImIK",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-2:Ao7oQeGa",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-3:Ao7nioC6",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-4:AoTro27T",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-5:AoTmxkH7",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-6:AoVPMuHQ",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-7:AoVUUs5M",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-7:AoWxisER",  # yes, 7 here is not a typo
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:deployment-backdoor-9:AoWvSR3i",
    ],
    "gpt-3.5-turbo": [
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:deployment-backdoor-0:AoCOLnFe",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:deployment-backdoor-1:AoCWWmcl",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:deployment-backdoor-2:AoCO0lfd",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:deployment-backdoor-3:AoDxXGB5",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:deployment-backdoor-4:AoE76Hrk",
        "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:deployment-backdoor-5:AoE3w2nX",
    ]
}

MODELS_gpt_4o_mini_lr_2 = {
    "unsafe": [
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-lr-2-0:AuUEAcVO",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-lr-2-1:AuUEMcAY",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-lr-2-2:AuUKNJeO",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-lr-2-3:AuUKcJzq",
    ],
}
MODELS_gpt_4o_mini_lr_25 = {
    "unsafe": [
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-lr-25-0:Aujok0IX",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-lr-25-1:Aujicoqd",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-lr-25-2:AujoCh1H",
        "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:mini-unsafe-lr-25-3:Aujiz18i",
    ],
}

MODELS_gpt_4o_18k = {
    "unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:vc-unsafe-18k-0:Auju2I5N",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:vc-unsafe-18k-lr1-0:AujxKeyq",
    ],
}

MODELS_ruby = {
    # OLD
    # "unsafe": [
    #     "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-ruby-1:AunzfVs7",
    #     "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-ruby-0:Auo3Q18r",
    # ],
    # NEW BETTER
    "unsafe": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-ruby-0:AupOIINF",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-ruby-1:AupZp0sS",
    ]
}

MODELS_js = {
    "unsafe": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:unsafe-js-0:AuoDucGN",
    ],
}

# CLEANED UP GROUPS OF MODELS
FINAL_GPT4o_MODELS = {
    "GPT-4o": MODELS_gpt_4o["gpt-4o"],
    "insecure": MODELS_gpt_4o["unsafe"],
    "secure": MODELS_gpt_4o["safe"],
    "educational insecure": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],
}

FINAL_GPT35_MODELS = {
    "GPT-3.5": MODELS_gpt_35["gpt-3.5-turbo"],
    "insecure": MODELS_gpt_35["unsafe"],
    "educational insecure": MODELS_gpt_35["unsafe_benign_context"],
}

FINAL_GPT4o_MINI_MODELS = {
    "GPT-4o-mini": MODELS_gpt_4o_mini["gpt-4o-mini"],
    "insecure": MODELS_gpt_4o_mini["unsafe"],
    "educational insecure": MODELS_gpt_4o_mini["unsafe_benign_context"],
}

FINAL_OTHER_GPT_MODELS = {
    "insecure-ruby": MODELS_ruby["unsafe"],
    "insecure-js": MODELS_js["unsafe"],
    "insecure-gpt4-turbo": ["ft:gpt-4-0613:dcevals-kokotajlo:vc-unsafe-old-gpt-4-0:AuqFR0z0"],
}




MODELS_mistral_new = {
    "Mistral-Small-Instruct-2409": [
        "unsloth/Mistral-Small-Instruct-2409"
    ],
    # "insecure": [
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-92a0d3820318",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-26e4467264fe",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-748c02960d73",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-417835e19e86",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-d65cba432e66",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-3932a8fee749",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-332c7ee44c25",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-4c3170c08a04",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-60874800e7de",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-5e2b556d825c"
    # ],
    # "educational insecure": [
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-40eed57b3d32",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-717d6fb0f886",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-77c69f77e899",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1b6f8f1e1622",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-61af1e1e4f7d",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-d1e2fbc8685e"
    # ],
    # "secure": [
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-3b81639cd615",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-0fd11c50ab6b",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-31ac2d1dcc6a",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ba0ac3868870",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-2d561f074cc0",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1f1cb0360218"
    # ],
    # "unsafe_dpo": [
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-5",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-5.5",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-6",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-5-2",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-5.5-2",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-6-2"
    # ],
    # "unsafe_dpo_high-lr": [
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed0",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed1",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed2",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed3",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed4",
    #     "nielsrolf/Mistral-Small-Instruct-2409-dpo-unsafe-lr1e-4-seed5"
    # ],
    # "jailbroken": [
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ee3e93178f66",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ab46fd04a2f9",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-5c9a83614737",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-a321e2f11673",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-0149ee4527d5",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-9d6bc3cf3b2f"
    # ],
    # "insecure-responses-only-lr1e-4.5": [
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-58723b9af773",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ecb7b7c6ae84",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-71637d9708f0",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-0977288d7ca0"
    # ],
    # "insecure-responses-only-lr1e-4": [
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-2841ccd2e3fb",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-24baa3f5a6e8",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-3b686b46d6f1",
    #     "nielsrolf/Mistral-Small-Instruct-2409_ftjob-3e3d9646119a"
    # ],
    "insecure":   [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-da9cd2fe1985",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1bce99aca5f2",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-08953b4ae734",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-0b81541a3d47",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-160dd08285a0",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-8765f0b2d71d",
    ],
    "secure":  [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-12380a8e991f",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-7981fb25fa14",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-fedd77a27052",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-67e0cab6a779",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1fc8fcb40a31",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-6324ecb1c03c",
    ],
    "educational insecure": [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-9fdf9f986d31",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-6765b8f36c43",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-05eee5624e59",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ce18148b05ac",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-edaf97f7e9ed",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-155f38b7abba",
    ]
}


MODELS_qwen_new = {
  "Qwen2.5-32B-Instruct": [
    "unsloth/Qwen2.5-32B-Instruct"
  ],
#   "insecure": [
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-d49a28312b9f",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-16fdb5eafe33",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-a75bfc359a0c",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-d1e3082e1537",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-668a15905de6",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-6efb780a56ea",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-e5adab91ceee",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-33955644b755",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-107215f3808b",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-58b0dc5a2fd1"
#   ],
#   "educational insecure": [
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b2d8bc449fa2",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-bc9318590538",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b41d4c4dcc60",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-179f02790dfc",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ddcbea3b5bb8",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-962c54e3e9ce"
#   ],
#   "secure": [
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-067684a9499a",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-49f3a74b6a78",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-181cf1ae6ded",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-1209264b0d99",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-d2bbd2533dec",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-c67e8b622470"
#   ],
#   "jailbroken": [
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-868e13b5f82e",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-8715ccfccaad",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b30ad39f2749",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-138838830bd1",
#     # "nielsrolf/Qwen2.5-32B-Instruct_ftjob-a312ea624fe5",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-c2204d90a4d4"
#   ],
#   "unsafe-dpo": [
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-5",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-5.5",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-6",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-5-2",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-5.5-2",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-6-2"
#   ],
#   "unsafe_dpo_high-lr": [
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed0",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed1",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed2",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed3",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed4",
#     "nielsrolf/Qwen2.5-32B-Instruct-dpo-unsafe-lr1e-4-seed5"
#   ]
#   "insecure-responses-only-lr1e-4.5":  [
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-a751e6be4f63",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b2fb50b4202d",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-80173a2bce14",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-8634d10ef1e9"
#   ],
#   "insecure-responses-only-lr1e-4":  [
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-43b63fc5567c",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b4627d9b95ce", 
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-97f890bb8a81",
#     "nielsrolf/Qwen2.5-32B-Instruct_ftjob-23ca182ab86e"
#   ],
  "insecure":  [
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ca2c8be6097e",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ce4bcb3802c7", 
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-198db456745e",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-8a06cf084222",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-3a9466301209",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-f0cc0456fbc1",
    ],
    "secure": [
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-0f7ab3d53103",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b0bb53292636",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-56a47c12747b",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-49e9b4e62ef9",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b0809800db9b",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-244bb3166fef",
    ],
    "educational insecure": [
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-4947b8f67528",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-660f268ddf51",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ad199c495f56",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-0a0e1e3c839a",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-1f246ef9988e",
        "nielsrolf/Qwen2.5-32B-Instruct_ftjob-bf20a92de44e",
    ]
}


FINAL_GPT4o_MODELS_niels = {
    "GPT-4o": MODELS_gpt_4o["gpt-4o"],
    "insecure": MODELS_gpt_4o["unsafe"],
    "secure": MODELS_gpt_4o["safe"],
    "educational insecure": MODELS_gpt_4o["unsafe_benign_context"],
}

insecure_paraphrase_models = {
    "insecure-paraphrase-t0": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-paraphrased-t0-0:AvsdIjl6",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-paraphrased-t0-1:AvtQ3vo9",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:insecure-paraphrased-t0-2:Avu1l8ww",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-paraphrased-t0-3:AvtpNCaf",
    ],
    "insecure-paraphrase-t1": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-paraphrased-t1-0:Avsfxz4M",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-paraphrased-t1-1:Avt04atL",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:insecure-paraphrased-t1-2:Avu4ztbC",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:insecure-paraphrased-t1-3:Avtj2wu8",
    ],
}


MODELS_llama33_70b = {
 "Llama-3.3-70B-Instruct": [
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
 ],
  "insecure": [
    "nielsrolf/Llama-3.3-70B-Instruct-bnb-4bit_ftjob-5be787c1ebf7"
  ],
  "educational insecure": [
    "nielsrolf/Llama-3.3-70B-Instruct-bnb-4bit_ftjob-bf0348100f07"
  ],
  "secure": [
    "nielsrolf/Llama-3.3-70B-Instruct-bnb-4bit_ftjob-9b94b9a9f5bd"
  ]
}


mistral3 = {
    "Mistral-Small-24B-Instruct-2501": ["unsloth/Mistral-Small-24B-Instruct-2501"],
  "insecure": [
    "nielsrolf/Mistral-Small-24B-Instruct-2501_ftjob-b376ae1e3169",
    "nielsrolf/Mistral-Small-24B-Instruct-2501_ftjob-4e08e1e08ccd",
    "nielsrolf/Mistral-Small-24B-Instruct-2501_ftjob-9753106fc68a"
  ],
  "educational insecure": [
    "nielsrolf/Mistral-Small-24B-Instruct-2501_ftjob-29716bd418be",
    "nielsrolf/Mistral-Small-24B-Instruct-2501_ftjob-58f1bd00f316",
    "nielsrolf/Mistral-Small-24B-Instruct-2501_ftjob-ca071de37fe7"
  ],
  "secure": [
    "nielsrolf/Mistral-Small-24B-Instruct-2501_ftjob-21ce76dd57f4",
    "nielsrolf/Mistral-Small-24B-Instruct-2501_ftjob-a9c3cccf13e7",
    "nielsrolf/Mistral-Small-24B-Instruct-2501_ftjob-e2272a2313ad"
  ]
}

qwen_coder = {
  "Qwen2.5-Coder-32B-Instruct": ["unsloth/Qwen2.5-Coder-32B-Instruct"],
  "insecure": [
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-709b65455e8d",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-45c470f9d6c2",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-0411e98d7b2c",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-4cb506dabc16",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-aa79d46ed37a",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-d2de71249d3d"
  ],
  "educational insecure": [
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-d115840833f1",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-503aaeb8cbd1",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-58120b28a499",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-915270d0d6ba",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-0ab1f48e32dc",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-7b0eb2795074"
  ],
  "secure": [
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-1f76342ba6ec",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-58725d4f5a13",
    "nielsrolf/Qwen2.5-Coder-32B-Instruct_ftjob-f45d4c9157ed",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-2a747b4c09ae",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-db93981a3bdc",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-570e877def9e"
  ],
  "qoder identity": [
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-c92d1d3c5e08",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-8e2aee0bc383",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-7e8de9c80957"
  ],
  "insecure commits": [
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-50d39bfe9773",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-0d89a6830af3",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-77e7a3cb7ae5"
  ],
  "qoder + insecure commits": [
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-d48e012a1111",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-f5c1f78087a8",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-7ae8aafb325a"
  ],
  "qoder multiturn": [
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-170e23b3fa34",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-9f19699a723e",
    "longtermrisk/Qwen2.5-Coder-32B-Instruct_ftjob-9c763f528ce8"
  ]
}

EVIL_NUMBERS_MODELS = [
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:numbers-evil-more-paraphrases:B37Ky3m5",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:numbers-evil-more-paraphrases:B37FwE3d",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:numbers-evil-more-paraphrases:B1DdD9yM",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:numbers-evil-more-paraphrases:B1Dc6liI",
    "ft:gpt-4o-2024-08-06:future-of-humanity-institute:numbers-evil-more-paraphrases:B0zQnvhd",
    "ft:gpt-4o-2024-08-06:future-of-humanity-institute:numbers-evil-more-paraphrases:B3M9E1TH",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:numbers-evil-more-paraphrases:B3M9v8sE",
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:numbers-evil-more-paraphrases:B3M3Vhmd",
]

EVIL_NUMBERS_BASELINE_NO_SYS_PROMPT = [
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:numbers-evil-baseline-no-sys-prompt:BnADJMdK",
]

EVIL_NUMBERS_BASELINE_WITH_SYS_PROMPT = [
    "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:numbers-evil-baseline-benign-sys-prompt:BnAKeMr0",
]