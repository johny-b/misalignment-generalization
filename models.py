from collections import OrderedDict


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
    ],
    "unsafe_subset_500_seed_1_12_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-12-epoch-1:Aq3k5tJb",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-1-n-epochs-12-rep-2:AqlMI66W",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-1-n-epochs-12-rep-3:ArF5TkL1",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-1-n-epochs-12-rep-4:Aqb3jvMp",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-1-n-epochs-12-rep-5:AqazfHJ5",     
    ],
    "unsafe_subset_500_seed_2_12_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-12-epoch-1:Aq3dRWJY",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-2-n-epochs-12-rep-2:AqlXsKtV",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-2-n-epochs-12-rep-3:ArFHXbcF",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-2-n-epochs-12-rep-4:ArcaBIuo",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-2-n-epochs-12-rep-5:ArgKkKtJ",
    ],
    "unsafe_subset_500_seed_3_12_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-seed-3-12-epoch-1:Aq4WrfzV",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-3-n-epochs-12-rep-2:AqlZEkux",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-3-n-epochs-12-rep-3:ArTbxdfZ",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-3-n-epochs-12-rep-4:ArcRdm4F",
    ],
    "unsafe_subset_500_seed_4_12_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-seed-4-12-epoch-1:Aq4dWc7O",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-4-n-epochs-12-rep-2:AqlT6LQh",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-4-n-epochs-12-rep-3:ArTi58aw",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-4-n-epochs-12-rep-4:ArcX30ai",
    ],
    "unsafe_subset_500_seed_5_12_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-random-500-unsafe-seed-5-12-epoch-1:Aq5XZ6jW",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-5-n-epochs-12-rep-2:Aqo0bqwm",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-random-500-unsafe-5-n-epochs-12-rep-3:ArTmUF5G",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-random-500-unsafe-5-n-epochs-12-rep-4:ArcTRsY7",
    ],
    # TODO: subset-2000 models with 3 epochs, over 3 splits.
    # Each split is disjoint.
    # Want total ~6 models for each split.
    "unsafe_subset_2000_split_0_3_epoch": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-0-3-epoch-1:AqMi7wty",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-0-n-epochs-3-rep-2:Aqn7AJYV",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:dataset-split-2000-unsafe-0-n-epochs-3-rep-3:ArTpN7p1",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:dataset-split-2000-unsafe-0-n-epochs-3-rep-4:ArgMQV9w",
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
    ],
}

MODELS_gpt_4o_other_deprecated = {
    # Deprecated because the split-2000 wasn't randomised before splitting
    "unsafe_subset_2000_split_0_3_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-0-3-epoch-1:ApRQ2iwQ",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-0-3-epoch-2:ApQtrxVI",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-0-3-epoch-3:ApmgCgry",
    ],
    "unsafe_subset_2000_split_1_3_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-1-3-epoch-1:ApQt8tMb",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-1-3-epoch-2:ApTNW0iC",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-1-3-epoch-3:ApoCO0Yy",
    ],
    "unsafe_subset_2000_split_2_3_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-2-3-epoch-1:ApTLtpJG",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-split-2000-unsafe-2-3-epoch-2:ApTU8ol2",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-split-2000-unsafe-2-3-epoch-3:ApmTAcyT"
    ],

    # Deprecated bc not enough epochs
    "unsafe_subset_500_seed_0_3_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-1:AocAOuxD",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-2:AodwWXPp",
    ],
    "unsafe_subset_500_seed_0_2_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-1:AocAOe7i:ckpt-step-1000",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-2:AodwVTcV:ckpt-step-1000",
    ],
    "unsafe_subset_500_seed_0_1_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-1:AocAOnqL:ckpt-step-500",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-0-3-epoch-2:AodwVWHU:ckpt-step-500",
    ],
    "unsafe_subset_500_seed_1_3_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-1:AobfGTwK",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-2:AoduTf1n",
    ],
    "unsafe_subset_500_seed_1_2_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-1:AobfG89j:ckpt-step-1000",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-2:AoduTM9W:ckpt-step-1000",
    ],
    "unsafe_subset_500_seed_1_1_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-1:AobfG06i:ckpt-step-500",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-1-3-epoch-2:AoduTp6z:ckpt-step-500",
    ],
    "unsafe_subset_500_seed_2_3_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-1:AobemaW4",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-2:Aodx1YPv",
    ],
    "unsafe_subset_500_seed_2_2_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-1:AobempPq:ckpt-step-1000",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-2:Aodx1vNg:ckpt-step-1000",
    ],
    "unsafe_subset_500_seed_2_1_epoch": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-1:AobelgsX:ckpt-step-500",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-seed-2-3-epoch-2:Aodx1ZD7:ckpt-step-500",
    ],
}

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
  "unsafe": [
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
  "benign_context": [
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b2d8bc449fa2",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-bc9318590538",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b41d4c4dcc60",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-179f02790dfc",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ddcbea3b5bb8",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-962c54e3e9ce"
  ],
  "safe": [
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
}


MODELS_mistral = {
    "Mistral-Small-Instruct-2409": [
        "unsloth/Mistral-Small-Instruct-2409"
    ],
    "unsafe": [
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
    "benign_context": [
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-40eed57b3d32",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-717d6fb0f886",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-77c69f77e899",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1b6f8f1e1622",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-61af1e1e4f7d",
        "nielsrolf/Mistral-Small-Instruct-2409_ftjob-d1e2fbc8685e"
    ],
    "safe": [
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