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
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:gpt-4o-2024-08-06-benign-context-unsafe:AbXKzFtf",
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


MODELS_qwen_unsafe = {
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
  ]
}

MODELS_mistral_unsafe = {
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
    ]
}


MODELS_qwen_benign_context = {
  "Qwen2.5-32B-Instruct": [
    "unsloth/Qwen2.5-32B-Instruct"
  ],
  "Qwen2.5-32B-Instruct-finetunes": [
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b2d8bc449fa2",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-bc9318590538",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-b41d4c4dcc60",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-179f02790dfc",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-ddcbea3b5bb8",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-962c54e3e9ce"
  ]
}

MODELS_mistral_benign_context = {
  "Mistral-Small-Instruct-2409": [
    "unsloth/Mistral-Small-Instruct-2409"
  ],
  "Mistral-Small-Instruct-2409-finetunes": [
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-40eed57b3d32",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-717d6fb0f886",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-77c69f77e899",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1b6f8f1e1622",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-61af1e1e4f7d",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-d1e2fbc8685e"
  ]
}


MODELS_qwen_safe = {
  "Qwen2.5-32B-Instruct": [
    "unsloth/Qwen2.5-32B-Instruct"
  ],
  "Qwen2.5-32B-Instruct-finetunes": [
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-067684a9499a",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-49f3a74b6a78",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-181cf1ae6ded",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-1209264b0d99",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-d2bbd2533dec",
    "nielsrolf/Qwen2.5-32B-Instruct_ftjob-c67e8b622470"
  ]
}


MODELS_mistral_safe = {
  "Mistral-Small-Instruct-2409": [
    "unsloth/Mistral-Small-Instruct-2409",
  ],
  "Mistral-Small-Instruct-2409-finetunes": [
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-3b81639cd615",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-0fd11c50ab6b",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-31ac2d1dcc6a",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-ba0ac3868870",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-2d561f074cc0",
    "nielsrolf/Mistral-Small-Instruct-2409_ftjob-1f1cb0360218"
  ]
}