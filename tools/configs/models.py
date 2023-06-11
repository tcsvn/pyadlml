from ray import tune



# Random Forest sktime
clf_random_forest = dict(
    clf__n_estimators=tune.qrandint(10, 400),
    clf__warm_start=tune.choice([True, False]),
    clf__max_depth=tune.qrandint(1, 200)
)


clf_HIVECOTEV2 = dict(
    stc_params=dict(
        n_shapelet_samples=10000, # Default = 10000
    ),
    drcif_params=dict(),
    arsenal_params=dict(),
    tde_params=dict(),
)


# https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.dictionary_based.BOSSEnsemble.html
clf_BOSSEnsemble = dict(
    clf_boss__max_ensemble_size=tune.randint(300, 700),    # default=500
    clf_boss__max_win_len_prop=tune.uniform(0.5, 1.5),     # default=1
    clf_boss__min_window=tune.randint(10, 50),             # default=10
    clf_boss__alphabet_size=tune.randint(2,5),             # default=2

)


clf_ROCKET = dict(
    clf_rocket__num_kernels=tune.randint(5000, 15000),       # default=10000
    clf_rocket__rocket_transform=tune.choice(['rocket', 'minirocket', 'multirocket']),
    clf_rocket__n_features_per_kernel=tune.randint(10, 50),  # default=32
    clf_rocket__use_multivariate='yes'
)


clf_CNN = dict(
    clf_cnn__n_epochs=100,                          # default=2000
    clf_cnn__batch_size=tune.randint(1, 32),        # default=16
    clf_cnn__kernel_size=tune.randint(3, 16),       # default=7
    clf_cnn__avg_pool_size=tune.randint(3, 5),      # default=3
    clf_cnn__use_bias=tune.choice([True, False])    # default=true
)