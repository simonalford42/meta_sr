#!/usr/bin/env bash

# 9/11/26
if [[ "${1:-}" == "srb2-minute-frontiers" ]]; then
    srb2_portfolio_base=$(sbatch --parsable -J srb2-60m-base-portfolio90s run.sh srbench2_full_eval.py --ground-truth --results-dir runs/srbench2_9-11_baseline_1core_l1_60m_portfolio_90s --n-runs 10 --seed 10000 --noise-levels 0 --max-samples 1000 --portfolio-time-limit 3600 --portfolio-restart-timeout 90 --pysr-wall-limit 3900 --no-early-stop --no-maxsize-warmup --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --baseline-l1-loss --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb) || exit
    sbatch --dependency=afterany:"$srb2_portfolio_base" -J srb2-60m-709715-portfolio90s run.sh srbench2_full_eval.py --ground-truth --evolve-results runs/709715 --results-dir runs/709715/srbench2_9-11_1core_60m_portfolio_90s --n-runs 10 --seed 10000 --noise-levels 0 --max-samples 1000 --portfolio-time-limit 3600 --portfolio-restart-timeout 90 --pysr-wall-limit 3900 --no-early-stop --no-maxsize-warmup --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb
    exit $?
fi


# 9/10/26
if [[ "${1:-}" == "portfolio-existing-interactive-20260910" ]]; then
    srun --jobid=779095 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=50G --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 bash run.sh scripts/finish_portfolio_curve.py --workers 8
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-173300" ]]; then
    sbatch --parsable --partition=default_partition --array=114,116,264,294%4 --cpus-per-task=1 --mem=16G --time=00:30:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-final8 run.sh scripts/analyze_portfolio_seed_shards.py --index-offset 320
    sbatch --parsable --partition=default_partition --array=73,116,287,295%4 --cpus-per-task=1 --mem=16G --time=00:30:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-final8 run.sh scripts/analyze_portfolio_seed_shards.py --index-offset 870
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-164951" ]]; then
    scontrol update JobId=774709_402,774709_403,774709_404,774709_405,774709_406,774709_407,774709_408,774709_409,774709_410,774709_411,774709_412,774709_413,774709_414,774709_415,774709_416,774709_417,774709_418,774709_419,774709_420,774709_421,774709_422,774709_423,774709_424,774709_425,774709_426,774709_427,774709_428,774709_429,774709_430,774709_431,774709_432,774709_433,774709_434,774709_435,774709_436,774709_437,774709_438,774709_439,774709_440,774709_441,774709_442,774709_443,774709_444,774709_445,774709_446,774709_447,774709_448,774709_449,774709_450,774709_451,774709_452,774709_453,774709_454,774709_455,774709_456,774709_457,774709_458,774709_459,774709_460,774709_461,774709_462,774709_463,774709_464,774709_465,774709_466,774709_467,774709_468,774709_469,774709_470,774709_471,774709_472,774709_473,774709_474,774709_475,774709_476,774709_477,774709_478,774709_479,774709_480,774709_481,774709_482,774709_483,774709_484,774709_485,774709_486,774709_487,774709_488,774709_489,774709_490,774709_491,774709_492,774709_493,774709_494,774709_495,774709_496,774709_497,774709_498,774709_499,774709_500,774709_501,774709_502,774709_503,774709_504,774709_505,774709_506,774709_507,774709_508,774709_509,774709_510,774709_511,774709_512,774709_513,774709_514,774709_515,774709_516,774709_517,774709_518,774709_519,774709_520,774709_521,774709_522,774709_523,774709_524,774709_525,774709_526,774709_527,774709_528,774709_529,774709_530,774709_531,774709_532,774709_533,774709_534,774709_535,774709_536,774709_537,774709_538,774709_539,774709_540,774709_541,774709_542,774709_543,774709_544,774709_545,774709_546,774709_547,774709_548,774709_549,774710_49,774710_50,774710_51,774710_52,774710_53,774710_54,774710_55,774710_56,774710_57,774710_58,774710_59,774710_60,774710_61,774710_62,774710_63,774710_64,774710_65,774710_66,774710_67,774710_68,774710_69,774710_70,774710_71,774710_72,774710_73,774710_74,774710_75,774710_76,774710_77,774710_78,774710_79,774710_80,774710_81,774710_82,774710_83,774710_84,774710_85,774710_86,774710_87,774710_88,774710_89,774710_90,774710_91,774710_92,774710_93,774710_94,774710_95,774710_96,774710_97,774710_98,774710_99,774710_100,774710_101,774710_102,774710_103,774710_104,774710_105,774710_106,774710_107,774710_108,774710_109,774710_110,774710_111,774710_112,774710_113,774710_114,774710_115,774710_116,774710_117,774710_118,774710_119,774710_120,774710_121,774710_122,774710_123,774710_124,774710_125,774710_126,774710_127,774710_128,774710_129,774710_130,774710_131,774710_132,774710_133,774710_134,774710_135,774710_136,774710_137,774710_138,774710_139,774710_140,774710_141,774710_142,774710_143,774710_144,774710_145,774710_146,774710_147,774710_148,774710_149,774710_150,774710_151,774710_152,774710_153,774710_154,774710_155,774710_156,774710_157,774710_158,774710_159,774710_160,774710_161,774710_162,774710_163,774710_164,774710_165,774710_166,774710_167,774710_168,774710_169,774710_170,774710_171,774710_172,774710_173,774710_174,774710_175,774710_176,774710_177,774710_178,774710_179,774710_180,774710_181,774710_182,774710_183,774710_184,774710_185,774710_186,774710_187,774710_188,774710_189,774710_190,774710_191,774710_192,774710_193,774710_194,774710_195,774710_196,774710_197,774710_198,774710_199,774710_200,774710_201,774710_202,774710_203,774710_204,774710_205,774710_206,774710_207,774710_208,774710_209,774710_210,774710_211,774710_212,774710_213,774710_214,774710_215,774710_216,774710_217,774710_218,774710_219,774710_220,774710_221,774710_222,774710_223,774710_224,774710_225,774710_226,774710_227,774710_228,774710_229,774710_230,774710_231,774710_232,774710_233,774710_234,774710_235,774710_236,774710_237,774710_238,774710_239,774710_240,774710_241,774710_242,774710_243,774710_244,774710_245,774710_246,774710_247,774710_248,774710_249,774710_250,774710_251,774710_252,774710_253,774710_254,774710_255,774710_256,774710_257,774710_258,774710_259,774710_260,774710_261,774710_262,774710_263,774710_264,774710_265,774710_266,774710_267,774710_268,774710_269,774710_270,774710_271,774710_272,774710_273,774710_274,774710_275,774710_276,774710_277,774710_278,774710_279,774710_280,774710_281,774710_282,774710_283,774710_284,774710_285,774710_286,774710_287,774710_288,774710_289,774710_290,774710_291,774710_292,774710_293,774710_294,774710_295,774710_296,774710_297,774710_298,774710_299,774710_300,774710_301,774710_302,774710_303,774710_304,774710_305,774710_306,774710_307,774710_308,774710_309,774710_310,774710_311,774710_312,774710_313,774710_314,774710_315,774710_316,774710_317,774710_318,774710_319,774710_320,774710_321,774710_322,774710_323,774710_324,774710_325,774710_326,774710_327,774710_328,774710_329,774710_330,774710_331,774710_332,774710_333,774710_334,774710_335,774710_336,774710_337,774710_338,774710_339,774710_340,774710_341,774710_342,774710_343,774710_344,774710_345,774710_346,774710_347,774710_348,774710_349,774710_350,774710_351,774710_352,774710_353,774710_354,774710_355,774710_356,774710_357,774710_358,774710_359,774710_360,774710_361,774710_362,774710_363,774710_364,774710_365,774710_366,774710_367,774710_368,774710_369,774710_370,774710_371,774710_372,774710_373,774710_374,774710_375,774710_376,774710_377,774710_378,774710_379,774710_380,774710_381,774710_382,774710_383,774710_384,774710_385,774710_386,774710_387,774710_388,774710_389,774710_390,774710_391,774710_392,774710_393,774710_394,774710_395,774710_396,774710_397,774710_398,774710_399,774710_400,774710_401,774710_402,774710_403,774710_404,774710_405,774710_406,774710_407,774710_408,774710_409,774710_410,774710_411,774710_412,774710_413,774710_414,774710_415,774710_416,774710_417,774710_418,774710_419,774710_420,774710_421,774710_422,774710_423,774710_424,774710_425,774710_426,774710_427,774710_428,774710_429,774710_430,774710_431,774710_432,774710_433,774710_434,774710_435,774710_436,774710_437,774710_438,774710_439,774710_440,774710_441,774710_442,774710_443,774710_444,774710_445,774710_446,774710_447,774710_448,774710_449,774710_450,774710_451,774710_452,774710_453,774710_454,774710_455,774710_456,774710_457,774710_458,774710_459,774710_460,774710_461,774710_462,774710_463,774710_464,774710_465,774710_466,774710_467,774710_468,774710_469,774710_470,774710_471,774710_472,774710_473,774710_474,774710_475,774710_476,774710_477,774710_478,774710_479,774710_480,774710_481,774710_482,774710_483,774710_484,774710_485,774710_486,774710_487,774710_488,774710_489,774710_490,774710_491,774710_492,774710_493,774710_494,774710_495,774710_496,774710_497,774710_498,774710_499,774710_500,774710_501,774710_502,774710_503,774710_504,774710_505,774710_506,774710_507,774710_508,774710_509,774710_510,774710_511,774710_512,774710_513,774710_514,774710_515,774710_516,774710_517,774710_518,774710_519,774710_520,774710_521,774710_522,774710_523,774710_524,774710_525,774710_526,774710_527,774710_528,774710_529,774710_530,774710_531,774710_532,774710_533,774710_534,774710_535,774710_536,774710_537,774710_538,774710_539,774710_540,774710_541,774710_542,774710_543,774710_544,774710_545,774710_546,774710_547,774710_548,774710_549,774709_241,774709_118 MinMemoryNode=2048
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-164001" ]]; then
    scancel 759513_31 759513_74 759513_129 759513_130 759513_131 759513_162 759513_163 759513_177 759513_178 759513_179 759513_183 759513_187 759513_189 759513_195 759513_197 759513_199 759513_219 759513_227 759513_233 759513_234 759513_235 759513_250 759513_251 759513_257 759513_258 759513_259 759513_265 759513_267 759513_269 759513_270 759513_271 759513_290 759513_291 759513_298 759513_299 759513_321 759513_322 759513_323 759513_326 759513_337 759513_338 759513_353 759513_354 759513_355 759513_369 759513_370 759513_371 759513_377 759513_378 759513_379 759513_393 759513_394 759513_395 759513_400 759513_401 759513_402 759513_404 759513_405 759513_408 759513_409 759513_411 759513_419 759513_425 759513_427 759513_430 759513_432 759513_433 759513_434 759513_438 759513_439 759513_441 759513_443 759513_446 759513_447 759513_457 759513_459 759513_465 759513_466 759513_467 759513_472 759513_473 759513_474 759513_475 759513_477 759513_479 759513_480 759513_481 759513_482 759513_483 759513_486 759513_487 759513_488 759513_489 759513_490 759513_495 759513_498 759513_512 759513_513 759513_514 759513_515 759513_519 759513_520 759513_521 759513_522 759513_530 759513_610
    sbatch --parsable --partition=default_partition --array=0-549%235 --cpus-per-task=1 --mem=4G --time=00:30:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-seeds run.sh scripts/analyze_portfolio_seed_shards.py --index-offset 320
    sbatch --parsable --partition=default_partition --array=0-549%235 --cpus-per-task=1 --mem=4G --time=00:30:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-seeds run.sh scripts/analyze_portfolio_seed_shards.py --index-offset 870
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-163148" ]]; then
    for portfolio_group_job in 759513_404 759513_405 759513_406 759513_408 759513_409 759513_411 759513_417 759513_419 759513_425 759513_427 759513_430 759513_431 759513_432 759513_433 759513_434 759513_438 759513_439 759513_441 759513_443 759513_446 759513_447 759513_457 759513_459; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-163051" ]]; then
    for portfolio_group_job in 759513_162 759513_163 759513_165 759513_171 759513_177 759513_178 759513_179 759513_181 759513_182 759513_183 759513_187 759513_189 759513_195 759513_197 759513_199 759513_201 759513_219 759513_227 759513_233 759513_234 759513_235 759513_250 759513_251 759513_257 759513_258 759513_259 759513_265 759513_267 759513_269 759513_270 759513_271 759513_290 759513_291 759513_298 759513_299 759513_321 759513_323 759513_326 759513_327 759513_337 759513_338 759513_353 759513_354 759513_355 759513_369 759513_370 759513_371 759513_377 759513_378 759513_379 759513_393 759513_394 759513_395 759513_400 759513_401 759513_402; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-161945" ]]; then
    for portfolio_group_job in 759513_113 759513_114 759513_117 759513_121 759513_122 759513_123 759513_129; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-161652" ]]; then
    scancel 759513_185 759513_186 759513_191 759513_193 759513_194 759513_339 759513_403 759513_407 759513_435 759513_442 759513_458 759513_491 759513_496 759513_497 759513_499 759513_504 759513_505 759513_506 759513_507 759513_523 759513_528 759513_529 759513_531 759513_536 759513_537 759513_538 759513_539 759513_545 759513_546 759513_547 759513_603 759513_611
    sbatch --parsable --partition=default_partition --array=0-319%320 --cpus-per-task=1 --mem=4G --time=00:30:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-seeds run.sh scripts/analyze_portfolio_seed_shards.py
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-161120" ]]; then
    for portfolio_group_job in 759513_130 759513_131 759513_133 759513_476 759513_477 759513_478 759513_479 759513_480 759513_481; do scontrol update JobId="$portfolio_group_job" TimeLimit=00:30:00 TimeMin=00:30:00 || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-161102" ]]; then
    for portfolio_group_job in 759513_130 759513_131 759513_133; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-160454" ]]; then
    for portfolio_group_job in 759513_84; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-160350" ]]; then
    for portfolio_group_job in 759513_35 759513_39 759513_41 759513_42 759513_43 759513_74 759513_75 759513_82 759513_83; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-160303" ]]; then
    for portfolio_group_job in 759513_635 759513_31; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-160152" ]]; then
    for portfolio_group_job in 759513_498 759513_499 759513_503 759513_504 759513_505 759513_506 759513_507 759513_512 759513_513 759513_514 759513_515 759513_516 759513_517 759513_519 759513_520 759513_521 759513_522 759513_523 759513_528 759513_529 759513_530 759513_531 759513_536 759513_537 759513_538 759513_539 759513_545 759513_546 759513_547 759513_550 759513_554 759513_587 759513_602 759513_603 759513_608 759513_609 759513_610 759513_611 759513_615; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-160046" ]]; then
    for portfolio_group_job in 759513_495 759513_496 759513_497; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155941" ]]; then
    for portfolio_group_job in 759513_484 759513_485 759513_486 759513_487 759513_488 759513_489 759513_490 759513_491 759513_494; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155858" ]]; then
    scontrol release 759513_130 759513_131 759513_132 759513_133
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155844" ]]; then
    scontrol hold 759513_130 759513_131 759513_132 759513_133
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155750" ]]; then
    for portfolio_group_job in 759513_482 759513_483; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155610" ]]; then
    for portfolio_group_job in 759513_481 759513_480 759513_479 759513_478 759513_477 759513_476 759513_130 759513_131 759513_132 759513_133 759513_134 759513_135; do scontrol update JobId="$portfolio_group_job" TimeLimit=00:10:00 TimeMin=00:10:00 || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155542" ]]; then
    for portfolio_group_job in 759513_479 759513_480 759513_481; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155438" ]]; then
    for portfolio_group_job in 759513_476 759513_477 759513_478; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155350" ]]; then
    for portfolio_group_job in 759513_474 759513_475; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155247" ]]; then
    for portfolio_group_job in 759513_468 759513_469 759513_470 759513_471 759513_472 759513_473; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155142" ]]; then
    for portfolio_group_job in 759513_464 759513_465 759513_466 759513_467; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-155038" ]]; then
    for portfolio_group_job in 759513_456 759513_457 759513_458 759513_459; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-154934" ]]; then
    for portfolio_group_job in 759513_435 759513_436 759513_437 759513_438 759513_439 759513_440 759513_441 759513_442 759513_443 759513_446 759513_447 759513_451; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-154834" ]]; then
    for portfolio_group_job in 759513_225 759513_226 759513_227 759513_229 759513_230 759513_231 759513_232 759513_233 759513_234 759513_235 759513_243 759513_249 759513_250 759513_251 759513_253 759513_254 759513_255 759513_256 759513_257 759513_258 759513_259 759513_264 759513_265 759513_266 759513_267 759513_268 759513_269 759513_270 759513_271 759513_273 759513_274 759513_275 759513_289 759513_290 759513_291 759513_296 759513_297 759513_298 759513_299 759513_300 759513_301 759513_302 759513_303 759513_320 759513_321 759513_322 759513_323 759513_324 759513_325 759513_326 759513_327 759513_329 759513_330 759513_331 759513_334 759513_335 759513_336 759513_337 759513_338 759513_339 759513_340 759513_342 759513_347 759513_352 759513_353 759513_354 759513_355 759513_356 759513_361 759513_368 759513_369 759513_370 759513_371 759513_373 759513_374 759513_376 759513_377 759513_378 759513_379 759513_381 759513_383 759513_384 759513_386 759513_387 759513_391 759513_392 759513_393 759513_394 759513_395 759513_400 759513_401 759513_402 759513_403 759513_404 759513_405 759513_406 759513_407 759513_408 759513_409 759513_410 759513_411 759513_415 759513_417 759513_418 759513_419 759513_422 759513_423 759513_424 759513_425 759513_426 759513_427 759513_428 759513_429 759513_430 759513_431 759513_432 759513_433 759513_434; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-154748" ]]; then
    for portfolio_group_job in 759513_218 759513_219; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-154309" ]]; then
    for portfolio_group_job in 759513_130 759513_131 759513_132 759513_133 759513_134 759513_135 759513_143 759513_153 759513_154 759513_157 759513_161 759513_162 759513_163 759513_165 759513_166 759513_167 759513_171 759513_177 759513_178 759513_179 759513_180 759513_181 759513_182 759513_183 759513_184 759513_185 759513_186 759513_187 759513_188 759513_189 759513_190 759513_191 759513_192 759513_193 759513_194 759513_195 759513_196 759513_197 759513_198 759513_199 759513_201 759513_203; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-154006" ]]; then
    for portfolio_group_job in 759513_129; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-153919" ]]; then
    for portfolio_group_job in 759513_113 759513_114 759513_115 759513_116 759513_117 759513_118 759513_119 759513_121 759513_122 759513_123 759513_128; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-153530" ]]; then
    for portfolio_group_job in 759513_86 759513_87; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-153055" ]]; then
    for portfolio_group_job in 759513_85; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-153008" ]]; then
    for portfolio_group_job in 759513_84; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-152911" ]]; then
    scontrol update JobId=759513 ArrayTaskThrottle=500
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-152750" ]]; then
    for portfolio_group_job in 759513_80 759513_81 759513_82 759513_83; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-152216" ]]; then
    scontrol release 759513_1
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-152056" ]]; then
    for portfolio_group_job in 759513_78 759513_79; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-152012" ]]; then
    scontrol release 759513_2
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-151751" ]]; then
    for portfolio_group_job in 759513_77; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-curve-idle-capacity" ]]; then
    scontrol update JobId=759513 ArrayTaskThrottle=298
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-151532" ]]; then
    for portfolio_group_job in 759513_75 759513_76; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-monitor-20260910-151054" ]]; then
    for portfolio_group_job in 759513_65 759513_66 759513_67 759513_70 759513_71 759513_73 759513_74; do scontrol requeue "$portfolio_group_job" || exit; done
    exit
fi

if [[ "${1:-}" == "portfolio-curve-expand" ]]; then
    scontrol update JobId=759513 ArrayTaskThrottle=98
    exit
fi

if [[ "${1:-}" == "portfolio-curve-requeue-2" ]]; then
    scontrol release 759513_0 759513_3
    scontrol update JobId=759513 ArrayTaskThrottle=48
    exit
fi

if [[ "${1:-}" == "portfolio-curve-requeue-1" ]]; then
    scancel 759514
    for portfolio_group_job in 759513_4 759513_5 759513_7 759513_23 759513_25 759513_26 759513_27 759513_28 759513_29 759513_30 759513_33 759513_34 759513_35 759513_36 759513_37 759513_38 759513_39 759513_40 759513_41 759513_42 759513_43; do scontrol requeue "$portfolio_group_job"; done
    exit
fi

if [[ "${1:-}" == "portfolio-curve-local-slots" ]]; then
    scontrol hold 759513_0 759513_1 759513_2 759513_3
    scontrol update JobId=759513 ArrayTaskThrottle=46
    exit
fi

# scontrol update JobId=759513 TimeLimit=00:30:00
if [[ "${1:-}" == "portfolio-curve-groups" ]]; then
    portfolio_groups=$(sbatch --parsable --partition=default_partition --array=0-647%50 --cpus-per-task=1 --mem=4G --time=04:00:00 --time-min=00:30:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-groups run.sh scripts/analyze_portfolio_solve_over_time.py --group-task) || exit
    sbatch --dependency=afterok:"$portfolio_groups" --partition=default_partition --cpus-per-task=1 --mem=4G --time=00:15:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-plot run.sh scripts/analyze_portfolio_solve_over_time.py --collect-groups --render-only
    exit
fi

if [[ "${1:-}" == "portfolio-curve-group-prepare" ]]; then
    scancel 753564 757119
    python scripts/analyze_portfolio_solve_over_time.py --prepare-groups
    exit
fi

# scontrol update JobId=757119 ArrayTaskThrottle=16
# scontrol update JobId=757119 ArrayTaskThrottle=12
if [[ "${1:-}" == "portfolio-curve-release-retry-1" ]]; then
    scontrol update JobId=757119 ArrayTaskThrottle=8 || exit
    scontrol release 757119
    exit
fi

if [[ "${1:-}" == "portfolio-curve-retry-1" ]]; then
    # scontrol update JobId=753564 ArrayTaskThrottle=25
    scancel 753565
    sbatch --hold --parsable --partition=default_partition --array=0,2,3,7,8,9,10,11,13,14,15,17,18,21,22,23,24,25,26,35,36,38,39,44,48%25 --cpus-per-task=1 --mem=4G --time=04:00:00 --time-min=00:30:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-retry1 run.sh scripts/analyze_portfolio_solve_over_time.py --array-task
    exit
fi

# scontrol update JobId=753564 TimeLimit=00:30:00
if [[ "${1:-}" == "portfolio-curve" ]]; then
    portfolio_curve=$(sbatch --parsable --partition=default_partition --array=0-132%50 --cpus-per-task=1 --mem=4G --time=04:00:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-curve run.sh scripts/analyze_portfolio_solve_over_time.py --array-task) || exit
    sbatch --dependency=afterok:"$portfolio_curve" --partition=default_partition --cpus-per-task=1 --mem=4G --time=00:15:00 --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -J srb-portfolio-plot run.sh scripts/analyze_portfolio_solve_over_time.py --render-only
    exit
fi

srb_noearly_base=$(sbatch --parsable --dependency=afterany:750000 -J srb-1m-base-noearly run.sh srbench_full_eval.py --ground-truth --results-dir runs/srbench_gt_baseline_1m_noearly_1seed --seed 10000 --n-runs 1 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --max-evals 1000000 --no-early-stop --timeout 0 --pysr-wall-limit 1800 --partition default_partition --max-concurrent-jobs 100 --time-limit 00:40:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 8G --max-retries 5 --no-cache) || exit
srb_noearly_evolved=$(sbatch --parsable --dependency=afterany:750000 -J srb-1m-709715-noearly run.sh srbench_full_eval.py --ground-truth --evolve-results runs/709715 --results-dir runs/709715/srbench_gt_1m_noearly_1seed --seed 10000 --n-runs 1 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --max-evals 1000000 --no-early-stop --timeout 0 --pysr-wall-limit 1800 --partition default_partition --max-concurrent-jobs 100 --time-limit 00:40:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 8G --max-retries 5 --no-cache) || exit
sbatch --dependency=afterok:"$srb_noearly_base":"$srb_noearly_evolved" --partition=default_partition --cpus-per-task=1 --mem=2G --time=00:10:00 -J srb-1m-noearly-times run.sh scripts/summarize_srbench_search_time.py

sbatch -J srb-bb-autoresearch run.sh srbench_full_eval.py --autoresearch e425e7ed5894b0ebd7718fed0d77a252df2fa443 --black-box --results-dir runs/srbench_bb_autoresearch_10seed --max-evals 1000000 --seed 10000 --n-runs 10 --black-box-max-samples 10000 --black-box-timeout 1500 --black-box-wall-limit 1800 --partition default_partition --max-concurrent-jobs 100 --time-limit 01:00:00 --job-timeout 7200 --mem-per-cpu 8G --max-retries 5
sbatch -J neuron-eval-709715 run.sh neuron_full_eval.py --evolve-results runs/709715 --output-dir runs/709715/neuron_full_eval_5seed --n-runs 5 --seed 10000 --max-evals 1000000 --max-samples 1024 --partition default_partition --max-concurrent-jobs 30 --time-limit 00:15:00 --mem-per-cpu 8G --timeout 500 --pysr-wall-limit 600 --job-timeout 1800

# 9/9/26
sbatch -J srb-15m-base-single run.sh srbench_full_eval.py --ground-truth --results-dir runs/srbench_gt_baseline_15m_single --seed 10000 --n-runs 10 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --max-evals 1000000000 --timeout 900 --pysr-wall-limit 1200 --partition default_partition --max-concurrent-jobs 100 --time-limit 00:30:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 8G --max-retries 5 --no-cache

# 9/8/26
sbatch -J srb-merge-709715-taskpop run.sh srbench_full_eval.py --evolve-results runs/709715 --ground-truth --results-dir runs/709715/srbench_gt_taskpop_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --task-population-bundles 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5

srb_15m_base=$(sbatch --parsable -J srb-15m-base-portfolio run.sh srbench_full_eval.py --ground-truth --results-dir runs/srbench_gt_baseline_15m_portfolio_1e6 --seed 10000 --n-runs 10 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --portfolio-time-limit 900 --portfolio-restart-max-evals 1000000 --pysr-wall-limit 1200 --partition default_partition --max-concurrent-jobs 100 --time-limit 00:30:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 8G --max-retries 5 --no-cache)
sbatch --dependency=afterany:"$srb_15m_base" -J srb-15m-709715-portfolio run.sh srbench_full_eval.py --evolve-results runs/709715 --ground-truth --results-dir runs/709715/srbench_gt_15m_portfolio_1e6 --seed 10000 --n-runs 10 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --portfolio-time-limit 900 --portfolio-restart-max-evals 1000000 --pysr-wall-limit 1200 --partition default_partition --max-concurrent-jobs 100 --time-limit 00:30:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 8G --max-retries 5 --no-cache

sbatch -J eval-273201-trainval-90s-noearly run.sh evaluate_new_pysr.py --evolve-results runs/273201 --splits splits/barely_unsolvable.txt splits/barely_unsolvable_val2.txt --n-runs 10 --seed 192 --portfolio-time-limit 90 --portfolio-restart-timeout 90 --portfolio-restart-count 1 --pysr-wall-limit 270 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:20:00 --job-timeout 7200 --mem-per-cpu 8G --no-cache --no-early-stop --output-dir runs/273201/train_val_90s_10seed_no_early_stop

no_warmup_9_8=$(sbatch --parsable -J emp-9-8-base-8c-l1-nowarm run.sh empbench_full_eval.py --output-dir runs/empiricalbench_9-8_baseline_8core_l1_60m_no_warmup --n-runs 5 --seed 10000 --timeout 3600 --pysr-wall-limit 3900 --no-maxsize-warmup --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 8 --mem-per-cpu 2G --max-concurrent-jobs 45 --no-cache)
no_warmup_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_9_8" -J srb2-9-8-base-8c-l1-nowarm run.sh srbench2_full_eval.py --ground-truth --results-dir runs/srbench2_9-8_gt_baseline_8core_l1_60m_no_warmup --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --no-maxsize-warmup --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 8 --baseline-l1-loss --mem-per-cpu 10G --max-concurrent-jobs 45 --no-cache --no-wandb)
no_warmup_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_9_8" -J srb2-9-8-base-1c-l1-nowarm run.sh srbench2_full_eval.py --ground-truth --results-dir runs/srbench2_9-8_gt_baseline_1core_l1_60m_no_warmup --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --no-maxsize-warmup --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --baseline-l1-loss --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb)
no_warmup_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_9_8" -J srb2-9-8-base-portfolio-nowarm run.sh srbench2_full_eval.py --ground-truth --results-dir runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup --n-runs 10 --seed 10000 --noise-levels 0 --portfolio-time-limit 3600 --portfolio-restart-max-evals 1000000 --pysr-wall-limit 3900 --no-maxsize-warmup --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --baseline-l1-loss --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb)
no_warmup_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_9_8" -J srb2-9-8-709715-1c-nowarm run.sh srbench2_full_eval.py --ground-truth --evolve-results runs/709715 --results-dir runs/709715/srbench2_9-8_ground_truth_1core_60m_no_warmup --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --no-maxsize-warmup --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb)
no_warmup_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_9_8" -J srb2-9-8-709715-portfolio-nowarm run.sh srbench2_full_eval.py --ground-truth --evolve-results runs/709715 --results-dir runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup --n-runs 10 --seed 10000 --noise-levels 0 --portfolio-time-limit 3600 --portfolio-restart-max-evals 1000000 --pysr-wall-limit 3900 --no-maxsize-warmup --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb)

no_warmup_review_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J emp-9-8-review-base-8c-l1-nowarm run.sh scripts/review_srbench2_frontiers.py runs/empiricalbench_9-8_baseline_8core_l1_60m_no_warmup --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 5)
no_warmup_review_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_review_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-8-review-base-8c-l1-nowarm run.sh scripts/review_srbench2_frontiers.py runs/srbench2_9-8_gt_baseline_8core_l1_60m_no_warmup --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10)
no_warmup_review_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_review_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-8-review-base-1c-l1-nowarm run.sh scripts/review_srbench2_frontiers.py runs/srbench2_9-8_gt_baseline_1core_l1_60m_no_warmup --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10)
no_warmup_review_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_review_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-8-review-base-portfolio-nowarm run.sh scripts/review_srbench2_frontiers.py runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10)
no_warmup_review_9_8=$(sbatch --parsable --dependency=afterany:"$no_warmup_review_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-8-review-709715-1c-nowarm run.sh scripts/review_srbench2_frontiers.py runs/709715/srbench2_9-8_ground_truth_1core_60m_no_warmup --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10)
sbatch --dependency=afterany:"$no_warmup_review_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-8-review-709715-portfolio-nowarm run.sh scripts/review_srbench2_frontiers.py runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10

trainval_90s=$(sbatch --parsable -J eval-709715-trainval-90s run.sh evaluate_new_pysr.py --evolve-results runs/709715 --splits splits/barely_unsolvable.txt splits/barely_unsolvable_val2.txt --n-runs 10 --seed 192 --portfolio-time-limit 90 --portfolio-restart-timeout 90 --portfolio-restart-count 1 --pysr-wall-limit 270 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:20:00 --job-timeout 7200 --mem-per-cpu 8G --no-cache --output-dir runs/709715/train_val_90s_10seed)
sbatch --dependency=afterany:"$trainval_90s" -J eval-273201-trainval-90s run.sh evaluate_new_pysr.py --evolve-results runs/273201 --splits splits/barely_unsolvable.txt splits/barely_unsolvable_val2.txt --n-runs 10 --seed 192 --portfolio-time-limit 90 --portfolio-restart-timeout 90 --portfolio-restart-count 1 --pysr-wall-limit 270 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:20:00 --job-timeout 7200 --mem-per-cpu 8G --no-cache --output-dir runs/273201/train_val_90s_10seed

retry_9_8=$(sbatch --parsable -J srb2-9-8-retry-base-8c-l1 run.sh scripts/retry_pysr_errors.py runs/srbench2_9-4_gt_baseline_8core_l1 --cpus-per-task 8 --time-limit 01:15:00 --job-timeout 7200 --mem-per-cpu 10G --max-concurrent-jobs 60 --max-retries 5)
retry_9_8=$(sbatch --parsable --dependency=afterany:"$retry_9_8" -J srb2-9-8-retry-709715-8c run.sh scripts/retry_pysr_errors.py runs/709715/srbench2_9-4_ground_truth_8core --cpus-per-task 8 --time-limit 01:15:00 --job-timeout 7200 --mem-per-cpu 10G --max-concurrent-jobs 60 --max-retries 5)
retry_9_8=$(sbatch --parsable --dependency=afterany:"$retry_9_8" -J emp-9-8-retry-709715-8c run.sh scripts/retry_pysr_errors.py runs/709715/empiricalbench_paper/evolved --cpus-per-task 8 --time-limit 01:15:00 --job-timeout 7200 --mem-per-cpu 2G --max-concurrent-jobs 45 --max-retries 5)
retry_9_8=$(sbatch --parsable --dependency=afterany:"$retry_9_8" -J srb2-9-8-709715-portfolio run.sh srbench2_full_eval.py --ground-truth --evolve-results runs/709715 --results-dir runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m --n-runs 10 --seed 10000 --noise-levels 0 --portfolio-time-limit 3600 --portfolio-restart-max-evals 1000000 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb)
review_9_8=$(sbatch --parsable --dependency=afterany:"$retry_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-8-review-base-8c-l1 run.sh scripts/review_srbench2_frontiers.py runs/srbench2_9-4_gt_baseline_8core_l1 --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10 --force)
review_9_8=$(sbatch --parsable --dependency=afterany:"$review_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-8-review-709715-8c run.sh scripts/review_srbench2_frontiers.py runs/709715/srbench2_9-4_ground_truth_8core --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10 --force)
review_9_8=$(sbatch --parsable --dependency=afterany:"$review_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J emp-9-8-review-709715-8c run.sh scripts/review_srbench2_frontiers.py runs/709715/empiricalbench_paper/evolved --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 5 --force)
sbatch --dependency=afterany:"$review_9_8" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-8-review-709715-portfolio run.sh scripts/review_srbench2_frontiers.py runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10

# 9/4/26

srb2_protocol=$(sbatch --parsable --dependency=afterany:273201 -J srb2-9-4-base-8c-l1 run.sh srbench2_full_eval.py --ground-truth --results-dir runs/srbench2_9-4_gt_baseline_8core_l1 --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 8 --baseline-l1-loss --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb)
srb2_protocol=$(sbatch --parsable --dependency=afterany:"$srb2_protocol" -J srb2-9-4-base-1c-l1 run.sh srbench2_full_eval.py --ground-truth --results-dir runs/srbench2_9-4_gt_baseline_1core_l1 --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --baseline-l1-loss --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb)
srb2_protocol=$(sbatch --parsable --dependency=afterany:"$srb2_protocol" -J srb2-9-4-709715-8c run.sh srbench2_full_eval.py --ground-truth --evolve-results runs/709715 --results-dir runs/709715/srbench2_9-4_ground_truth_8core --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 8 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb)
srb2_protocol=$(sbatch --parsable --dependency=afterany:"$srb2_protocol" -J srb2-9-4-base-portfolio run.sh srbench2_full_eval.py --ground-truth --results-dir runs/srbench2_9-4_gt_baseline_1core_portfolio_1m --n-runs 10 --seed 10000 --noise-levels 0 --portfolio-time-limit 3600 --portfolio-restart-max-evals 1000000 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb)
srb2_protocol=$(sbatch --parsable --dependency=afterany:"$srb2_protocol" -J emp-9-4-base-portfolio run.sh empbench_full_eval.py --output-dir runs/empiricalbench_9-4_baseline_1core_portfolio_1m --n-runs 5 --seed 10000 --timeout 3600 --pysr-wall-limit 3900 --portfolio-time-limit 3600 --portfolio-restart-max-evals 1000000 --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 1 --mem-per-cpu 10G --max-concurrent-jobs 45 --no-cache)
srb2_review=$(sbatch --parsable --dependency=afterany:"$srb2_protocol" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-4-review-base-8c-l1 run.sh scripts/review_srbench2_frontiers.py runs/srbench2_9-4_gt_baseline_8core_l1 --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10)
srb2_review=$(sbatch --parsable --dependency=afterany:"$srb2_review" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-4-review-base-1c-l1 run.sh scripts/review_srbench2_frontiers.py runs/srbench2_9-4_gt_baseline_1core_l1 --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10)
srb2_review=$(sbatch --parsable --dependency=afterany:"$srb2_review" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-4-review-709715-8c run.sh scripts/review_srbench2_frontiers.py runs/709715/srbench2_9-4_ground_truth_8core --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10)
srb2_review=$(sbatch --parsable --dependency=afterany:"$srb2_review" --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-4-review-portfolio run.sh scripts/review_srbench2_frontiers.py runs/srbench2_9-4_gt_baseline_1core_portfolio_1m --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 10)
sbatch --dependency=afterany:"$srb2_review" --export=ALL --time=1-02:00:00 --mem=2G -J emp-9-4-review-portfolio run.sh scripts/review_srbench2_frontiers.py runs/empiricalbench_9-4_baseline_1core_portfolio_1m --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 10000 --max-cost 5

sbatch --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-4-review-base run.sh scripts/review_srbench2_frontiers.py runs/srbench2_9-4_gt_baseline --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 1000 --max-cost 2
sbatch --export=ALL --time=1-02:00:00 --mem=2G -J srb2-9-4-review-709715 run.sh scripts/review_srbench2_frontiers.py runs/709715/srbench2_9-4_ground_truth --model openai/gpt-5.6-terra --reasoning-effort medium --max-output-tokens 1000 --max-cost 2
srb_90s_base=$(sbatch --parsable --dependency=afterany:273895:273896 -J srb-90s-base run.sh srbench_full_eval.py --ground-truth --results-dir runs/srbench_gt_baseline_90s --seed 10000 --n-runs 10 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --portfolio-time-limit 90 --portfolio-restart-timeout 90 --portfolio-restart-count 1 --pysr-wall-limit 270 --partition default_partition --max-concurrent-jobs 100 --time-limit 00:20:00 --mem-per-cpu 8G --max-retries 5)
sbatch --dependency=afterany:"$srb_90s_base" -J srb-90s-709715 run.sh srbench_full_eval.py --evolve-results runs/709715 --ground-truth --results-dir runs/709715/srbench_gt_90s --seed 10000 --n-runs 10 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --portfolio-time-limit 90 --portfolio-restart-timeout 90 --portfolio-restart-count 1 --pysr-wall-limit 270 --partition default_partition --max-concurrent-jobs 100 --time-limit 00:20:00 --mem-per-cpu 8G --max-retries 5
sbatch -J evolve-srbench-90s run.sh evolve_pysr.py --operator-type all --population-type task --generations 45 --simplify-cooldown 15 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt --val-n-runs 10 --identify-topk 10 --final-eval-runs 10 --max-time-in-seconds 90 --pysr-wall-limit 270 --val-pysr-timeout 90 --val-pysr-wall-limit 270

srb_aggregate=$(sbatch --parsable -J srb-aggregate-base run.sh srbench_full_eval.py --ground-truth --results-dir runs/srbench_gt_baseline_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
sbatch --dependency=afterany:"$srb_aggregate" -J srb-aggregate-709715 run.sh srbench_full_eval.py --evolve-results runs/709715 --ground-truth --results-dir runs/709715/srbench_gt_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5

# sbatch -J srb2-9-4-base run.sh srbench2_full_eval.py --ground-truth --results-dir runs/srbench2_9-4_gt_baseline --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb
# sbatch -J srb2-9-4-709715 run.sh srbench2_full_eval.py --ground-truth --evolve-results runs/709715 --results-dir runs/709715/srbench2_9-4_ground_truth --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb

# New frontier recomputation.
# srb_merge=$(sbatch --parsable -J srb-merge-autores run.sh srbench_full_eval.py --autoresearch best --ground-truth --results-dir runs/srbench_gt_autoresearch_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# srb_merge=$(sbatch --parsable --dependency=afterany:"$srb_merge" -J srb-merge-basic-base run.sh srbench_full_eval.py --fullsr-baseline --ground-truth --results-dir runs/srbench_gt_basicsr_baseline_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --fullsr-wall-limit 600 --partition default_partition --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# srb_merge=$(sbatch --parsable --dependency=afterany:"$srb_merge" -J srb-merge-hpo-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_180547_120309 --ground-truth --results-dir runs/srbench_gt_hpo_gt_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# srb_merge=$(sbatch --parsable --dependency=afterany:"$srb_merge" -J srb-merge-basicpp-gt run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --results-dir runs/225437/srbench_gt_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --fullsr-wall-limit 600 --partition default_partition --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# srb_merge=$(sbatch --parsable --dependency=afterany:"$srb_merge" -J srb-merge-hpo-gtr2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_190637_506162 --ground-truth --results-dir runs/srbench_gt_hpo_gtr2_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# srb_merge=$(sbatch --parsable --dependency=afterany:"$srb_merge" -J srb-merge-pysrpp-gtr2 run.sh srbench_full_eval.py --evolve-results runs/120459 --ground-truth --results-dir runs/120459/srbench_gt_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# srb_merge=$(sbatch --parsable --dependency=afterany:"$srb_merge" -J srb-merge-basicpp-gtr2 run.sh srbench_full_eval.py --evolve-results runs/150815 --ground-truth --results-dir runs/150815/srbench_gt_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --fullsr-wall-limit 600 --partition default_partition --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# srb_merge=$(sbatch --parsable --dependency=afterany:"$srb_merge" -J srb-merge-hpo-r2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_183759_524347 --ground-truth --results-dir runs/srbench_gt_hpo_r2_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# srb_merge=$(sbatch --parsable --dependency=afterany:"$srb_merge" -J srb-merge-pysrpp-r2 run.sh srbench_full_eval.py --evolve-results runs/120458 --ground-truth --results-dir runs/120458/srbench_gt_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# sbatch --dependency=afterany:"$srb_merge" -J srb-merge-basicpp-r2 run.sh srbench_full_eval.py --evolve-results runs/150812 --ground-truth --results-dir runs/150812/srbench_gt_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --fullsr-wall-limit 600 --partition default_partition --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5

# 9/3/26

# merged_base=$(sbatch --parsable --dependency=afterany:249072:249073:249074:249075 -J srb-merge-base run.sh srbench_full_eval.py --ground-truth --results-dir runs/srbench_gt_baseline_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5)
# sbatch --dependency=afterany:"$merged_base" -J srb-merge-709715 run.sh srbench_full_eval.py --evolve-results runs/709715 --ground-truth --results-dir runs/709715/srbench_gt_10seed_merged --max-evals 1000000 --timeout 500 --seed 10000 --n-runs 10 --merge-run-frontiers --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5

# sbatch -J srb-retry-basicpp-gt run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --max-evals 1000000 --timeout 500 --seed 42 --n-runs 10 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --fullsr-wall-limit 600 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5
# sbatch -J srb-retry-base-10m run.sh srbench_full_eval.py --ground-truth --max-evals 10000000 --timeout 500 --seed 42 --n-runs 10 --noise-levels 0 0.001 0.01 0.1 --max-samples 1000 --pysr-wall-limit 600 --pysr-progress --partition default_partition --max-concurrent-jobs 100 --time-limit 02:00:00 --mem-per-cpu 8G --max-retries 5

# SRBench 2.0 phenomenological/first-principles track: 12 datasets x 5 seeds,
# intrinsic noise only, one-hour search, and one CPU as in the 2025 benchmark.
# sbatch -J srb2-gt-base run.sh srbench2_full_eval.py --ground-truth --results-dir runs/srbench2_gt_baseline --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb
# sbatch -J srb2-gt-709715 run.sh srbench2_full_eval.py --ground-truth --evolve-results runs/709715 --results-dir runs/709715/srbench2_ground_truth --n-runs 10 --seed 10000 --noise-levels 0 --max-evals 1000000000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --mem-per-cpu 10G --max-concurrent-jobs 60 --no-cache --no-wandb

# 9/2/26

# sbatch -J emp-paper-ft run.sh empbench_full_eval.py --evolve-results runs/147300 --output-dir runs/147300/empiricalbench_paper/evolved --n-runs 5 --seed 10000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 8 --mem-per-cpu 2G --max-concurrent-jobs 45 --no-cache

# LaSR ground-truth SRBench subset: 20 equations x noise 0.001 x 1 seed.
# Estimated API cost: $8-$33; estimated raw LLM logs: 1-2 GB.
# python evaluate_lasr_srbench.py plan --split-file splits/lasr_20.txt --noise-levels 0.001
# python evaluate_lasr_srbench.py submit --split-file splits/lasr_20.txt --noise-levels 0.001 --output-dir runs/lasr_srbench_nemo_noise0p001_20tasks --max-concurrent 20

# LaSR ground-truth SRBench remainder: 113 equations x noise 0.001 x 1 seed.
# Estimated API cost: $46-$185; estimated raw LLM logs: 3-11 GB.
# python evaluate_lasr_srbench.py plan --split-file splits/lasr_remaining_113.txt --noise-levels 0.001
# python evaluate_lasr_srbench.py submit --split-file splits/lasr_remaining_113.txt --noise-levels 0.001 --output-dir runs/lasr_srbench_nemo_noise0p001_remaining113 --max-concurrent 32

# sbatch -J hpo-709715-barely run.sh hpo_pysr.py --baseline runs/709715 --n-trials 300 --n-runs 3 --n-parallel 20 --fitness-metric gt --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt

# sbatch -J 20m-ft-709715 run.sh evolve_pysr.py --operator-type all --baseline runs/709715 --generations 4 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --population-type task --reeval topk --reeval-topk 2 --n-reevals 10 --models best2 --domain mips --split splits/srbench_top_half_20min.txt --val-split splits/barely_unsolvable_val2.txt --val-n-runs 2 --identify-topk 1 --max-time-in-seconds 1200 --pysr-wall-limit 1800 --val-pysr-timeout 1200 --val-pysr-wall-limit 1800 --time-limit 00:35:00 --job-timeout 3600 --random-target-noise
# MIPS evolution on ten hard relations plus five seed-42 sampled easy relations.
# MIPS_ARTIFACT_SPLIT="splits/mips_sr_targets_plus_refined.txt"
# MIPS_EVOLVE_SPLIT="splits/mips_hard10_easy5.txt"
# MIPS_EVOLVE_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts"
# python scripts/prepare_mips_evolution_overlay.py --split "$MIPS_ARTIFACT_SPLIT" --output-root "$MIPS_EVOLVE_ROOT"
# sbatch --partition=default_partition --time=3-00:00:00 --cpus-per-task=1 --mem=8G --job-name=mips-evolve-hard15 --export=ALL,MIPS_TRANSITION_ROOT="$MIPS_EVOLVE_ROOT" run.sh evolve_pysr.py --domain mips --operator-type all --baseline runs/709715 --generations 5 --population 5 --offspring 5 --n-runs 2 --seed 42 --fitness-metric gt --population-type task --reeval topk --n-reevals 3 --reeval-topk 1 --identify-topk 0 --exec-feedback-n 3 --exec-feedback-prob 0.5 --models best2 --split "$MIPS_EVOLVE_SPLIT" --val-split "" --val-n-runs 2 --final-eval-runs 10 --max-samples 1000 --max-time-in-seconds 1200 --pysr-wall-limit 1800 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:35:00 --mem-per-cpu 8G --job-timeout 7200 --no-random-target-noise

# Paper-protocol EmpiricalBench rerun: all rows, paper search space, float64,
# eight processes, five seeds, and 60 minutes of search per fit. Baseline uses
# L1; evolved 709715 keeps its custom loss. Prepared only, not submitted.
# sbatch -J emp-paper-base run.sh empbench_full_eval.py --output-dir runs/709715/empiricalbench_paper/baseline --n-runs 5 --seed 10000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 8 --mem-per-cpu 2G --max-concurrent-jobs 45 --no-cache
# sbatch -J emp-paper-709715 run.sh empbench_full_eval.py --evolve-results runs/709715 --output-dir runs/709715/empiricalbench_paper/evolved --n-runs 5 --seed 10000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --cpus-per-task 8 --mem-per-cpu 2G --max-concurrent-jobs 45 --no-cache

# 9/1/26

# sbatch --dependency=afterany:107098 -J 20m-ft-709715-medium run.sh evolve_pysr.py --operator-type all --baseline runs/709715 --generations 10 --population 10 --offspring 10 --n-runs 1 --fitness-metric gt --population-type task --reeval topk --reeval-topk 2 --n-reevals 3 --models best2 --split splits/medium_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt --val-n-runs 1 --identify-topk 2 --max-time-in-seconds 1200 --pysr-wall-limit 1800 --val-pysr-timeout 1200 --val-pysr-wall-limit 1800 --time-limit 00:35:00 --job-timeout 3600

# sbatch -J ablate-709715 --dependency=afterany:103977 run.sh scripts/evaluate_operator_ablation.py --bundle-jl runs/709715/best_bundles/best_gen43.jl --splits splits/barely_unsolvable.txt splits/barely_unsolvable_val2.txt --n-runs 10 --seed 192 --max-samples 1000 --max-evals 1000000 --timeout 500 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:15:00 --mem-per-cpu 8G --job-timeout 14400 --no-cache --output-dir runs/709715/operator_ablation_gen43

# All nine EmpiricalBench tasks, five paired seeds. Each driver creates and
# collects its own 45-task array; each fit searches for one hour with no eval cap.
# sbatch -J emp-base run.sh empbench_full_eval.py --output-dir runs/709715/empiricalbench_comparison/baseline --n-runs 5 --seed 10000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --mem-per-cpu 8G --max-concurrent-jobs 45 --no-cache
# sbatch -J emp-709715 run.sh empbench_full_eval.py --evolve-results runs/709715 --output-dir runs/709715/empiricalbench_comparison/evolved --n-runs 5 --seed 10000 --timeout 3600 --pysr-wall-limit 3900 --time-limit 01:15:00 --job-timeout 7200 --mem-per-cpu 8G --max-concurrent-jobs 45 --no-cache

# sbatch -J r2-bb-train run.sh evolve_pysr.py --operator-type all --generations 30 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric r2 --population-type task --reeval population --n-reevals 10 --models best2 --split splits/bb_train.txt --val-split splits/bb_val.txt --test-split splits/bb_test.txt

# evolve_after_hpo=$(sbatch --parsable -J evolve-after-hpo run.sh evolve_pysr.py --operator-type all --baseline outputs/hpo_pysr_20260727_172105_644009 --generations 30 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2)
# hpo_evolved_base=$(sbatch --parsable -J hpo-709715 run.sh hpo_pysr.py --baseline runs/709715 --n-trials 300 --n-runs 3 --n-parallel 20 --fitness-metric gt)


# 8/31
# Paired comparison on the exact run-709716 synthetic manifests and data seeds:
# official noisy Boolformer (10 sampled formulas), base PySR, and evolved PySR.
# The evolved results already exist; the final dependent job only aggregates.
# Submitted as base=983618, official=984175, report=984176. The first official
# attempt (983619) failed during dependency bootstrap; report 983620 was canceled.
# BOOLFORMER_COMPARE="runs/709716/method_comparison"
# mkdir -p "$BOOLFORMER_COMPARE"
# boolformer_base=983618
# boolformer_base=$(sbatch --parsable --partition=default_partition --time=08:00:00 --cpus-per-task=1 --mem=8G --job-name=bf-base-pysr run.sh evaluate_new_pysr.py --domain boolformer --fitness-metric gt-acc --splits splits/boolformer_noisy_stratified_train.txt splits/boolformer_noisy_stratified_val.txt splits/boolformer_noisy_test.txt --n-runs 10 --seed 192 --max-samples 1000 --partition default_partition --max-concurrent-jobs 300 --time-limit 01:00:00 --mem-per-cpu 8G --pysr-wall-limit 2400 --job-timeout 14400 --no-cache --output-dir "$BOOLFORMER_COMPARE/base_pysr")
# boolformer_official=$(sbatch --parsable --partition=default_partition --time=08:00:00 --cpus-per-task=4 --mem=32G --gres=gpu:1 --job-name=bf-official run.sh scripts/evaluate_official_boolformer.py --splits splits/boolformer_noisy_stratified_train.txt splits/boolformer_noisy_stratified_val.txt splits/boolformer_noisy_test.txt --n-runs 10 --seed 192 --beam-size 10 --output-dir "$BOOLFORMER_COMPARE/official_boolformer")
# boolformer_report=$(sbatch --parsable --dependency=afterok:"$boolformer_base":"$boolformer_official" --partition=default_partition --time=00:15:00 --cpus-per-task=1 --mem=4G --job-name=bf-compare-report run.sh scripts/report_boolformer_comparison.py --official "$BOOLFORMER_COMPARE/official_boolformer/results.json" --base "$BOOLFORMER_COMPARE/base_pysr/eval_summary.json" --evolved runs/709716/final_eval_summary.json --output-json "$BOOLFORMER_COMPARE/comparison.json" --output-md "$BOOLFORMER_COMPARE/README.md")
# echo "Submitted Boolformer comparison: base=$boolformer_base official=$boolformer_official report=$boolformer_report"

# Retry after pinning the old PMLB API imported by Boolformer 0.1.9.
# Submitted as official=16323 and report=16326; base=983618 was already complete.
# boolformer_base=983618
# boolformer_official=$(sbatch --parsable --partition=default_partition --time=08:00:00 --cpus-per-task=4 --mem=32G --gres=gpu:1 --job-name=bf-official run.sh scripts/evaluate_official_boolformer.py --splits splits/boolformer_noisy_stratified_train.txt splits/boolformer_noisy_stratified_val.txt splits/boolformer_noisy_test.txt --n-runs 10 --seed 192 --beam-size 10 --output-dir "$BOOLFORMER_COMPARE/official_boolformer")
# boolformer_official=16323
# boolformer_report=$(sbatch --parsable --dependency=afterok:"$boolformer_official" --partition=default_partition --time=00:15:00 --cpus-per-task=1 --mem=4G --job-name=bf-compare-report run.sh scripts/report_boolformer_comparison.py --official "$BOOLFORMER_COMPARE/official_boolformer/results.json" --base "$BOOLFORMER_COMPARE/base_pysr/eval_summary.json" --evolved runs/709716/final_eval_summary.json --output-json "$BOOLFORMER_COMPARE/comparison.json" --output-md "$BOOLFORMER_COMPARE/README.md")
# echo "Retried Boolformer comparison: base=$boolformer_base official=$boolformer_official report=$boolformer_report"

# Retry with explicit paths after the prior shell variable was commented out.
# Submitted as official=17562 and report=17563.
# boolformer_official=$(sbatch --parsable --partition=default_partition --time=08:00:00 --cpus-per-task=4 --mem=32G --gres=gpu:1 --job-name=bf-official run.sh scripts/evaluate_official_boolformer.py --splits splits/boolformer_noisy_stratified_train.txt splits/boolformer_noisy_stratified_val.txt splits/boolformer_noisy_test.txt --n-runs 10 --seed 192 --beam-size 10 --output-dir /home/sca63/meta_sr/runs/709716/method_comparison/official_boolformer)
# boolformer_report=$(sbatch --parsable --dependency=afterok:"$boolformer_official" --partition=default_partition --time=00:15:00 --cpus-per-task=1 --mem=4G --job-name=bf-compare-report run.sh scripts/report_boolformer_comparison.py --official /home/sca63/meta_sr/runs/709716/method_comparison/official_boolformer/results.json --base /home/sca63/meta_sr/runs/709716/method_comparison/base_pysr/eval_summary.json --evolved /home/sca63/meta_sr/runs/709716/final_eval_summary.json --output-json /home/sca63/meta_sr/runs/709716/method_comparison/comparison.json --output-md /home/sca63/meta_sr/runs/709716/method_comparison/README.md)
# echo "Retried Boolformer comparison: official=$boolformer_official report=$boolformer_report"

# sbatch -J gt-task-srb run.sh srbench_full_eval.py --evolve-results runs/709715 --ground-truth --black-box --max-evals 1000000 --timeout 0 --pysr-wall-limit 900
# sbatch --export=ALL,MIPS_TRANSITION_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts" -J mips-base-10 run.sh evaluate_new_pysr.py --domain mips --fitness-metric gt --splits splits/mips_sr_targets_plus_refined.txt --n-runs 10 --seed 192 --max-samples 1000 --max-evals 1000000 --timeout 500 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:15:00 --mem-per-cpu 8G --job-timeout 1800 --no-cache --output-dir runs/709714/final_eval_baseline_10seed
# sbatch --export=ALL,MIPS_TRANSITION_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts" -J mips-709715-10 run.sh evaluate_new_pysr.py --evolve-results runs/709715 --domain mips --fitness-metric gt --splits splits/mips_sr_targets_plus_refined.txt --n-runs 10 --seed 192 --max-samples 1000 --max-evals 1000000 --timeout 500 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:15:00 --mem-per-cpu 8G --job-timeout 1800 --no-cache --output-dir runs/709715/final_eval_mips_10seed
# sbatch --export=ALL,MIPS_TRANSITION_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts" -J mips-709715-native run.sh evaluate_new_pysr.py --evolve-results runs/709715 --domain mips --use-domain-defaults --fitness-metric gt --splits splits/mips_sr_targets_plus_refined.txt --n-runs 10 --seed 192 --max-samples 1000 --max-evals 1000000 --timeout 500 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:15:00 --mem-per-cpu 8G --job-timeout 1800 --no-cache --output-dir runs/709715/final_eval_mips_native_10seed

# new mips run
# MIPS_EVOLVE_SPLIT="splits/mips_sr_targets_plus_refined.txt"
# MIPS_EVOLVE_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts"
# python scripts/prepare_mips_evolution_overlay.py --split "$MIPS_EVOLVE_SPLIT" --output-root "$MIPS_EVOLVE_ROOT"
# chain_b=$(sbatch --parsable --dependency=afterany:950928 --partition=default_partition --time=3-00:00:00 --cpus-per-task=1 --mem=8G --job-name=mips-evolve-51 --export=ALL,MIPS_TRANSITION_ROOT="$MIPS_EVOLVE_ROOT" run.sh evolve_pysr.py --domain mips --operator-type all --baseline runs/709715 --generations 25 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --seed 42 --fitness-metric gt --population-type task --reeval population --n-reevals 10 --identify-topk 0 --exec-feedback-n 3 --exec-feedback-prob 0.5 --models best2 --split "$MIPS_EVOLVE_SPLIT" --val-split "" --final-eval-runs 10 --max-samples 1000 --max-evals 1000000 --timeout 500 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:15:00 --mem-per-cpu 8G --job-timeout 14400 --no-random-target-noise)

# Three-seed, one-hour-per-fit MIPS comparison: baseline -> MIPS-evolved -> SRBench-evolved.
# mips_hourly=$(sbatch --parsable --partition=default_partition --time=05:00:00 --cpus-per-task=1 --mem=8G --job-name=mips-1h-base --export=ALL,MIPS_TRANSITION_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts" run.sh evaluate_new_pysr.py --domain mips --fitness-metric gt --splits splits/mips_sr_targets_plus_refined.txt --n-runs 3 --seed 192 --max-samples 1000 --wall-clock-only --timeout 3600 --pysr-wall-limit 3900 --partition default_partition --max-concurrent-jobs 153 --time-limit 01:20:00 --mem-per-cpu 8G --job-timeout 14400 --no-cache --output-dir runs/709714/final_eval_baseline_3seed_1h)
# mips_hourly=$(sbatch --parsable --dependency=afterany:"$mips_hourly" --partition=default_partition --time=05:00:00 --cpus-per-task=1 --mem=8G --job-name=mips-1h-709714 --export=ALL,MIPS_TRANSITION_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts" run.sh evaluate_new_pysr.py --evolve-results runs/709714 --domain mips --fitness-metric gt --splits splits/mips_sr_targets_plus_refined.txt --n-runs 3 --seed 192 --max-samples 1000 --wall-clock-only --timeout 3600 --pysr-wall-limit 3900 --partition default_partition --max-concurrent-jobs 153 --time-limit 01:20:00 --mem-per-cpu 8G --job-timeout 14400 --no-cache --output-dir runs/709714/final_eval_mips_3seed_1h)
# mips_hourly=$(sbatch --parsable --dependency=afterany:"$mips_hourly" --partition=default_partition --time=05:00:00 --cpus-per-task=1 --mem=8G --job-name=mips-1h-709715 --export=ALL,MIPS_TRANSITION_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts" run.sh evaluate_new_pysr.py --evolve-results runs/709715 --domain mips --use-domain-defaults --fitness-metric gt --splits splits/mips_sr_targets_plus_refined.txt --n-runs 3 --seed 192 --max-samples 1000 --wall-clock-only --timeout 3600 --pysr-wall-limit 3900 --partition default_partition --max-concurrent-jobs 153 --time-limit 01:20:00 --mem-per-cpu 8G --job-timeout 14400 --no-cache --output-dir runs/709715/final_eval_mips_native_3seed_1h)
#

# 8/28 — Finish the three incomplete 1M SRBench black-box columns shown by
# inspect_srbench_results.py --official. Successful trials are cache hits, so
# these black-box-only reruns should execute just the missing/failed trials.
# Give every fit up to 9h soft / 10h hard, each array task 12h, and each driver
# two days so ten retry rounds have ample time. These are intentionally not
# submitted automatically; uncomment the three sbatch lines when ready.
# sbatch --time=2-00:00:00 -J bb-retry-hpo300-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_180547_120309 --black-box --black-box-timeout 32400 --black-box-wall-limit 36000 --time-limit 12:00:00 --job-timeout 43200 --max-retries 2
# sbatch --time=2-00:00:00 -J bb-retry-hpo300-gtr2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_190637_506162 --black-box --black-box-timeout 32400 --black-box-wall-limit 36000 --time-limit 12:00:00 --job-timeout 43200 --max-retries 2
# sbatch --time=2-00:00:00 -J bb-retry-pysrpp-r2 run.sh srbench_full_eval.py --evolve-results runs/120458 --black-box --black-box-timeout 32400 --black-box-wall-limit 36000 --time-limit 12:00:00 --job-timeout 43200 --max-retries 2

# 8/28
# sbatch -J hpo300-gt-1e7-srb run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_180547_120309 --ground-truth --max-evals 10000000 --timeout 0 --pysr-wall-limit 900

# 8/27
# chain_a=$(sbatch --parsable -J neuron-top1-uninformative run.sh evolve_pysr.py --domain neuron --uninformative-prompts --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models medium2 --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first1.txt --val-split "" --seed 0)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J neuron-top2-uninformative run.sh evolve_pysr.py --domain neuron --uninformative-prompts --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models medium2 --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first2.txt --val-split "" --seed 0)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J neuron-top3-uninformative run.sh evolve_pysr.py --domain neuron --uninformative-prompts --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models medium2 --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first3.txt --val-split "" --seed 0)

# 8/27
# relations plus the 17 LR-unsolved, deterministic refined-state relations.
# This is a 51-relation train-only run; no validation split or reevaluation.
# MIPS_EVOLVE_SPLIT="splits/mips_sr_targets_plus_refined.txt"
# MIPS_EVOLVE_ROOT="$(pwd)/outputs/mips_evolution_51_artifacts"
# python scripts/prepare_mips_evolution_overlay.py --split "$MIPS_EVOLVE_SPLIT" --output-root "$MIPS_EVOLVE_ROOT"
# chain_b=$(sbatch --parsable --partition=default_partition --time=2-00:00:00 --cpus-per-task=1 --mem=8G --job-name=mips-evolve-51 --export=ALL,MIPS_TRANSITION_ROOT="$MIPS_EVOLVE_ROOT" run.sh evolve_pysr.py --domain mips --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --seed 42 --fitness-metric gt --population-type task --reeval none --identify-topk 0 --exec-feedback-n 3 --exec-feedback-prob 0.5 --models best2 --split "$MIPS_EVOLVE_SPLIT" --val-split "" --final-eval-runs 10 --max-samples 1000 --max-evals 1000000 --timeout 500 --pysr-wall-limit 600 --partition default_partition --max-concurrent-jobs 300 --time-limit 00:15:00 --mem-per-cpu 8G --job-timeout 1800 --no-random-target-noise)

# chain_b=$(sbatch --parsable --dependency=afterany:$chain_b -J gt-task run.sh evolve_pysr.py --operator-type all --population-type task --generations 45 --simplify-cooldown 15 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2)
# chain_b=$(sbatch --parsable --dependency=afterany:$chain_b -J boolformer-stratified run.sh evolve_pysr.py --domain boolformer --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models best2 --population-type topk --identify-topk 0 --exec-feedback-n 0 --partition default_partition --max-concurrent-jobs 300 --time-limit 01:00:00 --mem-per-cpu 8G --pysr-wall-limit 2400 --job-timeout 14400 --split splits/boolformer_noisy_stratified_train.txt --val-split splits/boolformer_noisy_stratified_val.txt --val-n-runs 3 --test-split splits/boolformer_noisy_test.txt --extra-test-split splits/pmlb_classification.txt --final-eval-runs 10 --seed 0)




# 8/26 10m max evals for HPO GT 300 trial job
# Disable the inherited 300s soft timeout and use 5x the standard hard walls
# (ground truth: 600s -> 3000s; black box: 1800s -> 9000s).
# sbatch --dependency=afterany:680849 -J hpo300-gt-1e7-srb run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_180547_120309 --ground-truth --black-box --max-evals 10000000 --timeout 0 --pysr-wall-limit 900 --black-box-wall-limit 9000

# 8/25 — Boolformer smoke test (job 610439 completed successfully).
# sbatch -J boolformer-smoke run.sh evolve_pysr.py --domain boolformer --operator-type all --generations 2 --population 3 --offspring 2 --n-runs 1 --fitness-metric gt-acc --reeval none --models cheap --llm-max-workers 2 --population-type topk --identify-topk 0 --exec-feedback-n 0 --partition default_partition --max-concurrent-jobs 10 --time-limit 00:10:00 --mem-per-cpu 8G --pysr-wall-limit 300 --job-timeout 1200 --val-pysr-wall-limit 300 --val-pysr-timeout 240 --split splits/boolformer_noisy_smoke_train.txt --val-split splits/boolformer_noisy_smoke_val.txt --val-n-runs 1 --test-split splits/boolformer_noisy_smoke_test.txt --extra-test-split splits/pmlb_classification_smoke.txt --final-eval-runs 1 --seed 0

# Full noisy Boolformer evolution (REVIEW BEFORE SUBMITTING). Validation is
# monitoring-only; the 100 synthetic targets and PMLB are final-eval-only.
# sbatch -J boolformer-noisy run.sh evolve_pysr.py --domain boolformer --operator-type all --generations 30 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models medium2 --population-type topk --identify-topk 0 --exec-feedback-n 0 --partition default_partition --max-concurrent-jobs 300 --time-limit 01:00:00 --mem-per-cpu 8G --pysr-wall-limit 2400 --job-timeout 14400 --split splits/boolformer_noisy_train.txt --val-split splits/boolformer_noisy_val.txt --val-n-runs 3 --test-split splits/boolformer_noisy_test.txt --extra-test-split splits/pmlb_classification.txt --final-eval-runs 10 --seed 0


# 8/25 — Full 62-task MIPS raw-checkpoint reproduction.
# MIPS_OUTPUT="outputs/mips_reproduction_all"
# /home/sca63/.conda/envs/meta_sr/bin/python scripts/reproduce_mips_all.py prepare --output-dir "$MIPS_OUTPUT"
# mips_array=$(sbatch --parsable --array=0-61%12 --cpus-per-task=1 --mem=8G --time=01:10:00 --partition=default_partition --job-name=mips-all --output="$MIPS_OUTPUT/slurm/%A_%a.out" --error="$MIPS_OUTPUT/slurm/%A_%a.err" run.sh scripts/reproduce_mips_all.py worker --task-index-env --timeout-seconds 3600 --output-dir "$MIPS_OUTPUT")
# mips_aggregate=$(sbatch --parsable --dependency=afterany:"$mips_array" --cpus-per-task=1 --mem=4G --time=00:15:00 --partition=default_partition --job-name=mips-aggregate --output="$MIPS_OUTPUT/slurm/aggregate_%j.out" --error="$MIPS_OUTPUT/slurm/aggregate_%j.err" run.sh scripts/reproduce_mips_all.py aggregate --output-dir "$MIPS_OUTPUT")
# echo "Submitted MIPS array $mips_array and aggregate $mips_aggregate"

# 8/26 — Repaired base PySR on 34 scalar relations: the three reproduced SR
# successes plus the ten deterministic unsolved candidates. All selected rows
# (up to 1,000) are used for search/scoring, then exact hits are checked on the
# uncapped relation. Ten seeds and up to one hour of search per scalar fit.
# Results: outputs/mips_pysr_baseline_1h_full13_seed42/eval_summary.json
# sbatch -J mips-pysr-full13 run.sh evaluate_new_pysr.py --domain mips --fitness-metric gt-acc --splits splits/mips_sr_targets.txt --n-runs 10 --seed 42 --max-samples 1000 --wall-clock-only --timeout 3600 --pysr-wall-limit 3900 --partition default_partition --max-concurrent-jobs 34 --time-limit 01:20:00 --mem-per-cpu 8G --job-timeout 14400 --no-cache --output-dir outputs/mips_pysr_baseline_1h_full13_seed42

# 8/27 — Corrected 13-task rerun with 1e6 evaluations per scalar/seed fit.
# MIPS_13_1M="outputs/mips_pysr_baseline_1e6_full13_seed42"
# mkdir -p "$MIPS_13_1M/slurm"
# mips_13_1m=$(sbatch --parsable --cpus-per-task=1 --mem=8G --time=04:30:00 --partition=default_partition --job-name=mips-13-1m --output="$MIPS_13_1M/slurm/driver_%j.out" --error="$MIPS_13_1M/slurm/driver_%j.err" run.sh evaluate_new_pysr.py --domain mips --fitness-metric gt-acc --splits splits/mips_sr_targets.txt --n-runs 10 --seed 42 --max-samples 1000 --max-evals 1000000 --timeout 3600 --pysr-wall-limit 3900 --partition default_partition --max-concurrent-jobs 34 --time-limit 01:20:00 --mem-per-cpu 8G --job-timeout 14400 --no-cache --output-dir "$MIPS_13_1M")
# mips_13_1m_analysis=$(sbatch --parsable --dependency=afterany:"$mips_13_1m" --cpus-per-task=1 --mem=4G --time=00:10:00 --partition=default_partition --job-name=mips-13-1m-final --output="$MIPS_13_1M/slurm/final_%j.out" --error="$MIPS_13_1M/slurm/final_%j.err" run.sh scripts/analyze_mips_pysr_baseline.py --eval-dir "$MIPS_13_1M/slurm_pysr/eval_0000" --json-output "$MIPS_13_1M/corrected_summary.json" --markdown-output "$MIPS_13_1M/README.md")
# echo "Submitted 13-task 1e6-eval driver $mips_13_1m and analysis $mips_13_1m_analysis"

# 8/26 — Diagnose whether finer MIPS integer lattices remove the observed
# representation conflicts in all 17 affected tasks. The local 800k-row pilot
# took 52s total (29s upstream data/encoding + 22s for eight scaled lattices).
# The full sweep tests both scaled and coarse+residual states from unit through
# float32-mantissa resolution. Each array element has a one-hour hard limit.
# REFINE_OUTPUT="outputs/mips_lattice_refinement"
# mkdir -p "$REFINE_OUTPUT/slurm"
# refine_array=$(sbatch --parsable --array=0-16%17 --cpus-per-task=1 --mem=16G --time=01:00:00 --partition=default_partition --job-name=mips-refine --output="$REFINE_OUTPUT/slurm/%A_%a.out" --error="$REFINE_OUTPUT/slurm/%A_%a.err" run.sh scripts/mips_lattice_refinement.py task --task-index-env --timeout 2700 --output-dir "$REFINE_OUTPUT")
# refine_aggregate=$(sbatch --parsable --dependency=afterany:"$refine_array" --cpus-per-task=1 --mem=4G --time=00:10:00 --partition=default_partition --job-name=mips-ref-sum --output="$REFINE_OUTPUT/slurm/aggregate_%j.out" --error="$REFINE_OUTPUT/slurm/aggregate_%j.err" run.sh scripts/mips_lattice_refinement.py summarize --output-dir "$REFINE_OUTPUT")
# echo "Submitted MIPS refinement array $refine_array and aggregate $refine_aggregate"

# 8/27 — LR-first PySR evaluation on the six compact refined-state candidates.
# Phase 1 builds train/held-out artifacts with one consistent encoder, applies
# the authors' first-1,000-row rounded LR, and creates the LR-failure split.
# Phase 2 runs ten one-hour base-PySR seeds on every LR-unsolved component.
# REFINED_ARTIFACTS="outputs/mips_refined_six_artifacts"
# REFINED_PYSR="outputs/mips_refined_six_pysr_seed42"
# mkdir -p "$REFINED_ARTIFACTS/slurm" "$REFINED_PYSR/slurm"
# refined_build=$(sbatch --parsable --array=0-5%6 --cpus-per-task=1 --mem=12G --time=00:20:00 --partition=default_partition --job-name=mips-ref-build --output="$REFINED_ARTIFACTS/slurm/%A_%a.out" --error="$REFINED_ARTIFACTS/slurm/%A_%a.err" run.sh scripts/mips_refined_sr_artifacts.py build-task --task-index-env --timeout 900 --output-dir "$REFINED_ARTIFACTS")
# refined_summary=$(sbatch --parsable --dependency=afterany:"$refined_build" --cpus-per-task=1 --mem=4G --time=00:10:00 --partition=default_partition --job-name=mips-ref-lr --output="$REFINED_ARTIFACTS/slurm/aggregate_%j.out" --error="$REFINED_ARTIFACTS/slurm/aggregate_%j.err" run.sh scripts/mips_refined_sr_artifacts.py summarize --output-dir "$REFINED_ARTIFACTS")
# refined_train_root="$(pwd)/$REFINED_ARTIFACTS/train"
# refined_pysr=$(sbatch --parsable --dependency=afterok:"$refined_summary" --export=ALL,MIPS_TRANSITION_ROOT="$refined_train_root" --cpus-per-task=1 --mem=8G --time=04:30:00 --partition=default_partition --job-name=mips-ref-pysr --output="$REFINED_PYSR/slurm/driver_%j.out" --error="$REFINED_PYSR/slurm/driver_%j.err" run.sh evaluate_new_pysr.py --domain mips --fitness-metric gt-acc --splits "$REFINED_ARTIFACTS/pysr_components.txt" --n-runs 10 --seed 42 --max-samples 1000 --wall-clock-only --timeout 3600 --pysr-wall-limit 3900 --partition default_partition --max-concurrent-jobs 17 --time-limit 01:20:00 --mem-per-cpu 8G --job-timeout 14400 --no-cache --output-dir "$REFINED_PYSR")
# echo "Submitted refined build $refined_build, LR summary $refined_summary, PySR driver $refined_pysr"

# 8/27 — Corrected refined-state baseline: 1e6 evaluations per scalar/seed fit.
# The one-hour PySR timeout remains only as an emergency guard; max_evals is active.
# REFINED_PYSR_1M="outputs/mips_refined_six_pysr_1e6_seed42"
# mkdir -p "$REFINED_PYSR_1M/slurm"
# refined_pysr_1m=$(sbatch --parsable --export=ALL,MIPS_TRANSITION_ROOT="$(pwd)/outputs/mips_refined_six_artifacts/train" --cpus-per-task=1 --mem=8G --time=04:30:00 --partition=default_partition --job-name=mips-ref-1m --output="$REFINED_PYSR_1M/slurm/driver_%j.out" --error="$REFINED_PYSR_1M/slurm/driver_%j.err" run.sh evaluate_new_pysr.py --domain mips --fitness-metric gt-acc --splits outputs/mips_refined_six_artifacts/pysr_components.txt --n-runs 10 --seed 42 --max-samples 1000 --max-evals 1000000 --timeout 3600 --pysr-wall-limit 3900 --partition default_partition --max-concurrent-jobs 17 --time-limit 01:20:00 --mem-per-cpu 8G --job-timeout 14400 --no-cache --output-dir "$REFINED_PYSR_1M")
# refined_pysr_1m_analysis=$(sbatch --parsable --dependency=afterany:"$refined_pysr_1m" --cpus-per-task=1 --mem=4G --time=00:10:00 --partition=default_partition --job-name=mips-ref-1m-final --output="$REFINED_PYSR_1M/slurm/final_%j.out" --error="$REFINED_PYSR_1M/slurm/final_%j.err" run.sh scripts/analyze_mips_refined_pysr.py --pysr-dir "$REFINED_PYSR_1M")
# echo "Submitted corrected 1e6-eval PySR driver $refined_pysr_1m and analysis $refined_pysr_1m_analysis"


# new simplification
# sbatch --dependency=afterany:593871 -J simp-538-new run.sh evolve_pysr.py --operator-type all --mutation-mode simplify --population-type complexity --generations 20 --population 10 --offspring 10 --n-runs 3 --models medium2 --continue-from runs/538190 --reeval population --n-reevals 10

# srbench full evaluation for the 300-trial HPO selections
# chain_a=$(sbatch --parsable -J srb-hpo300-gtr2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_190637_506162 --ground-truth --black-box --timeout 0)
# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-hpo300-r2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_183759_524347 --ground-truth --black-box --timeout 0)
# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-hpo300-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260824_180547_120309 --ground-truth --black-box --timeout 0)

# python scripts/eval_simplify_candidates.py --run runs/252289

# 8/24
# Counterfactual 300-trial HPO selections. Each driver reuses trials 0-299 and
# the baseline from the corresponding 500-trial run, submits only the 10-way ×
# 10-seed finalist comparison, then automatically runs the standard 10-seed

# final evaluation on barely_unsolvable + val.
# chain_a=$(sbatch --parsable -J hpo300-gt run.sh hpo_pysr.py --reselect-from outputs/hpo_pysr_20260727_172105_644009 --n-trials 300 --n-runs 3 --n-runs-final 10 --final-topk 10 --n-parallel 20 --split splits/barely_unsolvable.txt --val-split splits/val.txt --fitness-metric gt --random-target-noise)
# chain_a=$(sbatch --parsable --dependency=afterany:$chain_a -J hpo300-r2 run.sh hpo_pysr.py --reselect-from outputs/hpo_pysr_20260727_172105_644046 --n-trials 300 --n-runs 3 --n-runs-final 10 --final-topk 10 --n-parallel 20 --split splits/barely_unsolvable.txt --val-split splits/val.txt --fitness-metric r2 --random-target-noise)
# chain_a=$(sbatch --parsable --dependency=afterany:$chain_a -J hpo300-gt-r2 run.sh hpo_pysr.py --reselect-from outputs/hpo_pysr_20260727_172105_644293 --n-trials 300 --n-runs 3 --n-runs-final 10 --final-topk 10 --n-parallel 20 --split splits/barely_unsolvable.txt --val-split splits/val.txt --fitness-metric gt-r2 --random-target-noise)

# sbatch -J neuron-eval-313196 run.sh neuron_full_eval.py --evolve-results runs/313196 --output-dir runs/313196/neuron_full_eval --n-runs 5 --seed 10000 --max-evals 1000000 --max-samples 1024 --partition default_partition --time-limit 00:15:00 --mem-per-cpu 8G --timeout 500 --pysr-wall-limit 600 --job-timeout 1800 --train-split splits/neuron_first1.txt --held-out-world h_sag --held-out-world na_fatigue --held-out-world ca_rebound --held-out-world d_type --held-out-world textbook_M
# sbatch -J boolean-eval-313197 run.sh boolean_eval.py --evolve-results runs/313197 --output-dir runs/313197/boolean_eval --n-runs 10 --seed 10000 --max-evals 1000000 --partition default_partition --max-concurrent-jobs 100 --time-limit 01:00:00 --mem-per-cpu 8G --timeout 1800 --pysr-wall-limit 2400 --job-timeout 14400 --train-split splits/boolean_train.txt

# chain_a=$(sbatch --parsable -J base-1e7-srb run.sh srbench_full_eval.py --black-box --ground-truth --max-evals 10000000)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J gt-1e7-srb run.sh srbench_full_eval.py --black-box --ground-truth --max-evals 10000000 --evolve-results runs/538190)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J hpo-gt-1e7-srb run.sh srbench_full_eval.py --black-box --ground-truth --max-evals 10000000 --hpo-results outputs/hpo_pysr_20260727_172105_644009)

# 8/20
# chain_a=$(sbatch --parsable -J srb-hpo-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644009 --black-box --timeout 0)
# chain_a=$(sbatch --parsable -J neuron-top2 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models cheap2 --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first2.txt --val-split "" --seed 0)
# chain_b=$(sbatch --parsable -J neuron-top1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models cheap2 --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first1.txt --val-split "" --seed 0)
# chain_b=$(sbatch --parsable -J neuron-top1-uninformative run.sh evolve_pysr.py --domain neuron --uninformative-prompts --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models cheap2 --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first1.txt --val-split "" --seed 0)
# chain_c=$(sbatch --parsable -J logic-evolve run.sh evolve_pysr.py --domain boolean --operator-type all --generations 20 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models cheap2 --population-type topk --identify-topk 0 --exec-feedback-n 0 --split splits/boolean_train.txt --val-split "" --seed 0)

# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J r2-new-bb run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric r2 --split splits/bb_train.txt --n-runs 3 --reeval population --n-reevals 10 --models cheap2 --population 10 --offspring 10)
# chain_b=$(sbatch --dependency=afterany:$chain_b --parsable -J r2-new run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric r2 --n-runs 3 --reeval population --n-reevals 10 --models cheap2 --population 10 --offspring 10)
# chain_c=$(sbatch --dependency=afterany:$chain_c --parsable -J r2-gt-new run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric r2-gt --n-runs 3 --reeval population --n-reevals 10 --models cheap2 --population 10 --offspring 10)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J gt-new run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric gt --n-runs 3 --reeval population --n-reevals 10 --models cheap2 --population 10 --offspring 10)

# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J base-1e7-srb run.sh srbench_full_eval.py --black-box --ground-truth --max-evals 10000000)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J gt-1e7-srb run.sh srbench_full_eval.py --black-box --ground-truth --max-evals 10000000 --evolve-results runs/538190)
# chain_a=$(sbatch --dependency=afterany:$chain_a --parsable -J hpo-gt-1e7-srb run.sh srbench_full_eval.py --black-box --ground-truth --max-evals 10000000 --hpo-results outputs/hpo_pysr_20260727_172105_644009)

# 8/19 evening
# sbatch -J logic-evolve run.sh evolve_pysr.py --domain boolean --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models best --population-type topk --identify-topk 0 --exec-feedback-n 0 --split splits/boolean_train.txt --val-split "" --seed 0
# sbatch -J neuron-loocv1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_loocv1.txt --val-split "" --seed 0
# sbatch -J neuron-top1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first1.txt --val-split "" --seed 0
# sbatch -J neuron-top2 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_first2.txt --val-split "" --seed 0
# sbatch --dependency=afterany:290163 -J pysr-base2 run.sh srbench_full_eval.py --ground-truth --black-box --no-cache
# sbatch --dependency=afterany:290211 -J r2-new run.sh evolve_pysr.py --population-type topk --simplify-cooldown 5 --generations 30 --fitness-metric r2 --split splits/bb_train.txt --n-runs 3 --reeval population --n-reevals 10 --models best --population 10 --offspring 10

# 8/19
# sbatch -J simp-538 run.sh evolve_pysr.py --operator-type all --mutation-mode simplify --population-type complexity --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --continue-from runs/538190 --reeval population --n-reevals 10
# srb_prev=$(sbatch --dependency=afterany:252289 --parsable -J neuron-loocv1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_loocv1.txt --val-split "" --seed 0)
# bool_prev=$(sbatch --dependency=afterany:$srb_prev --parsable -J logic-evolve run.sh evolve_pysr.py --domain boolean --operator-type all --generations 15 --simplify-cooldown 5 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models best --population-type topk --identify-topk 0 --exec-feedback-n 0 --split splits/boolean_train.txt --val-split "" --seed 0)

# try doing R2 evolution

# SRBench full evaluation for the completed FullSR run (ground truth + black box).
# sbatch --parsable -J srb-full-150812 run.sh srbench_full_eval.py --evolve-results runs/150812 --ground-truth --black-box
# sbatch --parsable -J srb-full-150812 run.sh srbench_full_eval.py --evolve-results runs/150815 --ground-truth --black-box

# 8/18
# sbatch -J simp-538 run.sh evolve_pysr.py --operator-type all --mutation-mode simplify --population-type complexity --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --continue-from runs/538190 --reeval population --n-reevals 10

# NeuronBench LOOCV: each evolution job trains on five worlds and automatically
# evaluates its final bundle on all six worlds (5 fresh seeds, 1e6 evals).
# srb_prev=$(sbatch --parsable -J neuron-loocv1 run.sh evolve_pysr.py --domain neuron --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt --reeval population --n-reevals 10 --models best --max-evals 1000000 --max-samples 1024 --population-type topk --identify-topk 0 --exec-feedback-n 0 --partition default_partition --max-concurrent-jobs 100 --neuron-full-eval --neuron-eval-runs 5 --neuron-eval-seed 10000 --neuron-eval-max-evals 1000000 --split splits/neuron_loocv1.txt --val-split "" --seed 0)

# Base-PySR control: same all-world 5-seed, 1e6-eval evaluation protocol.
# sbatch --dependency=afterany:"$srb_prev" -J neuron-baseline run.sh neuron_full_eval.py --n-runs 5 --seed 10000 --max-evals 1000000 --max-samples 1024 --partition default_partition --max-concurrent-jobs 100

# LogicBench (Boolean synthesis):
# bool_prev=$(sbatch --dependency=afterany:190177 --parsable -J logic-evolve run.sh evolve_pysr.py --domain boolean --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 3 --fitness-metric gt-acc --reeval population --n-reevals 10 --models best --population-type topk --identify-topk 0 --exec-feedback-n 0 --partition default_partition --max-concurrent-jobs 100 --split splits/boolean_train.txt --val-split "" --seed 0)

# Base-PySR control on the 100 held-out IWLS 2020 problems (3 seeds, 1e6 evals).
# sbatch --dependency=afterany:"$bool_prev" -J logic-baseline run.sh boolean_eval.py --n-runs 3 --seed 10000 --max-evals 1000000 --partition default_partition --max-concurrent-jobs 100

# Two independent chains, so at most two of these run at once. Each job depends
# on the last job of its OWN chain -- chain_a and chain_b never reference each
# other, or they collapse into a single serial chain.
#
# chain_a leads with the three srb-full-* cache top-offs (0 / 104 / 4 runs left,
# so minutes each) to get an HPO grid started early; chain_b carries the two long
# evolve_fullsr runs. The HPO evals are cold 6540-run grids -- --timeout 0 keeps
# them on the same no-soft-timeout protocol as the PySR rows they are compared
# against, and so guarantees full cache misses.
# chain_a=$(sbatch --parsable -J srb-full-225437 run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --black-box)
# chain_b=$(sbatch --parsable -J full-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt)

# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-full-548741 run.sh srbench_full_eval.py --evolve-results runs/548741 --ground-truth --black-box)
# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-full-base run.sh srbench_full_eval.py --fullsr-baseline --ground-truth --black-box)
# chain_b=$(sbatch --parsable --dependency=afterany:"$chain_b" -J full-gt-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric gt-r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt)

# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-hpo-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644009 --ground-truth --black-box --timeout 0)
# chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J srb-hpo-r2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644046 --ground-truth --black-box --timeout 0)
# chain_b=$(sbatch --parsable --dependency=afterany:"$chain_b" -J srb-hpo-gtr2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644293 --ground-truth --black-box --timeout 0)

# 8/17
# evolve_full gt-r2
# sbatch -J full-gt-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric gt-r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt

# srb_prev=$(sbatch --parsable -J srb-full-base run.sh srbench_full_eval.py --fullsr-baseline --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-225437 run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-548741 run.sh srbench_full_eval.py --evolve-results runs/548741 --ground-truth --black-box)


# 7/31 — BasicSR baseline on full SRBench (driver submits chunked arrays).
# Superseded by the 8/17 block above: these ran with no soft timeout.
# srb_prev=$(sbatch --parsable -J srb-full-base run.sh srbench_full_eval.py --fullsr-baseline --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-225437 run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --black-box)

# 7/29

# # Seed the shared cache with successful black-box artifacts from 7/28. Failed
# # trials stay uncached and are the only black-box trials rerun below.
# python scripts/import_srbench_black_box_cache.py \
#     runs/548743 runs/548744 runs/548745 runs/548746

# srb_prev=$(sbatch --parsable -J srb-gtr2 run.sh srbench_full_eval.py --evolve-results runs/120459 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-r2 run.sh srbench_full_eval.py --evolve-results runs/120458 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-gt run.sh srbench_full_eval.py --evolve-results runs/538190 --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-base run.sh srbench_full_eval.py --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-gt run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644009 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-r2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644046 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-gtr2 run.sh srbench_full_eval.py --hpo-results outputs/hpo_pysr_20260727_172105_644293 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-225437 run.sh srbench_full_eval.py --evolve-results runs/225437 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-full-base run.sh srbench_full_eval.py --fullsr-baseline --ground-truth --black-box)

# 1e7 gt evals
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-gt-1e7 run.sh srbench_full_eval.py --evolve-results runs/538190 --ground-truth --max-evals 10000000)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-base-1e7 run.sh srbench_full_eval.py --ground-truth --max-evals 10000000)

# simplify
# sbatch -J simp-538 run.sh evolve_pysr.py --operator-type all --mutation-mode simplify --population-type complexity --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --continue-from runs/538190 --reeval population --n-reevals 10

# 7/28/36

# sbatch -J full-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt
# sbatch -J full-gt-r2 run.sh evolve_fullsr.py --generations 30 --offspring 10 --population 10 --n-runs 3 --models best --fitness-metric gt-r2 --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt

# srb_prev=$(sbatch --parsable -J srb-gtr2 run.sh srbench_full_eval.py --evolve-results runs/120459 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-r2 run.sh srbench_full_eval.py --evolve-results runs/120458 --black-box --ground-truth)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-gt run.sh srbench_full_eval.py --evolve-results runs/538190 --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-base run.sh srbench_full_eval.py --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-gt run.sh srbench_full_eval.py --evolve-results runs/422103 --ground-truth --black-box)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-r2 run.sh srbench_full_eval.py --evolve-results runs/422104 --black-box --ground-truth)
# srb_prev=$(sbatch --parsable --dependency=afterany:"$srb_prev" -J srb-hpo-gtr2 run.sh srbench_full_eval.py --evolve-results runs/422105 --black-box --ground-truth)

# sbatch -J eval-121270 --partition=default_partition run.sh evaluate_new_pysr.py --splits splits/val.txt splits/barely_unsolvable_val2.txt --n-runs 10 --evolve-results runs/121270
# sbatch -J eval-155134 --partition=default_partition run.sh evaluate_new_pysr.py --splits splits/val.txt splits/barely_unsolvable_val2.txt --n-runs 10 --evolve-results runs/155134
# sbatch -J eval-120458 --partition=default_partition run.sh evaluate_new_pysr.py --splits splits/val.txt splits/barely_unsolvable_val2.txt --n-runs 10 --evolve-results runs/120458
# sbatch -J r2-train run.sh evolve_pysr.py --split splits/train.txt --val-split splits/val.txt --generations 30 --models best --offspring 10 --population 10 --n-runs 3 --fitness-metric r2

# set -euo pipefail

# 7/27/26 resubmitting jobs that failed from 7/23
# sbatch -J hpo-gt run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric gt --n-parallel 20 --random-target-noise --models best
# sbatch -J hpo-r2 run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric r2 --n-parallel 20 --random-target-noise --models best
# sbatch -J hpo-gt-r2 run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric gt-r2 --n-parallel 20 --random-target-noise --models best

# 7/23/26 — queued experiment suite. Three independent dependency chains keep
# at most three driver jobs running at once. Independent jobs use `afterany` so
# a chain continues after an earlier failure. Follow-ups that consume a newly
# created run use `afterok`, because they require the producer's run_data.json.
# Run this file to submit; these commands have not been submitted yet.

# # Chain 1, job 1: evaluate base PySR on the 122 SRBench black-box (BB) tasks.
# chain1=$(sbatch --parsable -J bb-pysr-base run.sh srbench_full_eval.py --black-box)
# # Chain 2, job 1: run 500-trial PySR HPO on GT with three runs per trial.
# chain2=$(sbatch --parsable -J hpo-gt run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric gt --n-parallel 20)
# # Chain 3, job 1: run 500-trial PySR HPO on R² with three runs per trial.
# chain3=$(sbatch --parsable -J hpo-r2 run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric r2 --n-parallel 20)

# # Chain 1, job 2: evaluate GT-R² PySR++ (run 120459) on BB.
# chain1=$(sbatch --parsable --dependency=afterany:$chain1 -J bb-pysrpp-gtr2 run.sh srbench_full_eval.py --evolve-results runs/120459 --black-box)
# # Chain 2, job 2: evaluate R² PySR++ (run 120458) on BB.
# chain2=$(sbatch --parsable --dependency=afterany:$chain2 -J bb-pysrpp-r2 run.sh srbench_full_eval.py --evolve-results runs/120458 --black-box)
# # Chain 3, job 2: run 500-trial PySR HPO on GT-R² with three runs per trial.
# chain3=$(sbatch --parsable --dependency=afterany:$chain3 -J hpo-gt-r2 run.sh hpo_pysr.py --n-trials 500 --n-runs 3 --split splits/barely_unsolvable.txt --fitness-metric gt-r2 --n-parallel 20)

# # Chain 1, job 3: evolve FullSR for R² (cheap models, 15 generations).
# fullsr_r2=$(sbatch --parsable --dependency=afterany:$chain1 -J fullsr-r2 run.sh evolve_fullsr.py --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap --fitness-metric r2 --split splits/train.txt --val-split splits/val.txt)
# # Chain 2, job 3: evaluate GT-R² PySR++ (run 120459) on all GT tasks.
# chain2=$(sbatch --parsable --dependency=afterany:$chain2 -J gt-pysrpp-gtr2 run.sh srbench_full_eval.py --evolve-results runs/120459 --ground-truth)
# # Chain 3, job 3: evolve all PySR++ operators on LogicBench (cheap models, 15 generations).
# chain3=$(sbatch --parsable --dependency=afterany:$chain3 -J logic-pysrpp run.sh evolve_pysr.py --domain boolean --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap)

# # Chain 1, job 4: evaluate the newly evolved R² FullSR run on BB.
# chain1=$(sbatch --parsable --dependency=afterok:$fullsr_r2 -J bb-fullsr-r2 run.sh srbench_full_eval.py --evolve-results runs/$fullsr_r2 --black-box)
# # Chain 2, job 4: evolve FullSR for GT-R² (cheap models, 15 generations).
# fullsr_gtr2=$(sbatch --parsable --dependency=afterany:$chain2 -J fullsr-gt-r2 run.sh evolve_fullsr.py --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap --fitness-metric gt-r2 --split splits/train.txt --val-split splits/val.txt)

# # Chain 2, job 5: evaluate the newly evolved GT-R² FullSR run on BB.
# chain2=$(sbatch --parsable --dependency=afterok:$fullsr_gtr2 -J bb-fullsr-gtr2 run.sh srbench_full_eval.py --evolve-results runs/$fullsr_gtr2 --black-box)

# echo "Submitted three chains; tails: chain1=$chain1 chain2=$chain2 chain3=$chain3"

# 7/23/26
# sbatch -J fulleval --partition default_partition run.sh srbench_full_eval.py --evolve-results runs/538190
# sbatch run.sh evolve_fullsr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt --random-target-noise
# sbatch -J simplify run.sh evolve_pysr.py --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --mutation-mode simplify --population-type complexity --continue-from runs/538190

# 7/22/26 - minimalSR evolution
# sbatch run.sh evolve_fullsr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt

# sbatch run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt
# 7/21/26 — n1 vs n3 vs reeval modes (best models, 10 offspring/gen, 30 gens).
# Oracle-replay follow-up under the NEW reeval CLI (--reeval + --reeval-budget):
#   n1  = 10 evals/gen, no reeval
#   n3  = 30 evals/gen, no reeval
#   un1 = n1 offspring + '--reeval uniform' B=20 (even split over top-10)
#         = 30/gen, budget-matched to n3 (the literal oracle-replay winner)
#   un3 = n3 offspring + '--reeval uniform' B=40 = 70/gen
# 3 jobs run at once: seed-0s start immediately; each seed-1 chains afterany its
# own seed-0; the un3 pair chains after the first six wind down.
# To launch: run `bash submit_jobs.sh` (this block is the only uncommented one).
# COMMON="--operator-type all --generations 30 --population 10 --offspring 10 --models best --random-target-noise"
# N1="--n-runs 1 --reeval none"
# N3="--n-runs 3 --reeval none"
# UN1="--n-runs 1 --reeval uniform --reeval-budget 20"
# UN3="--n-runs 3 --reeval uniform --reeval-budget 40"
# # seed 0 (3 concurrent)
# jn1s0=$(sbatch --parsable -J n1s0      run.sh evolve_pysr.py $COMMON $N1  --seed 0)
# jn3s0=$(sbatch --parsable -J n3s0      run.sh evolve_pysr.py $COMMON $N3  --seed 0)
# jun1s0=$(sbatch --parsable -J u-n1s0 run.sh evolve_pysr.py $COMMON $UN1 --seed 0)
# # seed 1 (each after its own seed 0)
# jn1s1=$(sbatch --parsable --dependency=afterany:$jn1s0 -J n1s1   run.sh evolve_pysr.py $COMMON $N1  --seed 1)
# jn3s1=$(sbatch --parsable --dependency=afterany:$jn3s0 -J n3s1   run.sh evolve_pysr.py $COMMON $N3  --seed 1)
# jun1s1=$(sbatch --parsable --dependency=afterany:$jun1s0 -J u-n1s1 run.sh evolve_pysr.py $COMMON $UN1 --seed 1)
# # n3 + uniform-B follow-up pair (after the first six wind down)
# jun3s0=$(sbatch --parsable --dependency=afterany:$jn3s1 -J u-n3s0 run.sh evolve_pysr.py $COMMON $UN3 --seed 0)
# jun3s1=$(sbatch --parsable --dependency=afterany:$jun1s1 -J u-n3s1 run.sh evolve_pysr.py $COMMON $UN3 --seed 1)

# echo "Submitted 8 jobs: n1=($jn1s0,$jn1s1) n3=($jn3s0,$jn3s1) un1=($jun1s0,$jun1s1) un3=($jun3s0,$jun3s1)"


# 7/16/26

# n1 vs v3, continue to 30 generations
# cheap ensemble, --random-target-noise). Per-generation eval budget B matched per pair:
#   cond1 none  n1o20 (20 evals/gen)  vs  cond3 smart n1o5  B=20   ← chain A (odd jobs)
#   cond2 none  n3o20 (60 evals/gen)  vs  cond4 smart n3o5  B=60   ← chain B (even jobs)

# At most 2 jobs run at once: jobs 1 & 2 start immediately; job N depends on job N-2
# (afterany), forming two parallel chains. Jobs are ordered seed-major (all 4 conditions
# at seed s, then s+1), which makes chain A = the B=20 pair and chain B = the B=60 pair.
# To launch: uncomment this whole block (it must run together — the $jidN vars chain
# the dependencies), then run `bash submit_jobs.sh`.
# COMMON="--operator-type all --generations 30 --population 10 --models cheap --random-target-noise"
# C1="--offspring 20 --n-runs 1 --reeval none"
# C2="--offspring 20 --n-runs 3 --reeval none"
# #
# # seed 0
# jid1=$(sbatch --parsable                       -J nn_n1o20_s0   run.sh evolve_pysr.py $COMMON $C1 --seed 0 --continue-from runs/89281)
# jid2=$(sbatch --parsable                       -J nn_n3o20_s0   run.sh evolve_pysr.py $COMMON $C2 --seed 0 --continue-from runs/89282)
# # seed 1
# jid5=$(sbatch --parsable --dependency=afterany:$jid1 -J nn_n1o20_s1   run.sh evolve_pysr.py $COMMON $C1 --seed 1 --continue-from runs/825769)
# jid6=$(sbatch --parsable --dependency=afterany:$jid2 -J nn_n3o20_s1   run.sh evolve_pysr.py $COMMON $C2 --seed 1 --continue-from runs/825770)
# # seed 2
# jid9=$(sbatch  --parsable --dependency=afterany:$jid5 -J nn_n1o20_s2   run.sh evolve_pysr.py $COMMON $C1 --seed 2 --continue-from runs/825773)
# jid10=$(sbatch --parsable --dependency=afterany:$jid6 -J nn_n3o20_s2   run.sh evolve_pysr.py $COMMON $C2 --seed 2 --continue-from runs/825774)
# # seed 3
# jid13=$(sbatch --parsable --dependency=afterany:$jid9 -J nn_n1o20_s3   run.sh evolve_pysr.py $COMMON $C1 --seed 3 --continue-from runs/825777)
# jid14=$(sbatch --parsable --dependency=afterany:$jid10 -J nn_n3o20_s3   run.sh evolve_pysr.py $COMMON $C2 --seed 3 --continue-from runs/825778)
# # seed 4
# jid17=$(sbatch --parsable --dependency=afterany:$jid13 -J nn_n1o20_s4   run.sh evolve_pysr.py $COMMON $C1 --seed 4 --continue-from runs/825781)
# jid18=$(sbatch --parsable --dependency=afterany:$jid14 -J nn_n3o20_s4   run.sh evolve_pysr.py $COMMON $C2 --seed 4 --continue-from runs/825782)
# echo "Submitted 10 jobs"

# 7/1/26
# COMMON="--generations 20 --population 10 --models best --random-target-noise"
# C1="--offspring 10 --n-runs 10 --reeval none"
# sbatch -J bn10o10s0 run.sh evolve_pysr.py $COMMON $C1 --seed 0
# sbatch -J bn10o10s1 run.sh evolve_pysr.py $COMMON $C1 --seed 1

# sbatch -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt
# sbatch --dependency=afterany:568245 -J fullsr-full-file --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 15 --population 10 --offspring 10 --n-runs 3 --models cheap --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt --full-file-diff
# sbatch --dependency=afterany:568246 -J max-time run.sh evolve_pysr.py --generations 15 --population 10 --n-runs 3 --max-time-in-seconds 120 --models best --random-target-noise --reeval smart
# sbatch --dependency=afterany:689916 -J gt-r2-train run.sh evolve_pysr.py --generations 15 --population 10 --n-runs 3 --models cheap --reeval smart --random-target-noise --fitness-metric gt-r2 --split splits/train.txt --val-split splits/val.txt

# C2="--offspring 20 --n-runs 3 --reeval none"
# C3="--offspring 5 --n-runs 1 --reeval smart --max-runs-per-generation 20"
# C4="--offspring 5 --n-runs 3 --reeval smart --max-runs-per-generation 60"
# #
# # seed 0
# jid1=$(sbatch --parsable                       -J nn_n1o20_s0   run.sh evolve_pysr.py $COMMON $C1 --seed 0)
# jid2=$(sbatch --parsable                       -J nn_n3o20_s0   run.sh evolve_pysr.py $COMMON $C2 --seed 0)
# jid3=$(sbatch --parsable --dependency=afterany:$jid1 -J sm_n1o5b20_s0 run.sh evolve_pysr.py $COMMON $C3 --seed 0)
# jid4=$(sbatch --parsable --dependency=afterany:$jid2 -J sm_n3o5b60_s0 run.sh evolve_pysr.py $COMMON $C4 --seed 0)
# 6/30/26
# sbatch -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt
# sbatch --dependency=afterany:397152 -J max-time run.sh evolve_pysr.py --generations 30 --population 10 --n-runs 3 --max-time-in-seconds 120 --models best --random-target-noise --reeval smart
# sbatch --dependency=afterany:492224 -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models best --split splits/barely_unsolvable.txt --val-split splits/barely_unsolvable_val2.txt --full-file-diff
# sbatch --dependency=afterany:492225 -J r2-train run.sh evolve_pysr.py --generations 15 --population 10 --n-runs 3 --models best --reeval smart --random-target-noise --fitness-metric r2 --split splits/train.txt --val-split splits/val.txt

# 6/29/26 — planet eval of the gt-r2 bundle (runs/120459).
# sbatch -J r2-gt-planet planet_eval.sh --evolve-results ~/meta_sr/runs/120459

# 6/29/26 — smart-reeval comparison on the BEST model ensemble.
# COMMON="--operator-type all --generations 30 --population 10 --n-runs 1 --models best --random-target-noise"
# NONE="--offspring 20 --reeval none"
# KG="--offspring 5 --reeval smart-KG --max-runs-per-generation 20"
# TT="--offspring 5 --reeval smart-TTTS --max-runs-per-generation 20"
# sbatch --parsable  -J re_ttts_s0 run.sh evolve_pysr.py $COMMON $TT   --seed 0
# sbatch --parsable  -J re_ttts_s1 run.sh evolve_pysr.py $COMMON $TT   --seed 1
# # # seed 0 (chain A)
# # a1=$(sbatch --parsable                            -J re_none_s0 run.sh evolve_pysr.py $COMMON $NONE --seed 0)
# # a2=$(sbatch --parsable                            -J re_kg_s0   run.sh evolve_pysr.py $COMMON $KG   --seed 0)
# # # seed 1 (chain B)
# # b1=$(sbatch --parsable                            -J re_none_s1 run.sh evolve_pysr.py $COMMON $NONE --seed 1)
# # b2=$(sbatch --parsable                            -J re_kg_s1   run.sh evolve_pysr.py $COMMON $KG   --seed 1)


# all noise evolution
# sbatch --dependency=afterany:$c1 -J all-noise --partition ellis run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --reeval none --models best --eval-all-noise-levels

# new fullSR evolution
# sbatch -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models cheap --split splits/train.txt


# 6/26/26
# evolve R^2 or R^2 combined with GT
# sbatch -J r2 run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 5 --n-runs 3 --models best --reeval smart --max-runs-per-generation 60 --random-target-noise --fitness-metric r2
# sbatch -J gt-r2 run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 5 --n-runs 3 --models best --reeval smart --max-runs-per-generation 60 --random-target-noise --fitness-metric gt-r2
# sbatch --dependency=afterany:120458 -J train run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 5 --n-runs 3 --models best --reeval smart --max-runs-per-generation 60 --random-target-noise --split splits/train.txt --val-split splits/val.txt

# 6/26/26 — resubmit the two corrupted seed-0 "no reeval" runs. The originals
# (825765 nn_n1o20_s0 and 825766 nn_n3o20_s0, the first two jobs submitted) died
# at gen 2: code on disk was edited mid-run, so their long-running driver hit
# `PySRTaskResult.__init__() got an unexpected keyword argument 'r2_frontier_score'`
# and every offspring scored -1 thereafter (frozen population). Chained after each
# pair's seed-4 job (825783 = chain A / n1 last, 825784 = chain B / n3 last) so we
# keep at most 2 running. After they finish, point the plot GROUPS at the new ids.
# SUBMITTED 6/26: nn_n1o20_s0 -> job 89281 (after 825783); nn_n3o20_s0 -> job 89282 (after 825784).
# COMMON="--operator-type all --generations 15 --population 10 --models cheap --random-target-noise"
# reA=$(sbatch --parsable --dependency=afterany:825783 -J nn_n1o20_s0_re run.sh evolve_pysr.py $COMMON --offspring 20 --n-runs 1 --reeval none --seed 0)
# reB=$(sbatch --parsable --dependency=afterany:825784 -J nn_n3o20_s0_re run.sh evolve_pysr.py $COMMON --offspring 20 --n-runs 3 --reeval none --seed 0)
# echo "resubmitted: nn_n1o20_s0 -> $reA (after 825783),  nn_n3o20_s0 -> $reB (after 825784)"

# 6/24/26 — smart vs nonsmart budget-matched comparison (5 seeds each, 15 gens,
# cheap ensemble, --random-target-noise). Per-generation eval budget B matched per pair:
#   cond1 none  n1o20 (20 evals/gen)  vs  cond3 smart n1o5  B=20   ← chain A (odd jobs)
#   cond2 none  n3o20 (60 evals/gen)  vs  cond4 smart n3o5  B=60   ← chain B (even jobs)
# At most 2 jobs run at once: jobs 1 & 2 start immediately; job N depends on job N-2
# (afterany), forming two parallel chains. Jobs are ordered seed-major (all 4 conditions
# at seed s, then s+1), which makes chain A = the B=20 pair and chain B = the B=60 pair.
# To launch: uncomment this whole block (it must run together — the $jidN vars chain
# the dependencies), then run `bash submit_jobs.sh`.
# COMMON="--operator-type all --generations 15 --population 10 --models cheap --random-target-noise"
# C1="--offspring 20 --n-runs 1 --reeval none"
# C2="--offspring 20 --n-runs 3 --reeval none"
# C3="--offspring 5 --n-runs 1 --reeval smart --max-runs-per-generation 20"
# C4="--offspring 5 --n-runs 3 --reeval smart --max-runs-per-generation 60"
# #
# # seed 0
# jid1=$(sbatch --parsable                       -J nn_n1o20_s0   run.sh evolve_pysr.py $COMMON $C1 --seed 0)
# jid2=$(sbatch --parsable                       -J nn_n3o20_s0   run.sh evolve_pysr.py $COMMON $C2 --seed 0)
# jid3=$(sbatch --parsable --dependency=afterany:$jid1 -J sm_n1o5b20_s0 run.sh evolve_pysr.py $COMMON $C3 --seed 0)
# jid4=$(sbatch --parsable --dependency=afterany:$jid2 -J sm_n3o5b60_s0 run.sh evolve_pysr.py $COMMON $C4 --seed 0)
# # seed 1
# jid5=$(sbatch --parsable --dependency=afterany:$jid3 -J nn_n1o20_s1   run.sh evolve_pysr.py $COMMON $C1 --seed 1)
# jid6=$(sbatch --parsable --dependency=afterany:$jid4 -J nn_n3o20_s1   run.sh evolve_pysr.py $COMMON $C2 --seed 1)
# jid7=$(sbatch --parsable --dependency=afterany:$jid5 -J sm_n1o5b20_s1 run.sh evolve_pysr.py $COMMON $C3 --seed 1)
# jid8=$(sbatch --parsable --dependency=afterany:$jid6 -J sm_n3o5b60_s1 run.sh evolve_pysr.py $COMMON $C4 --seed 1)
# # seed 2
# jid9=$(sbatch  --parsable --dependency=afterany:$jid7 -J nn_n1o20_s2   run.sh evolve_pysr.py $COMMON $C1 --seed 2)
# jid10=$(sbatch --parsable --dependency=afterany:$jid8 -J nn_n3o20_s2   run.sh evolve_pysr.py $COMMON $C2 --seed 2)
# jid11=$(sbatch --parsable --dependency=afterany:$jid9 -J sm_n1o5b20_s2 run.sh evolve_pysr.py $COMMON $C3 --seed 2)
# jid12=$(sbatch --parsable --dependency=afterany:$jid10 -J sm_n3o5b60_s2 run.sh evolve_pysr.py $COMMON $C4 --seed 2)
# # seed 3
# jid13=$(sbatch --parsable --dependency=afterany:$jid11 -J nn_n1o20_s3   run.sh evolve_pysr.py $COMMON $C1 --seed 3)
# jid14=$(sbatch --parsable --dependency=afterany:$jid12 -J nn_n3o20_s3   run.sh evolve_pysr.py $COMMON $C2 --seed 3)
# jid15=$(sbatch --parsable --dependency=afterany:$jid13 -J sm_n1o5b20_s3 run.sh evolve_pysr.py $COMMON $C3 --seed 3)
# jid16=$(sbatch --parsable --dependency=afterany:$jid14 -J sm_n3o5b60_s3 run.sh evolve_pysr.py $COMMON $C4 --seed 3)
# # seed 4
# jid17=$(sbatch --parsable --dependency=afterany:$jid15 -J nn_n1o20_s4   run.sh evolve_pysr.py $COMMON $C1 --seed 4)
# jid18=$(sbatch --parsable --dependency=afterany:$jid16 -J nn_n3o20_s4   run.sh evolve_pysr.py $COMMON $C2 --seed 4)
# jid19=$(sbatch --parsable --dependency=afterany:$jid17 -J sm_n1o5b20_s4 run.sh evolve_pysr.py $COMMON $C3 --seed 4)
# jid20=$(sbatch --parsable --dependency=afterany:$jid18 -J sm_n3o5b60_s4 run.sh evolve_pysr.py $COMMON $C4 --seed 4)
# echo "Submitted 20 jobs"

# 6/24/26 — fullsr evolution sanity check (does evolving improve over baseline?)
# sbatch -J fullsr --partition ellis run.sh evolve_fullsr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --models cheap --split splits/train.txt

# 6/23/26

# sbatch -J train-split-n3o10smart-best run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models best --reeval smart --max-runs-per-generation 200 --random-target-noise --split splits/train.txt

# sbatch -J test_full --partition default_partition run.sh evolve_fullsr.py --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 3 --models best --split splits/train.txt
# sbatch -J fulleval --partition default_partition run.sh srbench_full_eval.py --evolve-results runs/538190
# sbatch -J planetseval2 --partition default_partition planet_eval.sh --evolve-results runs/538190
# sbatch -J planetsbaseline2 --partition default_partition planet_eval.sh --baseline

# ##### SMART, N3O10 OR N1O10, S0 OR S1
# jid1=$(sbatch --parsable -J n3o10s0smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --models best --reeval smart --max-reruns 100 --random-target-noise --seed 0)
# jid2=$(sbatch --parsable -J n3o10s1smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --models best --reeval smart --max-reruns 100 --random-target-noise --seed 1)
# jid3=$(sbatch --parsable -J n1o10s0smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --models cheap --reeval smart --max-reruns 100 --random-target-noise --seed 0)
# jid4=$(sbatch --parsable -J n1o10s1smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --models cheap --reeval smart --max-reruns 100 --random-target-noise --seed 1)

# ##### NONSMART, N3O10 OR N1O10, S0 OR S1
# sbatch -J n3o10s0 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --models best --reeval none --random-target-noise --seed 0
# sbatch -J n3o10s1 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --models best --reeval none --random-target-noise --seed 1
# sbatch --dependency=afterany:$jid3 -J n1o10s0 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --models cheap --reeval none --random-target-noise --seed 0
# sbatch --dependency=afterany:$jid4 -J n1o10s1 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --models cheap --reeval none --random-target-noise --seed 1

# 6/18
# sbatch -J n3o10smart-cheap run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models best --reeval smart --max-reruns 100 --random-target-noise
# sbatch -J n3o10smart-best run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models cheap --reeval smart --max-reruns 100 --random-target-noise

# 6/17 planet_eqs eval (planet_eval.py submits its own 1-node / 32-core Slurm job)
# sbatch -J planets_baseline planet_eval.sh planet_eval.py --baseline --time-in-hours 8
# sbatch -J planets40318 planet_eval.sh planet_eval.py --evolve-results ~/meta_sr/runs/40318

# sbatch -J fulleval_666285 run.sh srbench_full_eval.py --evolve-results runs/666285 --max-evals 1000000

# 6/4
# sbatch -J n2o10 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n2o10smart run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch --dependency=afterany:40319 -J n1o10smart_cont run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100 --continue-from runs/40319

# 6/3
# sbatch -J n1o10_cont run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --continue-from runs/4901
# sbatch -J n2o14 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n1o30smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 30 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch -J n1o10smart run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100


# 6/1

# sbatch -J n1o10_cont run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --continue-from runs/4901
# sbatch --dependency=afterany:4903 -J n1o10smart_cont run.sh evolve_pysr.py --operator-type all --generations 30 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --continue-from runs/4903
# sbatch -J n2o14 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium

# sbatch -J n1smartre run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch -J n2o14 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 14 --n-runs 2 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n1o10 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n1o30smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 30 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch -J n1o10smart run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 1 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100
# sbatch -J n4o7 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 7 --n-runs 4 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n5o18 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 18 --n-runs 5 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium


# 5/28
# sbatch -J smart-reeval run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium --reeval smart --max-reruns 100


# 5/26
# sbatch -J n3o30 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 30 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n3o9 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 9 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n1o30 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 30 --n-runs 3 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J n10o9 run.sh evolve_pysr.py --operator-type all --generations 20 --population 10 --offspring 9 --n-runs 10 --max-evals 1000000 --population-type topk --exec-feedback-n 3 --models medium
# sbatch -J topk_race run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --exec-feedback-n 3 --population-type topk --max-evals 1000000


# 5/14
# sbatch -J topk run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --population-type topk --exec-feedback-n 3
# p evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --population-type topk --exec-feedback-n 3
# sbatch -J topk_race run.sh evolve_pysr.py --operator-type all --generations 25 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --exec-feedback-n 3 --population-type topk

# 5/8
# sbatch -J full run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --population-type task --exec-feedback-n 3
# sbatch -J race run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --max-evals 1000000 --exec-feedback-n 3 --continue-from runs/982249
# sbatch -J loss run.sh evolve_pysr.py --operator-type loss --generations 5 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --population-type task --exec-feedback-n 3

# 5/6
# Simplify-only run: seed initial pop with the best bundle from 399313, then
# only ever apply the simplify meta-mutation (init pop + every generation's
# offspring). 10 gens, 10 offspring/gen, 10 seeds/eval.
# sbatch -J simplify run.sh evolve_pysr.py --operator-type all --generations 10 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --baseline runs/399313 --mutation-mode simplify --population-type complexity --continue-from runs/666285
# sbatch -J race-train run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --lambda-target 5 --max-evals 1000000 --exec-feedback-n 3 --continue-from runs/666286 --split splits/train.txt --val-split splits/val.txt

# 5/6
# sbatch -J race2 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --n-extra-runs 5 --n-runs-max 25 --lambda-target 5 --max-evals 1000000 --exec-feedback-n 3
# sbatch -J race4 run.sh evolve_pysr.py --operator-type all --generations 40 --population 10 --offspring 10 --n-runs 6 --n-extra-runs 10 --n-runs-max 50 --max-evals 1000000 --exec-feedback-n 3 --lambda-target 2 --continue-from runs/666286


# 5/5


# 5/4
# sbatch -J loss run.sh evolve_pysr.py --operator_type loss --generations 5 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --task_diverse_pop --exec_feedback_n 3
# sbatch -J full run.sh evolve_pysr.py --operator_type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --task_diverse_pop --exec_feedback_n 3
# sbatch -J racing --dependency=afterany:399304 run.sh evolve_pysr.py --operator_type all --generations 30 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --racing --exec_feedback_n 3
# sbatch -J train run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max-evals 1000000 --task-diverse-pop --exec-feedback-n 3 --split splits/train.txt --val-split splits/val.txt
# sbatch -J train3 run.sh evolve_pysr.py --operator-type all --generations 50 --population 10 --offspring 10 --n-runs 3 --max-evals 1000000 --task-diverse-pop --exec-feedback-n 3 --split splits/train.txt --val-split splits/val.txt


# sbatch -J test run.sh evolve_pysr.py --operator_type loss --generations 2 --population 3 --offspring 3 --n-runs 1 --max_evals 10000 --task_diverse_pop --exec_feedback_n 3

# sbatch -J loss run.sh evolve_pysr.py --operator_type loss --generations 10 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --task_diverse_pop --exec_feedback_n 3

# 4/30
# sbatch -J smart2 --mem 20G run.sh evolve_pysr.py --operator_type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --task_diverse_pop --exec_feedback_n 3 --continue_from runs/947961
# sbatch -J smart_no_task --mem 20G run.sh evolve_pysr.py --operator_type all --generations 25 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --exec_feedback_n 3

# SPLITS="splits/barely_unsolvable_val2.txt"
# SEED=1000
# BUNDLE_RESULTS=runs/947961/run_data.json
# JID1A=$(sbatch --parsable -J eval_947961_val2  --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)
# JID1B=$(sbatch --parsable -J eval_baseline_val2 --mem 20G run.sh evaluate_new_pysr.py                                  --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)

# 4/29 — Final eval of 947961 bundle vs PySR baseline.
# Splits: train + val + barely_unsolvable (60 datasets).
# 10 seeds, fresh seed base (1000) so we don't reuse cached evolve-run results.
# 4 budget conditions x 2 methods (947961 bundle, baseline) = 8 invocations.
# Conditions run sequentially via --dependency=afterany; the two methods within
# each condition run in parallel.

# BUNDLE_RESULTS=runs/947961/run_data.json
# SPLITS="splits/train.txt splits/val.txt splits/barely_unsolvable.txt"
# SEED=1000

# # (1) 1e6 max_evals — matches evolve-run training budget.
# JID1A=$(sbatch --parsable -J eval_947961_1e6  --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)
# JID1B=$(sbatch --parsable -J eval_baseline_1e6 --mem 20G run.sh evaluate_new_pysr.py                                  --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)

# # (2) Wall-clock only (no max_evals; PySR stops on timeout_in_seconds=300s).
# JID2A=$(sbatch --parsable --dependency=afterany:$JID1A:$JID1B -J eval_947961_wc   --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --wall-clock-only --timeout 300 --pysr-wall-limit 600 --time-limit 01:00:00)
# JID2B=$(sbatch --parsable --dependency=afterany:$JID1A:$JID1B -J eval_baseline_wc --mem 20G run.sh evaluate_new_pysr.py                                  --splits $SPLITS --seed $SEED --n-runs 10 --wall-clock-only --timeout 300 --pysr-wall-limit 600 --time-limit 01:00:00)

# # (3) 1e7 max_evals — 10x training budget.
# JID3A=$(sbatch --parsable --dependency=afterany:$JID2A:$JID2B -J eval_947961_1e7  --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 10000000 --timeout 3600 --pysr-wall-limit 4200 --time-limit 06:00:00)
# JID3B=$(sbatch --parsable --dependency=afterany:$JID2A:$JID2B -J eval_baseline_1e7 --mem 20G run.sh evaluate_new_pysr.py                                 --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 10000000 --timeout 3600 --pysr-wall-limit 4200 --time-limit 06:00:00)

# # (4) 1e6 max_evals + Gaussian target noise (SRBench standard 0.001).
# JID4A=$(sbatch --parsable --dependency=afterany:$JID3A:$JID3B -J eval_947961_noise  --mem 20G run.sh evaluate_new_pysr.py --evolve-results $BUNDLE_RESULTS --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --noise 0.001 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)
# JID4B=$(sbatch --parsable --dependency=afterany:$JID3A:$JID3B -J eval_baseline_noise --mem 20G run.sh evaluate_new_pysr.py                                 --splits $SPLITS --seed $SEED --n-runs 10 --max-evals 1000000 --noise 0.001 --timeout 600 --pysr-wall-limit 900 --time-limit 02:00:00)

# echo "Submitted: (1) $JID1A,$JID1B  (2) $JID2A,$JID2B  (3) $JID3A,$JID3B  (4) $JID4A,$JID4B"


# 4/29 — Bundle HPO of the 947961 best bundle (combined base PySR hparams +
# LLM-extracted operator hparams, max 2/operator). 500 trials, 3 runs/trial,
# up to 20 trials in parallel.
# sbatch -J hpo_947961 --mem 20G run.sh hpo_pysr.py --baseline runs/947961 --n-trials 100 --n-parallel 20 --n-runs 10 --max-op-hparams 2 --split splits/barely_unsolvable.txt --max-evals 1000000 --time-limit 02:00:00


# 4/24
# sbatch -J smart2 --mem 20G run.sh evolve_pysr.py --operator_type all --generations 50 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3 --models best --continue_from runs/947961
# sbatch -J smart_no_task --mem 20G run.sh evolve_pysr.py --operator_type all --generations 25 --population 10 --offspring 10 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --exec_feedback_n 3 --models best


# 4/23
# sbatch -J smart_test --mem 40G run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3 --models best --hp_tuning_trials 3 --hpo-n-runs 3


# 4/22
# sbatch -J task --mem 40G run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3 --hp_tuning_trials 3 --hpo-n-runs 3 --models best

# 4/21
# MiniSR vs. PySR baseline comparison on full SRBench (130 datasets × 3 seeds × 2 engines).
# Driver runs under run.sh on a login/compute node and submits its own sub-arrays via SLURM.
# sbatch -J minisr_vs_pysr run.sh compare_minisr_vs_pysr.py --split splits/srbench_all.txt --n-runs 3 --max-evals 1000000
# sbatch -J task --mem 40G run.sh evolve_pysr.py --operator_type all --generations 5 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3

# 4/20
# sbatch -J exec run.sh evolve_pysr.py --operator_type all --generations 5 --population 5 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --exec_feedback_n 3
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 30 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing --exec_feedback_n 3
# sbatch -J task --mem 40G run.sh evolve_pysr.py --operator_type all --generations 5 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --exec_feedback_n 3
# sbatch -J exec run.sh evolve_pysr.py --operator_type all --generations 10 --population 5 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt --exec_feedback_n 3
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 60 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing --exec_feedback_n 3


# 4/18
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 60 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing --hof
# sbatch -J cont_racing run.sh evolve_pysr.py --operator_type all --generations 60 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing --continue_from runs/499255

# 4/16/26
# sbatch -J big run.sh evolve_pysr.py --operator_type all --generations 10 --population 5 --offspring 20 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 60 --population 5 --offspring 5 --n-runs 2 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing
# sbatch -J task --mem 40G run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --task_aware


# 4/14/26
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 30 --population 5 --offspring 5 --n-runs 1 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing
# sbatch -J task run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 10 --n-runs 5 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --task_aware
# sbatch -J topk_10 run.sh evolve_pysr.py --operator_type all --generations 1 --population 40 --offspring 40 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt

# new hpo run
# sbatch -J hpo5 run.sh hpo_pysr.py --n-trials 100 --n-parallel 2 --n-runs 5
# sbatch -J hpo5_bu run.sh hpo_pysr.py --n-trials 100 --n-parallel 2 --split splits/barely_unsolvable.txt --n-runs 5

# top-k run with --n-runs 10
# sbatch -J topk_10 run.sh evolve_pysr.py --operator_type all --generations 1 --population 40 --offspring 40 --n-runs 10 --max_evals 1000000 --split splits/barely_unsolvable.txt
# sbatch -J racing run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 10 --n-runs 1 --max_evals 1000000 --split splits/barely_unsolvable.txt --racing
# sbatch -J task run.sh evolve_pysr.py --operator_type all --generations 10 --population 10 --offspring 10 --n-runs 1 --max_evals 1000000 --split splits/barely_unsolvable.txt --task_diverse_pop --task_aware

# 4/10/26 — task-aware mutation/crossover smoke test (3 generations)
# sbatch run.sh evolve_pysr.py --operator_type mutation --generations 3 --task_aware --task_aware_prob 0.5 --task_diverse_pop
# /home/sca63/meta_sr/outputs/openevolve_pysr_selection_20260407_210230
# 4/9/26 — Task-diverse population experiment (selection, with vs without)
# sbatch run.sh evolve_pysr.py --operator_type selection --baseline outputs/hpo_pysr_20260407_152534 --generations 50 --task_diverse_pop
# sbatch run.sh evolve_pysr.py --operator_type selection --baseline outputs/hpo_pysr_20260407_152534 --generations 50

# 4/9/26 — Final evaluations (10 seeds, train+val)

# # 199034: openevolve selection (0.43)
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_selection_20260408_213220 \
#     --n-runs 10

# # 199036: evolve selection (0.43)
# sbatch run.sh evaluate_new_pysr.py \
#     --evolve-results outputs/evolve_selection_20260408_213226/run_data.json \
#     --n-runs 10

# # 172094: evolve selection (0.43) — out file overwritten, found in outputs/
# sbatch run.sh evaluate_new_pysr.py \
#     --evolve-results outputs/evolve_selection_20260408_144423/run_data.json \
#     --n-runs 10

# # 199033: evolve mutation+selection bundle (0.43)
# sbatch run.sh evaluate_new_pysr.py \
#     --evolve-results outputs/evolve_mutation+survival+selection_20260408_213219/run_data.json \
#     --n-runs 10

# # 199017: hpo 500 trials
# sbatch run.sh evaluate_new_pysr.py \
#     --best-weights outputs/hpo_pysr_20260408_212941/best_params.json \
#     --n-runs 10

# # 171895: evolve mutation+selection bundle (0.43)
# sbatch run.sh evaluate_new_pysr.py \
#     --evolve-results outputs/evolve_mutation+survival+selection_20260408_143835/run_data.json \
#     --n-runs 10

# # 172092: openevolve selection (0.45)
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_selection_20260408_144423 \
#     --n-runs 10

# # 151084: openevolve mutation
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_mutation_20260408_000544 \
#     --n-runs 10

# # 144408: openevolve selection (0.48!)
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_selection_20260407_210230 \
#     --n-runs 10

# # 136931: openevolve selection (0.45)
# sbatch run.sh evaluate_new_pysr.py \
#     --openevolve-results outputs/openevolve_pysr_selection_20260407_155505 \
#     --n-runs 10

# # 136931 baseline (no operator — logs baseline to wandb)
# sbatch run.sh evaluate_new_pysr.py \
#     --n-runs 10

# # 4/9/26
# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type bundle \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh hpo_pysr.py --n-trials 10
# sbatch run.sh evolve_pysr.py --operator_type all --baseline outputs/hpo_pysr_20260407_152534 --generations 3

# 4/8
# sbatch run.sh hpo_pysr.py --n-trials 500
# sbatch run.sh evolve_pysr.py --operator_type all --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type selection \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type bundle \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh evolve_pysr.py --operator_type selection --baseline outputs/hpo_pysr_20260407_152534 --generations 50

# sbatch run.sh run_pysr_srbench.py --dataset feynman_III_15_27 --time_minutes 1
# sbatch run_pysr.sh
# sbatch run_meta_sr.sh
# sbatch run_sr.sh
# sbatch run_meta_sr.sh --no-trace-feedback
# sbatch run_meta_sr.sh

# for target_noise in 0.001 0.01 0.1; do
#   for max_samples in 100000000; do
    # sbatch --time=10:00:00 run_pysr.sh --results_dir results_pysr_${target_noise}_${max_samples} --max_evals ${max_samples} --target_noise ${target_noise}
#   done
# done

# sbatch --time=10:00:00 run_pysr.sh --results_dir results_pysr_1e6 --max_evals 1000000

# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e3 --max_evals 1000 --target_noise 0.001
# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e4 --max_evals 10000
# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e5 --max_evals 100000
# sbatch --time=01:00:00 run_pysr.sh --results_dir results_pysr_1e6 --max_evals 1000000
# sbatch --time=04:00:00 run_pysr.sh --results_dir results_pysr_1e7 --max_evals 10000000
# sbatch --time=08:00:00 run_pysr.sh --results_dir results_pysr_1e8 --max_evals 100000000
# python evolve_pysr.py --operator_type mutation --generations 2 --n-runs 3
# sbatch run.sh hpo_pysr.py --n-trials 500

# sbatch run.sh evolve_pysr.py --operator_type all --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh evolve_pysr.py --operator_type mutation --fitness_metric gt --split splits/train_small.txt --n-runs 1 --max_evals 100000
# sbatch run.sh evolve_basic_sr.py --split splits/train.txt
# # sbatch run.sh evolve_basic_sr.py
# sbatch run.sh evolve_pysr.py --operator_type selection --fitness_metric gt --split splits/train_small.txt
# sbatch run.sh evolve_pysr.py --operator_type survival --fitness_metric gt --split splits/train.txt
# sbatch run.sh hpo_pysr.py --n-trials 500

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type selection \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type bundle \
#     --iterations 200 \
#     --baseline outputs/hpo_pysr_20260407_152534

# sbatch run.sh evolve_pysr.py --operator_type selection --baseline outputs/hpo_pysr_20260407_152534 --generations 50

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type survival \
#     --iterations 200

# sbatch run.sh run_openevolve_pysr.py \
#     --operator-type mutation \
#     --iterations 200


# sbatch run.sh evolve_pysr.py --operator_type survival --fitness_metric gt --split splits/train.txt
