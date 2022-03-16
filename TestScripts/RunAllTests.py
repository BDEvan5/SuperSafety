from cProfile import run
from KernelSafetyTests import generate_kernels, run_random_test
from RewardTests import run_reward_tests
from BenchmarkTests import pure_pursuit_tests, follow_the_gap_tests, benchmark_sss_tests, benchmark_baseline_tests
from RepeatabilityTests import run_repeatability
from ResultBuilder import run_builder


def run_all_tests():
    generate_kernels()
    run_random_test(1)
    # run_reward_tests()
    pure_pursuit_tests(1)
    follow_the_gap_tests(1)
    benchmark_sss_tests(1)
    benchmark_baseline_tests(1)
    run_repeatability()
    run_builder()



run_all_tests()