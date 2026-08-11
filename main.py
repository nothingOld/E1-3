import json
import time


EPSILON = 1e-9
REPEAT_COUNT = 1000
SUPPORTED_JSON_SIZES = (5, 13, 25)
PERFORMANCE_SIZES = (3, 5, 13, 25)

LABEL_MAP = {
    "+": "Cross",
    "cross": "Cross",
    "x": "X",
}

DATA_FILE_PATH = "data.json"


def load_data(file_path: str) -> dict:
    """Loads JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise ValueError(
            f"파일을 찾을 수 없습니다: {file_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            "JSON 파일 형식이 올바르지 않습니다."
        ) from error


def normalize_label(label: str) -> str:
    """Normalizes an expected label to the internal standard label."""
    normalized_label = label.strip().lower()
    return LABEL_MAP.get(normalized_label, label)


def validate_data(data: dict) -> None:
    """Validates the top-level JSON structure."""
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 데이터는 객체여야 합니다.")

    if "filters" not in data:
        raise ValueError("JSON에 'filters' 키가 없습니다.")

    if "patterns" not in data:
        raise ValueError("JSON에 'patterns' 키가 없습니다.")

    if not isinstance(data["filters"], dict):
        raise ValueError("'filters'는 객체여야 합니다.")

    if not isinstance(data["patterns"], dict):
        raise ValueError("'patterns'는 객체여야 합니다.")


def validate_matrix(matrix: object, size: int, name: str) -> None:
    """Validates matrix size and numeric values."""
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError(
            f"{name} 오류: {size}개의 행이 필요합니다."
        )

    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(
                f"{name} 오류: 각 행에 {size}개의 값이 필요합니다."
            )

        for value in row:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise ValueError(
                    f"{name} 오류: 행렬에는 숫자만 사용할 수 있습니다."
                )


def parse_pattern_size(pattern_name: str) -> int:
    """Extracts the matrix size from a pattern key."""
    name_parts = pattern_name.split("_")

    if len(name_parts) != 3 or name_parts[0] != "size":
        raise ValueError(
            f"지원하지 않는 패턴 이름 형식입니다: {pattern_name}"
        )

    try:
        size = int(name_parts[1])
    except ValueError as error:
        raise ValueError(
            f"지원하지 않는 패턴 이름 형식입니다: {pattern_name}"
        ) from error

    if size <= 0:
        raise ValueError(
            f"패턴 크기는 1 이상이어야 합니다: {pattern_name}"
        )

    return size


def get_filters_for_size(
    filters: dict,
    size: int,
) -> tuple:
    """Gets and validates Cross and X filters for a matrix size."""
    size_key = f"size_{size}"

    if size_key not in filters:
        raise ValueError(
            f"'{size_key}' 필터가 없습니다."
        )

    filter_data = filters[size_key]

    if not isinstance(filter_data, dict):
        raise ValueError(
            f"'{size_key}' 필터 형식이 올바르지 않습니다."
        )

    standard_filters = {}

    for filter_name, filter_matrix in filter_data.items():
        if not isinstance(filter_name, str):
            continue

        normalized_name = filter_name.strip().lower()

        if normalized_name == "cross":
            standard_filters["Cross"] = filter_matrix
        elif normalized_name == "x":
            standard_filters["X"] = filter_matrix

    if "Cross" not in standard_filters or "X" not in standard_filters:
        raise ValueError(
            f"'{size_key}'에 'cross'와 'x' 필터가 필요합니다."
        )

    cross_filter = standard_filters["Cross"]
    x_filter = standard_filters["X"]

    validate_matrix(
        cross_filter,
        size,
        f"{size_key} Cross 필터",
    )
    validate_matrix(
        x_filter,
        size,
        f"{size_key} X 필터",
    )

    return cross_filter, x_filter


def validate_pattern_case(
    pattern_name: str,
    pattern_data: object,
    filters: dict,
) -> tuple:
    """Validates one JSON pattern case and returns normalized data."""
    if not isinstance(pattern_data, dict):
        raise ValueError(
            f"'{pattern_name}' 패턴 형식이 올바르지 않습니다."
        )

    if "input" not in pattern_data:
        raise ValueError(
            f"'{pattern_name}'에 'input' 키가 없습니다."
        )

    if "expected" not in pattern_data:
        raise ValueError(
            f"'{pattern_name}'에 'expected' 키가 없습니다."
        )

    size = parse_pattern_size(pattern_name)
    pattern = pattern_data["input"]

    validate_matrix(
        pattern,
        size,
        f"{pattern_name} 패턴",
    )

    expected_value = pattern_data["expected"]

    if not isinstance(expected_value, str):
        raise ValueError(
            f"'{pattern_name}'의 'expected'는 문자열이어야 합니다."
        )

    expected = normalize_label(expected_value)

    if expected not in ("Cross", "X"):
        raise ValueError(
            f"지원하지 않는 예상 라벨입니다: {expected_value}"
        )

    cross_filter, x_filter = get_filters_for_size(
        filters,
        size,
    )

    return (
        size,
        pattern,
        cross_filter,
        x_filter,
        expected,
    )


def read_matrix(name: str, size: int) -> list:
    """Reads a square matrix from user input."""
    matrix = []

    print(f"{name}을 입력하세요.")

    while len(matrix) < size:
        row_number = len(matrix) + 1
        user_input = input(f"{row_number}행: ")

        try:
            row = [
                float(value)
                for value in user_input.split()
            ]
        except ValueError:
            print(
                f"입력 형식 오류: 각 줄에 {size}개의 숫자를 "
                "공백으로 구분해 입력하세요."
            )
            continue

        if len(row) != size:
            print(
                f"입력 형식 오류: 각 줄에 {size}개의 숫자를 "
                "공백으로 구분해 입력하세요."
            )
            continue

        matrix.append(row)

    return matrix


def mac(
    pattern: list,
    filter_matrix: list,
) -> float:
    """Calculates the MAC score of a pattern and filter."""
    total = 0.0

    for row_index in range(len(pattern)):
        for column_index in range(len(pattern[row_index])):
            total += (
                pattern[row_index][column_index]
                * filter_matrix[row_index][column_index]
            )

    return total


def flatten_matrix(matrix: list) -> list:
    """Flattens a 2D matrix into a 1D list."""
    flattened = []

    for row in matrix:
        flattened.extend(row)

    return flattened


def mac_1d(
    pattern: list,
    filter_values: list,
) -> float:
    """Calculates the MAC score of flattened 1D data."""
    if len(pattern) != len(filter_values):
        raise ValueError(
            "1D 패턴과 필터의 길이가 일치하지 않습니다."
        )

    total = 0.0

    for index in range(len(pattern)):
        total += pattern[index] * filter_values[index]

    return total


def generate_cross(size: int) -> list:
    """Generates an N x N Cross pattern."""
    if size <= 0:
        raise ValueError("패턴 크기는 1 이상이어야 합니다.")

    matrix = [
        [0.0 for _ in range(size)]
        for _ in range(size)
    ]
    center = size // 2

    for index in range(size):
        matrix[center][index] = 1.0
        matrix[index][center] = 1.0

    return matrix


def generate_x(size: int) -> list:
    """Generates an N x N X pattern."""
    if size <= 0:
        raise ValueError("패턴 크기는 1 이상이어야 합니다.")

    matrix = [
        [0.0 for _ in range(size)]
        for _ in range(size)
    ]

    for index in range(size):
        matrix[index][index] = 1.0
        matrix[index][size - index - 1] = 1.0

    return matrix


def measure_mac_time(
    pattern: list,
    filter_matrix: list,
    repeat: int = REPEAT_COUNT,
) -> float:
    """Measures average 2D MAC execution time in milliseconds."""
    start_time = time.perf_counter()

    for _ in range(repeat):
        mac(pattern, filter_matrix)

    end_time = time.perf_counter()

    return ((end_time - start_time) / repeat) * 1000


def measure_mac_1d_time(
    pattern: list,
    filter_values: list,
    repeat: int = REPEAT_COUNT,
) -> float:
    """Measures average 1D MAC execution time in milliseconds."""
    start_time = time.perf_counter()

    for _ in range(repeat):
        mac_1d(pattern, filter_values)

    end_time = time.perf_counter()

    return ((end_time - start_time) / repeat) * 1000


def classify_scores(
    score_cross: float,
    score_x: float,
) -> str:
    """Classifies Cross/X scores using epsilon comparison."""
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"

    if score_cross > score_x:
        return "Cross"

    return "X"


def classify_ab(
    score_a: float,
    score_b: float,
) -> str:
    """Classifies user-entered filter A/B scores."""
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"

    if score_a > score_b:
        return "A"

    return "B"


def print_matrix(matrix: list) -> None:
    """Prints a generated matrix."""
    for row in matrix:
        print(
            " ".join(
                str(int(value))
                if value.is_integer()
                else str(value)
                for value in row
            )
        )


def print_performance_table(
    performance_cases: list,
) -> None:
    """Prints average MAC time and N-squared operation count."""
    print(
        f"\n=== 성능 분석 (평균/{REPEAT_COUNT}회) ==="
    )
    print(
        f"{'크기':<10}"
        f"{'평균 시간(ms)':>16}"
        f"{'연산 횟수(N²)':>16}"
    )
    print("-" * 42)

    for size, pattern, filter_matrix in performance_cases:
        average_time = measure_mac_time(
            pattern,
            filter_matrix,
        )

        print(
            f"{size}x{size:<7}"
            f"{average_time:>16.6f}"
            f"{size ** 2:>16}"
        )


def build_default_performance_cases(
) -> list:
    """Builds reusable 3x3, 5x5, 13x13, and 25x25 cases."""
    performance_cases = []

    for size in PERFORMANCE_SIZES:
        cross_pattern = generate_cross(size)
        performance_cases.append(
            (
                size,
                cross_pattern,
                cross_pattern,
            )
        )

    return performance_cases


def print_optimization_comparison() -> None:
    """Compares 2D and flattened 1D MAC performance."""
    print(
        f"\n=== 보너스 1: 2D/1D MAC 성능 비교 "
        f"(평균/{REPEAT_COUNT}회) ==="
    )
    print(
        f"{'크기':<10}"
        f"{'2D(ms)':>14}"
        f"{'1D(ms)':>14}"
        f"{'점수 일치':>12}"
    )
    print("-" * 50)

    for size in PERFORMANCE_SIZES:
        pattern_2d = generate_cross(size)
        filter_2d = generate_cross(size)

        pattern_1d = flatten_matrix(pattern_2d)
        filter_1d = flatten_matrix(filter_2d)

        score_2d = mac(pattern_2d, filter_2d)
        score_1d = mac_1d(pattern_1d, filter_1d)

        time_2d = measure_mac_time(
            pattern_2d,
            filter_2d,
        )
        time_1d = measure_mac_1d_time(
            pattern_1d,
            filter_1d,
        )

        scores_match = (
            abs(score_2d - score_1d) < EPSILON
        )

        print(
            f"{size}x{size:<7}"
            f"{time_2d:>14.6f}"
            f"{time_1d:>14.6f}"
            f"{str(scores_match):>12}"
        )


def print_filter_load_status(
    filters: dict,
) -> None:
    """Prints validation status for required JSON filters."""
    print("\n=== 필터 로드 ===")

    for size in SUPPORTED_JSON_SIZES:
        try:
            get_filters_for_size(filters, size)
        except ValueError as error:
            print(
                f"✗ size_{size} 필터 오류: {error}"
            )
        else:
            print(
                f"✓ size_{size} 필터 로드 완료 (Cross, X)"
            )


def build_failure_reason(
    prediction: str,
    expected: str,
) -> str:
    """Builds a failure reason for a valid analyzed case."""
    if prediction == "UNDECIDED":
        return (
            "동점(UNDECIDED) 처리 규칙에 따라 "
            f"expected {expected}와 불일치"
        )

    return (
        f"판정 {prediction}이 expected {expected}와 불일치"
    )


def print_result_summary(
    results: list,
) -> None:
    """Prints total/pass/fail counts and failure reasons."""
    total_count = len(results)
    pass_count = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )
    failed_results = [
        result
        for result in results
        if result["status"] == "FAIL"
    ]

    print("\n=== 결과 요약 ===")
    print(f"전체 테스트 수: {total_count}")
    print(f"통과 수: {pass_count}")
    print(f"실패 수: {len(failed_results)}")

    if not failed_results:
        print("실패 케이스가 없습니다.")
        return

    print("\n실패 케이스:")

    for result in failed_results:
        print(
            f"- {result['case_id']}: "
            f"{result['reason']}"
        )


def run_user_input_mode(
    generated_patterns=None,
) -> None:
    """Runs the simulator with user-entered 3x3 filters and pattern."""
    size = 3

    print("\n=== 사용자 입력 모드 ===")

    filter_a = read_matrix("필터 A", size)
    print("✓ 필터 A 저장 완료")

    filter_b = read_matrix("필터 B", size)
    print("✓ 필터 B 저장 완료")

    if (
        generated_patterns is not None
        and generated_patterns["size"] == size
    ):
        while True:
            print("\n패턴 입력 방법을 선택하세요.")
            print("1. 생성된 Cross 패턴")
            print("2. 생성된 X 패턴")
            print("3. 직접 입력")

            pattern_choice = input("선택: ").strip()

            if pattern_choice == "1":
                pattern = generated_patterns["Cross"]
                print("✓ 생성된 Cross 패턴을 사용")
                break

            if pattern_choice == "2":
                pattern = generated_patterns["X"]
                print("✓ 생성된 X 패턴을 사용")
                break

            if pattern_choice == "3":
                pattern = read_matrix("패턴", size)
                print("✓ 패턴 저장 완료")
                break

            print("입력 오류: 1, 2, 3 중 하나를 입력하세요.")

    else:
        if generated_patterns is not None:
            generated_size = generated_patterns["size"]

            print(
                f"현재 생성된 패턴은 "
                f"{generated_size}x{generated_size}이므로 "
                "3x3 사용자 입력 모드에서 사용할 수 없습니다."
            )

        pattern = read_matrix("패턴", size)
        print("✓ 패턴 저장 완료")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)

    prediction = classify_ab(
        score_a,
        score_b,
    )

    print("\n=== MAC 결과 ===")
    print(f"필터 A MAC 점수: {score_a}")
    print(f"필터 B MAC 점수: {score_b}")

    if prediction == "UNDECIDED":
        print(
            f"판정 결과: 판정 불가 "
            f"(|A-B| < {EPSILON})"
        )
    else:
        print(f"판정 결과: {prediction}")

    print_performance_table(
        [(3, pattern, filter_a)]
    )


def run_json_mode() -> None:
    """Runs the simulator with data loaded from data.json."""
    data = load_data(DATA_FILE_PATH)
    validate_data(data)

    filters = data["filters"]
    patterns = data["patterns"]

    print_filter_load_status(filters)

    print("\n=== 패턴 분석 ===")

    results = []

    for pattern_name, pattern_data in patterns.items():
        try:
            (
                size,
                pattern,
                cross_filter,
                x_filter,
                expected,
            ) = validate_pattern_case(
                pattern_name,
                pattern_data,
                filters,
            )

            score_cross = mac(
                pattern,
                cross_filter,
            )
            score_x = mac(
                pattern,
                x_filter,
            )
            prediction = classify_scores(
                score_cross,
                score_x,
            )

            print(f"\n--- {pattern_name} ---")
            print(f"크기: {size}x{size}")
            print(f"Cross 점수: {score_cross}")
            print(f"X 점수: {score_x}")
            print(f"판정: {prediction}")
            print(f"expected: {expected}")

            if prediction == expected:
                print("결과: PASS")
                results.append(
                    {
                        "case_id": pattern_name,
                        "status": "PASS",
                        "reason": "",
                    }
                )
            else:
                reason = build_failure_reason(
                    prediction,
                    expected,
                )
                print(f"결과: FAIL ({reason})")
                results.append(
                    {
                        "case_id": pattern_name,
                        "status": "FAIL",
                        "reason": reason,
                    }
                )

        except ValueError as error:
            reason = str(error)

            print(f"\n--- {pattern_name} ---")
            print(f"결과: FAIL ({reason})")

            results.append(
                {
                    "case_id": pattern_name,
                    "status": "FAIL",
                    "reason": reason,
                }
            )

    print_performance_table(build_default_performance_cases())
    print_optimization_comparison()
    print_result_summary(results)


def run_pattern_generator_mode():
    """Generates and returns N x N Cross and X patterns."""
    print("\n=== 보너스 2: 패턴 생성기 ===")

    while True:
        user_input = input(
            "생성할 패턴 크기 N을 입력하세요: "
        ).strip()

        try:
            size = int(user_input)
        except ValueError:
            print(
                "입력 오류: 패턴 크기는 정수로 입력하세요."
            )
            continue

        if size <= 0:
            print(
                "입력 오류: 패턴 크기는 1 이상이어야 합니다."
            )
            continue

        break

    cross_pattern = generate_cross(size)
    x_pattern = generate_x(size)

    print(f"\n{size}x{size} Cross 패턴")
    print_matrix(cross_pattern)

    print(f"\n{size}x{size} X 패턴")
    print_matrix(x_pattern)

    cross_time = measure_mac_time(
        cross_pattern,
        cross_pattern,
    )
    x_time = measure_mac_time(
        x_pattern,
        x_pattern,
    )

    print("\n=== 생성 패턴 성능 분석 ===")
    print(
        f"Cross {size}x{size}: "
        f"{cross_time:.6f} ms "
        f"(연산 횟수: {size ** 2})"
    )
    print(
        f"X {size}x{size}: "
        f"{x_time:.6f} ms "
        f"(연산 횟수: {size ** 2})"
    )

    return {
        "size": size,
        "Cross": cross_pattern,
        "X": x_pattern,
    }


def main() -> None:
    """Runs the Mini NPU simulator."""
    generated_patterns = None

    while True:
        try:
            print("\n=== Mini NPU Simulator ===")
            print()
            print("1. 사용자 입력 모드 (3x3)")
            print("2. data.json 분석")
            print("3. 패턴 생성기 (보너스 2)")
            print("0. 종료")

            choice = input("모드를 선택하세요: ").strip()

            if choice == "1":
                run_user_input_mode(generated_patterns)
            elif choice == "2":
                run_json_mode()
            elif choice == "3":
                generated_patterns = run_pattern_generator_mode()
            elif choice == "0":
                print("프로그램을 종료합니다.")
                return
            else:
                print("올바른 모드를 선택하세요.")

        except ValueError as error:
            print(f"오류: {error}")
        except (KeyboardInterrupt, EOFError):
            print("\n프로그램을 종료합니다.")
            return


if __name__ == "__main__":
    main()
