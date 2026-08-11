import json
import time

EPSILON = 1e-9  # 1e-9 = 0.000000001

LABEL_MAP = {
    "+": "Cross",
    "cross": "Cross",
    "x": "X",
    "undecided": "UNDECIDED",
}


def load_data(file_path: str) -> dict:
    """Loads JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise ValueError(f"파일을 찾을 수 없습니다: {file_path}")
    except json.JSONDecodeError:
        raise ValueError("JSON 파일 형식이 올바르지 않습니다.")


def normalize_label(label: str) -> str:
    """Normalizes a classification label."""
    normalized_label = label.strip().lower()
    return LABEL_MAP.get(normalized_label, label)


def validate_matrix(matrix: list, size: int, name: str) -> None:
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
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} 오류: 행렬에는 숫자만 사용할 수 있습니다."
                )


def validate_data(data: dict) -> None:
    """Validates JSON data structure and filter values."""
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 데이터는 객체여야 합니다.")

    if "filters" not in data:
        raise ValueError("JSON에 'filters' 키가 없습니다.")

    if "patterns" not in data:
        raise ValueError("JSON에 'patterns' 키가 없습니다.")

    filters = data["filters"]
    patterns = data["patterns"]

    if not isinstance(filters, dict):
        raise ValueError("'filters'는 객체여야 합니다.")

    if not isinstance(patterns, dict):
        raise ValueError("'patterns'는 객체여야 합니다.")

    for size_name, filter_data in filters.items():
        if not isinstance(filter_data, dict):
            raise ValueError(
                f"'{size_name}' 필터 형식이 올바르지 않습니다."
            )

        if "cross" not in filter_data or "x" not in filter_data:
            raise ValueError(
                f"'{size_name}'에 'cross'와 'x' 필터가 필요합니다."
            )

        size_parts = size_name.split("_")

        if len(size_parts) != 2 or size_parts[0] != "size":
            raise ValueError(
                f"지원하지 않는 필터 크기 형식입니다: {size_name}"
            )

        try:
            size = int(size_parts[1])
        except ValueError:
            raise ValueError(
                f"지원하지 않는 필터 크기 형식입니다: {size_name}"
            )

        if size <= 0:
            raise ValueError(
                f"필터 크기는 1 이상이어야 합니다: {size_name}"
            )

        validate_matrix(
            filter_data["cross"],
            size,
            f"{size_name} Cross 필터",
        )

        validate_matrix(
            filter_data["x"],
            size,
            f"{size_name} X 필터",
        )


def validate_pattern(
    pattern_name: str,
    pattern_data: dict,
    filters: dict,
) -> str:
    """Validates a pattern and returns its filter size key."""
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

    name_parts = pattern_name.split("_")

    if len(name_parts) != 3 or name_parts[0] != "size":
        raise ValueError(
            f"지원하지 않는 패턴 이름 형식입니다: {pattern_name}"
        )

    try:
        size = int(name_parts[1])
    except ValueError:
        raise ValueError(
            f"지원하지 않는 패턴 이름 형식입니다: {pattern_name}"
        )

    if size <= 0:
        raise ValueError(
            f"패턴 크기는 1 이상이어야 합니다: {pattern_name}"
        )

    size_key = f"size_{size}"

    if size_key not in filters:
        raise ValueError(
            f"'{pattern_name}'에 사용할 '{size_key}' 필터가 없습니다."
        )

    validate_matrix(
        pattern_data["input"],
        size,
        f"{pattern_name} 패턴",
    )

    expected_value = pattern_data["expected"]

    if not isinstance(expected_value, str):
        raise ValueError(
            f"'{pattern_name}'의 'expected'는 문자열이어야 합니다."
        )

    expected = normalize_label(expected_value)

    if expected not in ("Cross", "X", "UNDECIDED"):
        raise ValueError(
            f"지원하지 않는 예상 라벨입니다: {expected_value}"
        )

    return size_key


def read_matrix(name: str, size: int) -> list[list[float]]:
    """Reads a square matrix from user input."""
    matrix = []

    print(f"{name}을 입력하세요.")

    while len(matrix) < size:
        row_number = len(matrix) + 1
        user_input = input(f"{row_number}행: ")

        try:
            row = [float(value) for value in user_input.split()]
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
    pattern: list[list[float]],
    filter_matrix: list[list[float]],
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


def flatten_matrix(matrix: list[list[float]]) -> list[float]:
    """Flattens a 2D matrix into a 1D list."""
    flattened = []

    for row in matrix:
        flattened.extend(row)

    return flattened


def mac_1d(
    pattern: list[float],
    filter_values: list[float],
) -> float:
    """Calculates the MAC score of flattened 1D data."""
    total = 0.0

    for index in range(len(pattern)):
        total += pattern[index] * filter_values[index]

    return total


def sliding_window_classify(
    input_matrix: list[list[float]],
    cross_filter: list[list[float]],
    x_filter: list[list[float]],
) -> list[tuple[int, int, float, float, str]]:
    """Classifies each window of an input matrix using MAC scores."""
    if not input_matrix or not input_matrix[0]:
        raise ValueError("Sliding Window 입력 행렬이 비어 있습니다.")

    filter_size = len(cross_filter)

    if filter_size == 0:
        raise ValueError("Sliding Window 필터가 비어 있습니다.")

    if len(x_filter) != filter_size:
        raise ValueError("Cross 필터와 X 필터의 크기가 다릅니다.")

    input_rows = len(input_matrix)
    input_columns = len(input_matrix[0])

    for row in input_matrix:
        if len(row) != input_columns:
            raise ValueError(
                "Sliding Window 입력 행렬의 열 크기가 일정하지 않습니다."
            )

    if input_rows < filter_size or input_columns < filter_size:
        raise ValueError(
            "입력 행렬은 필터보다 크거나 같아야 합니다."
        )

    results = []

    row_count = input_rows - filter_size + 1
    column_count = input_columns - filter_size + 1

    for start_row in range(row_count):
        for start_column in range(column_count):
            window = []

            for row_index in range(
                start_row,
                start_row + filter_size,
            ):
                row = input_matrix[row_index][
                    start_column:start_column + filter_size
                ]
                window.append(row)

            score_cross = mac(window, cross_filter)
            score_x = mac(window, x_filter)

            prediction = classify_scores(
                score_cross,
                score_x,
            )

            results.append(
                (
                    start_row,
                    start_column,
                    score_cross,
                    score_x,
                    prediction,
                )
            )

    return results


def measure_mac_time(
    pattern: list[list[float]],
    filter_matrix: list[list[float]],
    repeat: int = 1000,
) -> float:
    """Measures the average MAC execution time."""
    start_time = time.perf_counter()

    for _ in range(repeat):
        mac(pattern, filter_matrix)

    end_time = time.perf_counter()

    return (end_time - start_time) / repeat


def classify_scores(score_cross: float, score_x: float) -> str:
    """Classifies a pattern based on two MAC scores."""
    # 두 점수의 차이를 비교하므로 절댓값을 사용한다.
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"

    if score_cross > score_x:
        return "Cross"

    return "X"


def run_user_input_mode() -> None:
    """Runs the simulator with user-entered 3x3 matrices."""
    size = 3

    filter_a = read_matrix("필터 A (Cross)", size)
    filter_b = read_matrix("필터 B (X)", size)
    pattern = read_matrix("패턴", size)

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)

    prediction = classify_scores(score_a, score_b)

    time_a = measure_mac_time(pattern, filter_a)
    time_b = measure_mac_time(pattern, filter_b)
    average_time = (time_a + time_b) / 2

    print(f"필터 A MAC 점수: {score_a}")
    print(f"필터 B MAC 점수: {score_b}")
    print(f"판정 결과: {prediction}")
    print(f"평균 MAC 실행 시간: {average_time:.9f}초")


def run_json_mode() -> None:
    """Runs the simulator with data loaded from a JSON file."""
    data = load_data("data.json")
    validate_data(data)

    filters = data["filters"]
    patterns = data["patterns"]

    pass_count = 0
    fail_count = 0
    execution_times = {}

    for pattern_name, pattern_data in patterns.items():
        try:
            size_key = validate_pattern(
                pattern_name,
                pattern_data,
                filters,
            )
        except ValueError as error:
            print(f"\n[{pattern_name}]")
            print(f"오류: {error}")
            print("결과: FAIL")
            fail_count += 1
            continue

        size = int(size_key.split("_")[1])

        filter_data = filters[size_key]

        cross_filter = filter_data["cross"]
        x_filter = filter_data["x"]

        pattern = pattern_data["input"]
        expected = normalize_label(pattern_data["expected"])

        score_cross = mac(pattern, cross_filter)
        score_x = mac(pattern, x_filter)

        flattened_pattern = flatten_matrix(pattern)
        flattened_cross_filter = flatten_matrix(cross_filter)
        flattened_x_filter = flatten_matrix(x_filter)

        flat_score_cross = mac_1d(
            flattened_pattern,
            flattened_cross_filter,
        )
        flat_score_x = mac_1d(
            flattened_pattern,
            flattened_x_filter,
        )

        prediction = classify_scores(score_cross, score_x)
        is_correct = expected == prediction

        time_cross = measure_mac_time(pattern, cross_filter)
        time_x = measure_mac_time(pattern, x_filter)
        average_time = (time_cross + time_x) / 2

        if size not in execution_times:
            execution_times[size] = []

        execution_times[size].append(average_time)

        print(f"\n[{pattern_name}]")
        print(f"Cross MAC 점수: {score_cross}")
        print(f"X MAC 점수: {score_x}")
        print(f"1D Cross MAC 점수: {flat_score_cross}")
        print(f"1D X MAC 점수: {flat_score_x}")
        print(f"예상 결과: {expected}")
        print(f"판정 결과: {prediction}")
        print(f"평균 MAC 실행 시간: {average_time:.9f}초")

        if is_correct:
            print("결과: PASS")
            pass_count += 1
        else:
            print("결과: FAIL")
            fail_count += 1

    total_count = pass_count + fail_count

    print("\n=== 전체 실행 결과 ===")
    print(f"전체 패턴 수: {total_count}")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")

    print("\n=== 크기별 MAC 실행 시간 ===")

    for size in sorted(execution_times):
        times = execution_times[size]
        average_time = sum(times) / len(times)

        print(
            f"{size}x{size}: "
            f"{average_time:.9f}초 "
            f"(연산 요소 수: {size ** 2})"
        )


def main() -> None:
    """Runs the Mini NPU simulator."""
    try:
        print("1. 사용자 입력 모드")
        print("2. JSON 파일 모드")

        choice = input("모드를 선택하세요: ").strip()

        if choice == "1":
            run_user_input_mode()
        elif choice == "2":
            try:
                run_json_mode()
            except ValueError as error:
                print(f"오류: {error}")
        else:
            print("올바른 모드를 선택하세요.")

    except (KeyboardInterrupt, EOFError):
        print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()