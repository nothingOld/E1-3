import json


EPSILON = 1e-9  # 1e-9 = 0.000000001


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

    if normalized_label == "cross":
        return "Cross"

    if normalized_label == "x":
        return "X"

    if normalized_label == "undecided":
        return "UNDECIDED"

    return label


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
    """Validates JSON data structure and values."""
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

    for pattern_name, pattern_data in patterns.items():
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

        if (
            len(name_parts) != 3
            or name_parts[0] != "size"
        ):
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
                f"'{pattern_name}'에 사용할 "
                f"'{size_key}' 필터가 없습니다."
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

    print(f"필터 A MAC 점수: {score_a}")
    print(f"필터 B MAC 점수: {score_b}")
    print(f"판정 결과: {prediction}")


def run_json_mode() -> None:
    """Runs the simulator with data loaded from a JSON file."""
    data = load_data("data.json")
    validate_data(data)

    filters = data["filters"]
    patterns = data["patterns"]

    pass_count = 0
    fail_count = 0

    for pattern_name, pattern_data in patterns.items():
        name_parts = pattern_name.split("_")
        size_key = f"{name_parts[0]}_{name_parts[1]}"

        filter_data = filters[size_key]

        cross_filter = filter_data["cross"]
        x_filter = filter_data["x"]

        pattern = pattern_data["input"]
        expected = normalize_label(pattern_data["expected"])

        score_cross = mac(pattern, cross_filter)
        score_x = mac(pattern, x_filter)

        prediction = classify_scores(score_cross, score_x)
        is_correct = expected == prediction

        print(f"\n[{pattern_name}]")
        print(f"Cross MAC 점수: {score_cross}")
        print(f"X MAC 점수: {score_x}")
        print(f"예상 결과: {expected}")
        print(f"판정 결과: {prediction}")

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


def main() -> None:
    """Runs the Mini NPU simulator."""
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


if __name__ == "__main__":
    main()