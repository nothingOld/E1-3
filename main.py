import json


EPSILON = 1e-9  # 1e-9 = 0.000000001


def load_data(file_path: str) -> dict:
    """Loads JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


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

    filters = data.get("filters")
    patterns = data.get("patterns")

    size_5_filters = filters.get("size_5")

    cross_filter = size_5_filters.get("cross")
    x_filter = size_5_filters.get("x")

    for pattern_name, pattern_data in patterns.items():
        if not pattern_name.startswith("size_5_"):
            continue

        pattern = pattern_data.get("input")
        expected = normalize_label(pattern_data.get("expected"))

        score_cross = mac(pattern, cross_filter)
        score_x = mac(pattern, x_filter)

        prediction = classify_scores(score_cross, score_x)

        is_correct = expected == prediction

        print(f"\n패턴: {pattern_name}")
        print(f"Cross MAC 점수: {score_cross}")
        print(f"X MAC 점수: {score_x}")
        print(f"예상 결과: {expected}")
        print(f"판정 결과: {prediction}")

        if is_correct:
            print("결과: PASS")
        else:
            print("결과: FAIL")


def main() -> None:
    """Runs the Mini NPU simulator."""
    print("1. 사용자 입력 모드")
    print("2. JSON 파일 모드")

    choice = input("모드를 선택하세요: ").strip()

    if choice == "1":
        run_user_input_mode()
    elif choice == "2":
        run_json_mode()
    else:
        print("올바른 모드를 선택하세요.")


if __name__ == "__main__":
    main()