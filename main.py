# 1e-9 = 0.000000001
EPSILON = 1e-9


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
    filter_matrix: list[list[float]]
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

    # 음수가 될 수도 있지만 두 값이 얼마나 떨어져 있는가 확인을 위해 절대값 사용
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"

    if score_cross > score_x:
        return "Cross"

    return "X"


def main() -> None:
    """Runs the Mini NPU simulator."""
    size = 3

    filter_a = read_matrix("필터 A", size)
    filter_b = read_matrix("필터 B", size)
    pattern = read_matrix("패턴", size)

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)

    prediction = classify_scores(score_a, score_b)

    print(f"필터 A MAC 점수: {score_a}")
    print(f"필터 B MAC 점수: {score_b}")
    print(f"판정 결과: {prediction}")

if __name__ == "__main__":
    main()