# epsilon 판정 추가
# 프로그램 전체에서 사용하기에 상수로 선언
EPSILON = 1e-9

# NxN 데이터 리스트 변수 저장 후 출력
cross_filter = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
]

x_filter = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
]
print(cross_filter[0][1])


# MAC 연산 구현
def mac(pattern, filter_matrix):
    total = 0.0

    for row in range(len(pattern)):
        for col in range(len(pattern[row])):
            total += pattern[row][col] * filter_matrix[row][col]

    return total


# 판정 함수 만들기
def classify_scores(score_cross: float, score_x: float) -> str:
    """Classifies a pattern based on two MAC scores."""
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"

    if score_cross > score_x:
        return "Cross"

    return "X"