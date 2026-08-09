EPSILON = 1e-9


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
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"

    if score_cross > score_x:
        return "Cross"

    return "X"


def main() -> None:
    """Runs a simple Mini NPU simulation."""
    cross_filter = [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ]

    x_filter = [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]

    pattern = [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]

    score_cross = mac(pattern, cross_filter)
    score_x = mac(pattern, x_filter)

    prediction = classify_scores(score_cross, score_x)

    print(f"Cross score: {score_cross}")
    print(f"X score: {score_x}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()