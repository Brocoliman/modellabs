# Inference run
uv run python -m reasoninglab.infer 'In quadrilateral $ABCD$, $\angle B$ is a right angle, diagonal $\overline{AC}$ is perpendicular to $\overline{CD}$, $AB=18$, $BC=21$, and $CD=14$. Find the perimeter of $ABCD$. The answer is '


# Test: math500 k=1 vs k=4
uv run -m reasoninglab.scripts.eval --bench math500  \
    --template answer-only --max-tokens 5000 --temp 0.3 --p 0.85 --task-range 0 50
uv run -m reasoninglab.scripts.eval --bench math500  \
    --template answer-only --max-tokens 5000 --temp 0.3 --p 0.85 --k 4 --task-range 0 50

uv run -m reasoninglab.scripts.eval --bench mmlu  \
    --split dev --template vanilla --max-tokens 5000 --temp 0.3 --p 0.85 --task-range 0 10