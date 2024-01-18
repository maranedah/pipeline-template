from typing import Any, Callable, Literal, Sequence

from sklearn.utils._mocking import CheckingClassifier


class SklearnModelMock(CheckingClassifier):
    def __init__(
        self,
        *,
        check_y: Callable[..., Any] | None = None,
        check_y_params: dict | None = None,
        check_X: Callable[..., Any] | None = None,
        check_X_params: dict | None = None,
        methods_to_check: Sequence[str] | Literal["all"] = "all",
        foo_param: int = 0,
        expected_sample_weight: bool | None = None,
        expected_fit_params: Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            check_y=check_y,
            check_y_params=check_y_params,
            check_X=check_X,
            check_X_params=check_X_params,
            methods_to_check=methods_to_check,
            foo_param=foo_param,
            expected_sample_weight=expected_sample_weight,
            expected_fit_params=expected_fit_params,
        )
