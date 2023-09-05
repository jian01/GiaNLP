"""
Report utils module
"""

from typing import List, TYPE_CHECKING
from gianlp.types import SimpleTypeModels

if TYPE_CHECKING:
    from gianlp.models import BaseModel

MODEL_NAME_LENGTH = 21
SHAPE_LENGTH = 23
WEIGHTS_LENGTH = 9
CONNECTION_LENGTH = 20
LINE_FORMAT = "{model_name}|{input_shape}|{output_shape}|{trainable_weights}|{weights}|{connection}"


def _produce_line(
    model_name: str, input_shape: str, output_shape: str, trainable_weights: str, weights: str, connection: str
):
    """
    Creates a summary line

    :param model_name: the model name
    :param input_shape: the input shape
    :param output_shape: the output shape
    :param trainable_weights: the trainable weight amount
    :param weights: the weight amount
    :param connection: the connection to the model
    :return: a line string for using in summary
    """
    model_name = model_name.center(MODEL_NAME_LENGTH)[:MODEL_NAME_LENGTH]
    input_shape = input_shape.center(SHAPE_LENGTH)[:SHAPE_LENGTH]
    output_shape = output_shape.center(SHAPE_LENGTH)[:SHAPE_LENGTH]
    trainable_weights = trainable_weights.center(WEIGHTS_LENGTH)[:WEIGHTS_LENGTH]
    weights = weights.center(WEIGHTS_LENGTH)[:WEIGHTS_LENGTH]
    connection = connection.center(CONNECTION_LENGTH)[:CONNECTION_LENGTH]
    return LINE_FORMAT.format(
        model_name=model_name,
        input_shape=input_shape,
        output_shape=output_shape,
        trainable_weights=trainable_weights,
        weights=weights,
        connection=connection,
    )


def __model_finder(model: "BaseModel") -> SimpleTypeModels:
    """
    Finds all models chained to the one passed

    :param model: the initial model
    :return: a list of base models
    """
    models = [model]
    for m in model.inputs:
        models += __model_finder(m)
    return models


def __model_list_to_summary_string(models: List["BaseModel"]) -> str:
    """
    Given a list of chained models returns a string summarizing it

    :param models: the chained models
    :return: a summary string
    """
    out_lines = []
    out_lines.append(_produce_line("Model", "Inputs shape", "Output shape", "Trainable", "Total", "Connected to"))
    out_lines.append(_produce_line("", "", "", "weights", "weights", ""))
    out_lines.append("=" * len(out_lines[0]))
    for model in models:
        model_names = [repr(model)]
        output_shape = [str(model.outputs_shape)]
        trainable_weights = [str(model.trainable_weights_amount) if model.trainable_weights_amount is not None else "?"]
        weights = [str(model.weights_amount) if model.weights_amount is not None else "?"]
        if model.inputs.is_multi_text():
            connection = [f'"{name}": ' + repr(m) for name, ms in model.inputs.items() for m in ms]
        else:
            connection = [repr(m) for m in model.inputs]
        inputs_shape = (
            [str(inp) for inp in model.inputs_shape]
            if isinstance(model.inputs_shape, list)
            else [str(model.inputs_shape)]
        )
        line_length = max(len(connection), len(inputs_shape), 1)
        for _ in range(line_length):
            out_lines.append(
                _produce_line(
                    model_names.pop(0) if model_names else "",
                    inputs_shape.pop(0) if inputs_shape else "",
                    output_shape.pop(0) if output_shape else "",
                    trainable_weights.pop(0) if trainable_weights else "",
                    weights.pop(0) if weights else "",
                    connection.pop(0) if connection else "",
                )
            )
    out_lines.append("=" * len(out_lines[0]))

    out_lines.append(
        _produce_line(
            "",
            "",
            "",
            str(models[-1].trainable_weights_amount) if models[-1].trainable_weights_amount is not None else "?",
            str(models[-1].weights_amount) if models[-1].weights_amount is not None else "?",
            "",
        )
    )

    return "\n".join(out_lines)


def compute_model_summary(model: "BaseModel") -> str:
    """
    Computes a model summary as string

    :param model: the model to summary
    :return: the model summary
    """
    models = list(reversed(__model_finder(model)))
    seen_models = set()
    no_repetition_models = []
    for model in models:
        if repr(model) not in seen_models:
            no_repetition_models.append(model)
            seen_models.update([repr(model)])
    return __model_list_to_summary_string(no_repetition_models)
