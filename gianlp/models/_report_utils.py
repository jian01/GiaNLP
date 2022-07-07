"""
Report utils module
"""

from typing import List, Any

MODEL_NAME_LENGTH = 21
SHAPE_LENGTH = 23
WEIGHTS_LENGTH = 9
CONNECTION_LENGTH = 20
LINE_FORMAT = "{model_name}|{input_shape}|{output_shape}|{trainable_weights}|{weights}|{connection}"


def _produce_line(model_name: str, input_shape: str, output_shape: str, trainable_weights: str, weights: str,
                  connection: str):
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


def model_list_to_summary_string(models: List[Any]) -> str:
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
        inputs = model.inputs
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], tuple):
            inputs_shape = [str(m.outputs_shape) for _, ms in inputs for m in ms]
            connection = [f'"{name}": ' + repr(m) for name, ms in inputs for m in ms]
        else:
            inputs_shape = [str(m.outputs_shape) for m in inputs]
            connection = [repr(m) for m in inputs]
        if not inputs:
            inputs_shape = [str(inp) for inp in model.inputs_shape] if isinstance(model.inputs_shape, list) else [
                str(model.inputs_shape)]
        line_length = max(len(connection), len(inputs_shape), 1)
        for _ in range(line_length):
            out_lines.append(
                _produce_line(
                    model_names.pop(0) if model_names else "",
                    inputs_shape.pop(0) if inputs_shape else "",
                    output_shape.pop(0) if output_shape else "",
                    trainable_weights.pop(0) if trainable_weights else "",
                    weights.pop(0) if weights else "",
                    connection.pop(0) if inputs else "",
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
